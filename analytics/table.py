from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from analytics.base import BaseAnalyzer
from config import Config, get_config
from tracking.track_state import TrackFrameResult
from utils.geometry import Polygon, distance_to_polygon
from utils.logger import get_logger

logger = get_logger(__name__)

OCCUPIED = "occupied"
EMPTY = "empty"
RESERVED = "reserved"


@dataclass
class TableState:
    name: str
    polygon: Polygon
    reserved: bool = False
    zone_id: Optional[int] = None
    occupied: bool = False
    total_occupied_seconds: float = 0.0
    occupant_ids: set = field(default_factory=set)
    _present_since: Optional[datetime] = None

    @property
    def status(self) -> str:
        if self.occupied:
            return OCCUPIED
        if self.reserved:
            return RESERVED
        return EMPTY


@dataclass
class TableMetrics:
    total_tables: int = 0
    occupied_tables: int = 0
    empty_tables: int = 0
    reserved_tables: int = 0
    available_tables: int = 0
    occupancy_percentage: float = 0.0
    avg_table_usage_seconds: float = 0.0
    per_table: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TableOccupancy(BaseAnalyzer):
    def __init__(
        self,
        config: Optional[Config] = None,
        db: Optional[Any] = None,
        tables: Optional[List[TableState]] = None,
        role: str = "customer",
    ) -> None:
        self.config = config or get_config()
        self.acfg = self.config.analytics
        self.db = db
        self.role = role
        self._tables: List[TableState] = tables or []
        if not self._tables and db is not None:
            self.reload_tables()
        self._utilization: List[tuple] = []
        self._last_ts: Optional[datetime] = None


    def reload_tables(self) -> None:
        if self.db is None:
            return
        zones = self.db.get_zones(zone_type="table")
        self._tables = [
            TableState(
                name=z.name,
                polygon=[(float(x), float(y)) for x, y in z.points],
                reserved=bool(z.reserved),
                zone_id=z.id,
            )
            for z in zones
        ]
        logger.info("Loaded %d table(s).", len(self._tables))

    def set_reserved(self, name: str, reserved: bool) -> None:
        for table in self._tables:
            if table.name == name:
                table.reserved = reserved
                if self.db is not None:
                    self.db.upsert_zone(
                        table.name, "table", list(table.polygon), reserved=reserved
                    )
                return

    @property
    def tables(self) -> List[TableState]:
        return self._tables


    def _near(self, anchor, polygon: Polygon) -> bool:
        return distance_to_polygon(anchor, polygon) <= self.acfg.table_proximity_px

    def update(self, track_result: TrackFrameResult) -> None:
        ts = track_result.timestamp
        dt = (ts - self._last_ts).total_seconds() if self._last_ts else 0.0
        customers = [
            t for t in track_result.active_tracks
            if t.role == self.role and t.current_anchor is not None
        ]

        for table in self._tables:
            occupants = {
                t.track_id for t in customers if self._near(t.current_anchor, table.polygon)
            }
            table.occupant_ids = occupants

            if occupants:
                if table._present_since is None:
                    table._present_since = ts
                dwell = (ts - table._present_since).total_seconds()
                table.occupied = dwell >= self.acfg.table_occupied_seconds
            else:
                table._present_since = None
                table.occupied = False

            if table.occupied and dt > 0:
                table.total_occupied_seconds += dt

        occupied_count = sum(1 for t in self._tables if t.occupied)
        self._utilization.append((ts, occupied_count))
        self._last_ts = ts


    def compute(self) -> TableMetrics:
        total = len(self._tables)
        occupied = sum(1 for t in self._tables if t.occupied)
        reserved = sum(1 for t in self._tables if t.reserved and not t.occupied)
        empty = total - occupied - reserved
        available = sum(1 for t in self._tables if not t.occupied and not t.reserved)
        usages = [t.total_occupied_seconds for t in self._tables]
        per_table = [
            {
                "name": t.name,
                "status": t.status,
                "occupants": len(t.occupant_ids),
                "occupied_seconds": round(t.total_occupied_seconds, 1),
                "reserved": t.reserved,
            }
            for t in self._tables
        ]
        return TableMetrics(
            total_tables=total,
            occupied_tables=occupied,
            empty_tables=empty,
            reserved_tables=reserved,
            available_tables=available,
            occupancy_percentage=round((occupied / total * 100.0), 1) if total else 0.0,
            avg_table_usage_seconds=round(sum(usages) / len(usages), 1) if usages else 0.0,
            per_table=per_table,
        )

    def metrics(self) -> Dict[str, Any]:
        return self.compute().to_dict()

    def utilization_over_time(self) -> List[tuple]:
        return list(self._utilization)

    def reset(self) -> None:
        for table in self._tables:
            table.occupied = False
            table.total_occupied_seconds = 0.0
            table.occupant_ids = set()
            table._present_since = None
        self._utilization = []
        self._last_ts = None
        logger.debug("TableOccupancy reset.")
