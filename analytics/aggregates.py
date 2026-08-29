from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from utils.timeutils import utcnow
from typing import Any, Dict, List, Optional, Tuple

from database.models import Visit


def _filter_visits(
    visits: List[Visit],
    role: Optional[str] = None,
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
) -> List[Visit]:
    out = []
    for v in visits:
        if role is not None and v.role != role:
            continue
        if since is not None and v.entry_time < since:
            continue
        if until is not None and v.entry_time > until:
            continue
        out.append(v)
    return out


def hourly_visitors(
    db: Any, role: Optional[str] = "customer", day: Optional[date] = None
) -> Dict[int, int]:
    visits = _filter_visits(db.get_visits(), role=role)
    counts: Counter[int] = Counter()
    for v in visits:
        if day is not None and v.entry_time.date() != day:
            continue
        counts[v.entry_time.hour] += 1

    return {h: counts.get(h, 0) for h in range(24)}


def peak_hour(db: Any, role: Optional[str] = "customer", day: Optional[date] = None) -> Optional[int]:
    hourly = hourly_visitors(db, role=role, day=day)
    if not any(hourly.values()):
        return None
    return max(hourly, key=hourly.get)


def daily_visitors(
    db: Any, days: int = 30, role: Optional[str] = "customer"
) -> List[Tuple[date, int]]:
    today = utcnow().date()
    start = today - timedelta(days=days - 1)
    since = datetime.combine(start, datetime.min.time())
    visits = _filter_visits(db.get_visits(), role=role, since=since)

    per_day: Dict[date, set] = defaultdict(set)
    for v in visits:
        per_day[v.entry_time.date()].add(v.track_id)

    series: List[Tuple[date, int]] = []
    for i in range(days):
        d = start + timedelta(days=i)
        series.append((d, len(per_day.get(d, set()))))
    return series


def _period_visitors(
    db: Any, buckets: int, delta_days: int, role: Optional[str], label: str
) -> List[Tuple[str, int]]:
    today = utcnow().date()
    since = datetime.combine(today - timedelta(days=buckets * delta_days), datetime.min.time())
    visits = _filter_visits(db.get_visits(), role=role, since=since)

    per_bucket: Dict[str, set] = defaultdict(set)
    for v in visits:
        d = v.entry_time.date()
        if label == "week":
            iso = d.isocalendar()
            key = f"{iso[0]}-W{iso[1]:02d}"
        else:
            key = d.strftime("%Y-%m")
        per_bucket[key].add(v.track_id)
    return sorted((k, len(ids)) for k, ids in per_bucket.items())


def weekly_visitors(db: Any, weeks: int = 12, role: Optional[str] = "customer") -> List[Tuple[str, int]]:
    return _period_visitors(db, buckets=weeks, delta_days=7, role=role, label="week")


def monthly_visitors(db: Any, months: int = 12, role: Optional[str] = "customer") -> List[Tuple[str, int]]:
    return _period_visitors(db, buckets=months, delta_days=31, role=role, label="month")


@dataclass
class StayStats:
    count: int = 0
    mean: float = 0.0
    median: float = 0.0
    minimum: float = 0.0
    maximum: float = 0.0
    histogram: Dict[str, int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "count": self.count,
            "mean": round(self.mean, 1),
            "median": round(self.median, 1),
            "minimum": round(self.minimum, 1),
            "maximum": round(self.maximum, 1),
            "histogram": self.histogram or {},
        }


def stay_duration_stats(
    db: Any,
    role: Optional[str] = "customer",
    bins: Optional[List[Tuple[float, float, str]]] = None,
) -> StayStats:
    visits = _filter_visits(db.get_visits(), role=role)
    durations = sorted(
        v.duration_seconds for v in visits if v.duration_seconds is not None
    )
    if not durations:
        return StayStats(histogram={})

    n = len(durations)
    mean = sum(durations) / n
    median = (
        durations[n // 2]
        if n % 2
        else (durations[n // 2 - 1] + durations[n // 2]) / 2
    )

    if bins is None:
        bins = [
            (0, 60, "<1m"),
            (60, 300, "1-5m"),
            (300, 900, "5-15m"),
            (900, 1800, "15-30m"),
            (1800, 3600, "30-60m"),
            (3600, float("inf"), ">60m"),
        ]
    histogram: Dict[str, int] = {label: 0 for _, _, label in bins}
    for d in durations:
        for low, high, label in bins:
            if low <= d < high:
                histogram[label] += 1
                break

    return StayStats(
        count=n,
        mean=mean,
        median=median,
        minimum=durations[0],
        maximum=durations[-1],
        histogram=histogram,
    )


def occupancy_series(
    db: Any, session_id: Optional[str] = None, since: Optional[datetime] = None
) -> List[Tuple[datetime, int]]:
    snaps = db.get_snapshots(session_id=session_id, since=since)
    return [(s.timestamp, s.current_customers) for s in snaps]
