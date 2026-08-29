from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np

Point = Tuple[float, float]
BBox = Tuple[float, float, float, float]
Polygon = Sequence[Point]
Segment = Tuple[Point, Point]


def bbox_center(box: BBox) -> Point:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def bbox_bottom_center(box: BBox) -> Point:
    x1, _, x2, y2 = box
    return ((x1 + x2) / 2.0, y2)


def bbox_area(box: BBox) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def iou(box_a: BBox, box_b: BBox) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    if intersection <= 0.0:
        return 0.0

    union = bbox_area(box_a) + bbox_area(box_b) - intersection
    return intersection / union if union > 0.0 else 0.0


def euclidean(p: Point, q: Point) -> float:
    return float(np.hypot(p[0] - q[0], p[1] - q[1]))


def point_to_segment_distance(point: Point, segment: Segment) -> float:
    p = np.asarray(point, dtype=float)
    a = np.asarray(segment[0], dtype=float)
    b = np.asarray(segment[1], dtype=float)

    ab = b - a
    ab_len_sq = float(ab @ ab)
    if ab_len_sq == 0.0:
        return float(np.linalg.norm(p - a))
    t = float(np.clip((p - a) @ ab / ab_len_sq, 0.0, 1.0))
    projection = a + t * ab
    return float(np.linalg.norm(p - projection))


def point_in_polygon(point: Point, polygon: Polygon) -> bool:
    x, y = point
    verts = list(polygon)
    n = len(verts)
    if n < 3:
        return False

    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = verts[i]
        xj, yj = verts[j]
        intersects = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
        )
        if intersects:
            inside = not inside
        j = i
    return inside


def distance_to_polygon(point: Point, polygon: Polygon) -> float:
    verts = list(polygon)
    if len(verts) < 2:
        return float("inf")
    if len(verts) >= 3 and point_in_polygon(point, verts):
        return 0.0
    n = len(verts)
    best = float("inf")
    for i in range(n):
        edge = (verts[i], verts[(i + 1) % n])
        best = min(best, point_to_segment_distance(point, edge))
    return best


def polygon_area(polygon: Polygon) -> float:
    verts = np.asarray(polygon, dtype=float)
    if len(verts) < 3:
        return 0.0
    x = verts[:, 0]
    y = verts[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _orientation(a: Point, b: Point, c: Point) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def side_of_line(point: Point, line: Segment) -> int:
    value = _orientation(line[0], line[1], point)
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def segments_intersect(seg1: Segment, seg2: Segment) -> bool:
    p1, p2 = seg1
    p3, p4 = seg2
    d1 = _orientation(p3, p4, p1)
    d2 = _orientation(p3, p4, p2)
    d3 = _orientation(p1, p2, p3)
    d4 = _orientation(p1, p2, p4)

    if ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)):
        return True

    def _on_segment(a: Point, b: Point, c: Point) -> bool:
        return (
            min(a[0], b[0]) <= c[0] <= max(a[0], b[0])
            and min(a[1], b[1]) <= c[1] <= max(a[1], b[1])
        )

    if d1 == 0 and _on_segment(p3, p4, p1):
        return True
    if d2 == 0 and _on_segment(p3, p4, p2):
        return True
    if d3 == 0 and _on_segment(p1, p2, p3):
        return True
    if d4 == 0 and _on_segment(p1, p2, p4):
        return True
    return False


def clamp_point(point: Point, width: int, height: int) -> Point:
    x = min(max(point[0], 0.0), float(width))
    y = min(max(point[1], 0.0), float(height))
    return (x, y)


def normalize_polygon(polygon: Iterable[Sequence[float]]) -> list[Point]:
    return [(float(p[0]), float(p[1])) for p in polygon]
