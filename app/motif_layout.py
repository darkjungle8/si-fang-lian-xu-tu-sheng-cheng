"""圖章分組、背景版、環帶／整張重排。

清邊補花不再把殘片吸到 wrap 線上，而是以群組為單位清掉邊緣帶，
再用原稿圖章在環面上做藍噪聲放置。像素全部來自原稿。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt


@dataclass
class MotifGroup:
    """一組鄰近連通塊（腳印的掌＋趾、雪人＋帽）當成一枚圖章。"""

    ids: tuple[int, ...]
    area: int
    cy: float
    cx: float
    left: int
    top: int
    width: int
    height: int
    touching: bool
    rgb: tuple[float, float, float] = (0.0, 0.0, 0.0)


def group_components_into_motifs(
    fg: np.ndarray,
    min_area: int = 40,
    *,
    edge: np.ndarray | None = None,
) -> tuple[np.ndarray, list[MotifGroup], np.ndarray, np.ndarray, np.ndarray]:
    """
    以「間隙 ≤ 0.35×√主圖章面積」把鄰近連通塊合成一枚。

    回傳 (群組標籤圖 0=地, groups, 原始 labels, stats, centroids)。
    若閉合過度把圖章黏成一片，退回逐塊分組。
    """
    fg_u8 = fg.astype(np.uint8)
    h, w = fg_u8.shape[:2]
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(
        fg_u8, connectivity=8
    )
    empty: list[MotifGroup] = []
    if n <= 1:
        return np.zeros((h, w), dtype=np.int32), empty, labels, stats, centroids

    min_area = max(40, int(min_area))
    areas = [
        int(stats[i, cv2.CC_STAT_AREA])
        for i in range(1, n)
        if int(stats[i, cv2.CC_STAT_AREA]) >= min_area
    ]
    if len(areas) < 2:
        gmap, groups = _identity_groups(
            labels, stats, centroids, n, min_area, edge
        )
        return gmap, groups, labels, stats, centroids

    hero = float(np.percentile(np.asarray(areas, dtype=np.float64), 85))
    k = int(round(0.28 * np.sqrt(max(hero, 1.0))))
    k = max(3, k)
    if k % 2 == 0:
        k += 1
    large = [
        i
        for i in range(1, n)
        if int(stats[i, cv2.CC_STAT_AREA]) >= max(min_area, hero * 0.45)
    ]
    if len(large) >= 4:
        pts = [
            (float(centroids[i][1]), float(centroids[i][0])) for i in large
        ]
        nn = _median_nn(pts, h, w)
        if nn >= 16.0:
            cap = max(3, int(round(0.40 * nn)))
            if cap % 2 == 0:
                cap += 1
            k = min(k, cap)

    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    closed = cv2.morphologyEx(fg_u8, cv2.MORPH_CLOSE, ker)
    n2, lab2, _, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    valid = sum(
        1
        for i in range(1, n)
        if int(stats[i, cv2.CC_STAT_AREA]) >= min_area
    )
    n_closed = max(0, n2 - 1)
    if valid >= 8 and n_closed < max(2, int(round(0.35 * valid))):
        gmap, groups = _identity_groups(
            labels, stats, centroids, n, min_area, edge
        )
        return gmap, groups, labels, stats, centroids

    # 每個原始連通塊投給閉合標籤
    vote_of = np.zeros(n, dtype=np.int32)
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x0 = int(stats[i, cv2.CC_STAT_LEFT])
        y0 = int(stats[i, cv2.CC_STAT_TOP])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        roi = labels[y0 : y0 + bh, x0 : x0 + bw] == i
        votes = lab2[y0 : y0 + bh, x0 : x0 + bw][roi]
        if votes.size:
            vote_of[i] = int(np.bincount(votes).argmax())

    buckets: dict[int, list[int]] = {}
    for i in range(1, n):
        if int(stats[i, cv2.CC_STAT_AREA]) < min_area:
            continue
        key = int(vote_of[i])
        if key <= 0:
            key = 1_000_000 + i
        buckets.setdefault(key, []).append(i)

    gmap = np.zeros((h, w), dtype=np.int32)
    groups = []
    gid = 1
    touch_lut = None
    if edge is not None:
        touch_lut = np.zeros(n, dtype=bool)
        for t in np.unique(labels[edge]):
            ti = int(t)
            if 0 < ti < n:
                touch_lut[ti] = True
    for ids in buckets.values():
        union = np.zeros((h, w), dtype=bool)
        area = 0
        sy = sx = 0.0
        touching = False
        left = w
        top = h
        right = 0
        bot = 0
        for i in ids:
            x0 = int(stats[i, cv2.CC_STAT_LEFT])
            y0 = int(stats[i, cv2.CC_STAT_TOP])
            bw = int(stats[i, cv2.CC_STAT_WIDTH])
            bh = int(stats[i, cv2.CC_STAT_HEIGHT])
            roi = labels[y0 : y0 + bh, x0 : x0 + bw] == i
            union[y0 : y0 + bh, x0 : x0 + bw] |= roi
            a = int(stats[i, cv2.CC_STAT_AREA])
            area += a
            sy += float(centroids[i][1]) * a
            sx += float(centroids[i][0]) * a
            left = min(left, x0)
            top = min(top, y0)
            right = max(right, x0 + bw)
            bot = max(bot, y0 + bh)
            if touch_lut is not None and touch_lut[i]:
                touching = True
        if area <= 0:
            continue
        gmap[union] = gid
        groups.append(
            MotifGroup(
                ids=tuple(ids),
                area=area,
                cy=sy / area,
                cx=sx / area,
                left=left,
                top=top,
                width=right - left,
                height=bot - top,
                touching=touching,
            )
        )
        gid += 1
    return gmap, groups, labels, stats, centroids


def _identity_groups(
    labels: np.ndarray,
    stats: np.ndarray,
    centroids: np.ndarray,
    n: int,
    min_area: int,
    edge: np.ndarray | None,
) -> tuple[np.ndarray, list[MotifGroup]]:
    h, w = labels.shape[:2]
    gmap = np.zeros((h, w), dtype=np.int32)
    groups: list[MotifGroup] = []
    touch = set()
    if edge is not None:
        touch = {int(i) for i in np.unique(labels[edge]) if 0 < int(i) < n}
    gid = 1
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x0 = int(stats[i, cv2.CC_STAT_LEFT])
        y0 = int(stats[i, cv2.CC_STAT_TOP])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        gmap[y0 : y0 + bh, x0 : x0 + bw] = np.where(
            labels[y0 : y0 + bh, x0 : x0 + bw] == i, gid, gmap[y0 : y0 + bh, x0 : x0 + bw]
        )
        groups.append(
            MotifGroup(
                ids=(i,),
                area=area,
                cy=float(centroids[i][1]),
                cx=float(centroids[i][0]),
                left=x0,
                top=y0,
                width=bw,
                height=bh,
                touching=i in touch,
            )
        )
        gid += 1
    return gmap, groups


def _median_nn(
    pts: list[tuple[float, float]], h: int, w: int
) -> float:
    if len(pts) < 2:
        return 0.0
    dists: list[float] = []
    for i, (y, x) in enumerate(pts):
        best = 1e18
        for j, (y2, x2) in enumerate(pts):
            if i == j:
                continue
            dy = min(abs(y - y2), h - abs(y - y2))
            dx = min(abs(x - x2), w - abs(x - x2))
            best = min(best, float(np.hypot(dy, dx)))
        dists.append(best)
    return float(np.median(np.asarray(dists, dtype=np.float64)))


def group_stamp_from_roi(arr: np.ndarray, gmap: np.ndarray, gid: int, g: MotifGroup):
    from app.processor import _component_to_stamp_from_roi

    y0, x0 = g.top, g.left
    roi = (gmap[y0 : y0 + g.height, x0 : x0 + g.width] == gid).astype(np.uint8)
    if not roi.any():
        return None
    return _component_to_stamp_from_roi(arr, roi, y0, x0)


def attach_group_colors(
    arr: np.ndarray,
    labels: np.ndarray,
    stats: np.ndarray,
    groups: list[MotifGroup],
) -> None:
    from app.processor import _component_mean_rgb

    for g in groups:
        acc = np.zeros(3, dtype=np.float64)
        wsum = 0.0
        for i in g.ids:
            rgb = np.asarray(_component_mean_rgb(arr, labels, stats, int(i)))
            a = float(stats[int(i), cv2.CC_STAT_AREA])
            acc += rgb * a
            wsum += a
        if wsum > 0:
            m = acc / wsum
            g.rgb = (float(m[0]), float(m[1]), float(m[2]))


def background_plate(arr: np.ndarray, is_fg: np.ndarray) -> np.ndarray:
    """
    用環面最近的背景像素填前景位置。像素仍是原稿背景，不是平色。
    """
    is_fg = is_fg.astype(bool)
    h, w = arr.shape[:2]
    if not is_fg.any() or is_fg.all():
        return arr.copy()
    # 大圖不做 3×3 展開，避免記憶體爆掉
    if h * w > 4_000_000:
        _, inds = distance_transform_edt(is_fg, return_indices=True)
        iy, ix = inds[0], inds[1]
        return arr[iy, ix]
    reps = (3, 3, 1) if arr.ndim == 3 else (3, 3)
    a3 = np.tile(arr, reps)
    fg3 = np.tile(is_fg.astype(bool), (3, 3))
    _, inds = distance_transform_edt(fg3, return_indices=True)
    iy = np.clip(inds[0], 0, a3.shape[0] - 1)
    ix = np.clip(inds[1], 0, a3.shape[1] - 1)
    filled = a3[iy, ix]
    if filled.ndim == 3 and filled.shape[2] != arr.shape[2]:
        filled = filled[:, :, : arr.shape[2]]
    return filled[h : 2 * h, w : 2 * w].copy()


def fill_kill_with_plate(
    canvas: np.ndarray, kill: np.ndarray, plate: np.ndarray
) -> np.ndarray:
    out = canvas
    if kill.any():
        out[kill] = plate[kill]
    return out


@dataclass
class LayoutStats:
    d_nn: float
    d_min: float
    density: float
    areas: np.ndarray
    span: float
    colors: list[tuple[float, float, float]] = field(default_factory=list)


def layout_stats_from_groups(
    groups: list[MotifGroup],
    h: int,
    w: int,
    *,
    interior_only: bool = True,
) -> LayoutStats:
    use = [g for g in groups if (not interior_only) or (not g.touching)]
    if len(use) < 2:
        use = list(groups)
    if not use:
        return LayoutStats(
            d_nn=40.0,
            d_min=24.0,
            density=0.0,
            areas=np.array([200.0]),
            span=24.0,
        )
    pts = [(g.cy, g.cx) for g in use]
    nn = _median_nn(pts, h, w)
    dists: list[float] = []
    for i, (y, x) in enumerate(pts):
        best = 1e18
        for j, (y2, x2) in enumerate(pts):
            if i == j:
                continue
            dy = min(abs(y - y2), h - abs(y - y2))
            dx = min(abs(x - x2), w - abs(x - x2))
            best = min(best, float(np.hypot(dy, dx)))
        dists.append(best)
    d_min = float(np.percentile(np.asarray(dists, dtype=np.float64), 10))
    if not np.isfinite(nn) or nn < 8:
        nn = 40.0
    if not np.isfinite(d_min) or d_min < 6:
        d_min = max(8.0, 0.55 * nn)
    areas = np.array([g.area for g in use], dtype=np.float64)
    spans = [float(max(g.width, g.height, 8)) for g in use]
    span = float(np.median(np.asarray(spans, dtype=np.float64)))
    if areas.size >= 4:
        cut = float(np.percentile(areas, 70))
        big = [s for s, a in zip(spans, areas) if float(a) >= cut]
        if big:
            main = float(np.median(np.asarray(big, dtype=np.float64)))
            # 碎點把中位數拉到雪人／冬青的一半以下時，環帶才跟著主圖章加寬。
            if main > span * 2.0:
                span = main
    return LayoutStats(
        d_nn=float(nn),
        d_min=float(d_min),
        density=float(len(use)) / float(max(h * w, 1)),
        areas=areas,
        span=span,
        colors=[g.rgb for g in use],
    )


def _in_ring(cy: float, cx: float, h: int, w: int, r: float) -> bool:
    return min(cx, w - cx, cy, h - cy) <= r


def _sits_on_wrap_without_crossing(motif, cy: float, cx: float, h: int, w: int) -> bool:
    """圖章碰到畫框卻沒跨到對邊：wrap_cut 會當成切圖。"""
    top = int(round(cy - motif.cy))
    left = int(round(cx - motif.cx))
    ys, xs = np.where(motif.mask)
    if len(ys) == 0:
        return False
    y0 = top + int(ys.min())
    y1 = top + int(ys.max())
    x0 = left + int(xs.min())
    x1 = left + int(xs.max())
    wraps_v = y0 < 0 or y1 >= h
    wraps_h = x0 < 0 or x1 >= w
    band = max(2, min(8, min(h, w) // 80))
    if (y0 < band or y1 >= h - band) and not wraps_v:
        return True
    if (x0 < band or x1 >= w - band) and not wraps_h:
        return True
    return False


def _place_ok(
    motif,
    cy: float,
    cx: float,
    occupied: np.ndarray,
    cents: list[tuple[float, float]],
    d_min: float,
    h: int,
    w: int,
) -> bool:
    from app.processor import _overlap_ratio, _torus_distance

    if _sits_on_wrap_without_crossing(motif, cy, cx, h, w):
        return False
    for ky, kx in cents:
        if _torus_distance(cy, cx, ky, kx, h, w) < 0.85 * d_min:
            return False
    if _overlap_ratio(occupied, motif, cy, cx, occupied) > 0.03:
        return False
    return True


def dart_throw_motifs(
    canvas: np.ndarray,
    occupied: np.ndarray,
    templates: list,
    stats: LayoutStats,
    rng: np.random.Generator,
    *,
    ring: float | None,
    target: int,
    max_tries: int = 0,
) -> int:
    """在環面（或環帶）放置圖章。回傳成功件數。"""
    from app.processor import _pick_motif, stamp_motif_wrapped

    h, w = canvas.shape[:2]
    if target <= 0 or not templates:
        return 0
    if max_tries <= 0:
        max_tries = max(80, min(8000, target * 60))
    placed = 0
    cents: list[tuple[float, float]] = []
    # 既有佔用的質心（粗估）
    n, _, st, ct = cv2.connectedComponentsWithStats(
        occupied.astype(np.uint8), connectivity=8
    )
    for i in range(1, n):
        if int(st[i, cv2.CC_STAT_AREA]) >= 40:
            cents.append((float(ct[i][1]), float(ct[i][0])))
    d_need = max(8.0, 0.85 * stats.d_min)
    areas = stats.areas if stats.areas.size else np.array([templates[0].area])
    colors = stats.colors or [None]
    for _ in range(max_tries):
        if placed >= target:
            break
        if ring is None:
            cy = float(rng.uniform(0.0, h))
            cx = float(rng.uniform(0.0, w))
        else:
            cy = float(rng.uniform(0.0, h))
            cx = float(rng.uniform(0.0, w))
            if not _in_ring(cy, cx, h, w, ring):
                continue
        target_area = int(rng.choice(areas))
        color = colors[int(rng.integers(0, len(colors)))] if colors else None
        motif = _pick_motif(templates, target_area, rng, target_rgb=color)
        if not _place_ok(motif, cy, cx, occupied, cents, d_need, h, w):
            continue
        stamp_motif_wrapped(canvas, motif, cy, cx)
        my, mx = np.where(motif.mask)
        if len(my):
            top = int(round(cy - motif.cy))
            left = int(round(cx - motif.cx))
            occupied[(top + my) % h, (left + mx) % w] = True
        cents.append((cy, cx))
        placed += 1
    return placed


def relayout_ring(
    cleaned: np.ndarray,
    templates: list,
    stats: LayoutStats,
    ring: float,
    seed: int,
    occupied: np.ndarray | None = None,
    *,
    full: bool = False,
) -> np.ndarray:
    """環帶（或整張）藍噪聲補花。"""
    out = cleaned.copy()
    h, w = out.shape[:2]
    occ = occupied.astype(bool) if occupied is not None else np.zeros((h, w), dtype=bool)
    rng = np.random.default_rng(seed)
    if full:
        ring_area = float(h * w)
        use_ring = None
    else:
        band = np.zeros((h, w), dtype=bool)
        r = max(1, int(round(ring)))
        band[:r, :] = True
        band[h - r :, :] = True
        band[:, :r] = True
        band[:, w - r :] = True
        ring_area = float(np.count_nonzero(band))
        use_ring = float(r)
    target = int(round(stats.density * ring_area))
    if target < 1 and stats.density > 0:
        target = 1
    lo = max(1, int(round(target * 0.90))) if target else 0
    hi = max(lo, int(round(target * 1.10))) if target else 0
    dart_throw_motifs(
        out,
        occ,
        templates,
        stats,
        rng,
        ring=use_ring,
        target=hi,
    )
    _place_along_wrap(out, occ, templates, stats, rng)
    return out


def _place_along_wrap(
    canvas: np.ndarray,
    occupied: np.ndarray,
    templates: list,
    stats: LayoutStats,
    rng: np.random.Generator,
) -> int:
    """沿四條 wrap 線再補一圈，確保有圖章跨縫，而不是全部停在環帶內側。"""
    from app.processor import _pick_motif

    h, w = canvas.shape[:2]
    step = max(12.0, stats.d_nn)
    if step <= 0 or not templates:
        return 0
    centers: list[tuple[float, float]] = []
    t = 0.0
    while t < h:
        centers.append((t, 0.0))
        t += step
    t = 0.0
    while t < w:
        centers.append((0.0, t))
        t += step
    order = rng.permutation(len(centers))
    centers = [centers[int(i)] for i in order]
    areas = stats.areas if stats.areas.size else np.array([templates[0].area])
    colors = stats.colors or [None]
    d_need = max(8.0, 0.85 * stats.d_min)
    cents: list[tuple[float, float]] = []
    n, _, st, ct = cv2.connectedComponentsWithStats(
        occupied.astype(np.uint8), connectivity=8
    )
    for i in range(1, n):
        if int(st[i, cv2.CC_STAT_AREA]) >= 40:
            cents.append((float(ct[i][1]), float(ct[i][0])))
    placed = 0
    for cy, cx in centers:
        target_area = int(rng.choice(areas))
        color = colors[int(rng.integers(0, len(colors)))] if colors else None
        motif = _pick_motif(templates, target_area, rng, target_rgb=color)
        if not _place_ok(motif, cy, cx, occupied, cents, d_need, h, w):
            continue
        from app.processor import stamp_motif_wrapped

        stamp_motif_wrapped(canvas, motif, cy, cx)
        my, mx = np.where(motif.mask)
        if len(my):
            top = int(round(cy - motif.cy))
            left = int(round(cx - motif.cx))
            occupied[(top + my) % h, (left + mx) % w] = True
        cents.append((cy, cx))
        placed += 1
    return placed


def ring_fill_ratio(
    arr: np.ndarray,
    bg: Sequence[int],
    threshold: float,
    ring: float,
    stats: LayoutStats,
) -> float:
    from app.processor import _stamp_foreground

    h, w = arr.shape[:2]
    r = max(1, int(round(ring)))
    band = np.zeros((h, w), dtype=bool)
    band[:r, :] = True
    band[h - r :, :] = True
    band[:, :r] = True
    band[:, w - r :] = True
    fg = _stamp_foreground(arr, bg, threshold).astype(bool)
    n, _, st, _ = cv2.connectedComponentsWithStats(fg.astype(np.uint8), 8)
    count = 0
    for i in range(1, n):
        if int(st[i, cv2.CC_STAT_AREA]) < 40:
            continue
        # 質心在環帶內或外接框跨環帶都算
        cy = st[i, cv2.CC_STAT_TOP] + st[i, cv2.CC_STAT_HEIGHT] / 2.0
        cx = st[i, cv2.CC_STAT_LEFT] + st[i, cv2.CC_STAT_WIDTH] / 2.0
        if _in_ring(float(cy), float(cx), h, w, r):
            count += 1
    ring_area = float(np.count_nonzero(band))
    expect = stats.density * ring_area
    if expect < 1e-6:
        return 1.0
    return float(count) / expect
