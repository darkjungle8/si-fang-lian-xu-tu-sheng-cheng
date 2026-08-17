"""規則點綴：質心晶格構造裁切（縫落空隙），腐蝕切開硬門禁，禁止邊緣相關勝出。"""

from __future__ import annotations

import time
from typing import Callable, Sequence

import cv2
import numpy as np
from PIL import Image

from app.color_utils import color_distance

# 超過此邊長時，晶格搜尋改在縮圖上做，再映射回原圖裁切（不放寬接縫標準）
_DISCRETE_ANALYSIS_MAX_SIDE = 2048
_DISCRETE_FULLRES_TRIGGER = 2800


def _downscale_for_discrete_analysis(
    arr: np.ndarray,
    max_side: int = _DISCRETE_ANALYSIS_MAX_SIDE,
) -> tuple[np.ndarray, float]:
    """回傳 (縮圖, full/small 比例)。"""
    h, w = arr.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return arr, 1.0
    scale = m / float(max_side)
    nw = max(64, int(round(w / scale)))
    nh = max(64, int(round(h / scale)))
    small = np.asarray(
        Image.fromarray(arr).resize((nw, nh), Image.Resampling.BILINEAR)
    )
    return small, float(w) / float(small.shape[1])


def _map_lattice_box_to_full(
    dx: int,
    dy: int,
    cw: int,
    ch: int,
    px: int,
    py: int,
    scale: float,
    full_h: int,
    full_w: int,
) -> tuple[int, int, int, int, int, int]:
    """把縮圖裁切框映射回原圖，並把尺寸 snap 到整數倍週期。"""
    px_f = max(8, int(round(px * scale)))
    py_f = max(8, int(round(py * scale)))
    dx_f = int(round(dx * scale))
    dy_f = int(round(dy * scale))
    cw_f = int(round(cw * scale))
    ch_f = int(round(ch * scale))
    if px_f > 0:
        cw_f = max(px_f, int(round(cw_f / px_f)) * px_f)
    if py_f > 0:
        ch_f = max(py_f, int(round(ch_f / py_f)) * py_f)
    cw_f = min(cw_f, full_w)
    ch_f = min(ch_f, full_h)
    if px_f > 0:
        cw_f = max(px_f, (cw_f // px_f) * px_f)
    if py_f > 0:
        ch_f = max(py_f, (ch_f // py_f) * py_f)
    dx_f = min(max(0, dx_f), max(0, full_w - cw_f))
    dy_f = min(max(0, dy_f), max(0, full_h - ch_f))
    return dx_f, dy_f, cw_f, ch_f, px_f, py_f


def _fg_mask(arr: np.ndarray, bg: Sequence[int], threshold: float) -> np.ndarray:
    return color_distance(arr, bg) > threshold


def _unique_axis_count(coords: list[float], min_sep: float) -> int:
    if not coords:
        return 0
    pts = sorted(coords)
    uniq = [pts[0]]
    for p in pts[1:]:
        if p - uniq[-1] >= min_sep:
            uniq.append(p)
    return len(uniq)


def _merge_nearby_centroids(
    cents: list[tuple[float, float, int]],
    min_dist: float,
) -> list[tuple[float, float, int]]:
    """合併過近質心（花圈碎裂常留下 2～5px 雙心）。"""
    if len(cents) < 2 or min_dist < 2:
        return list(cents)
    order = sorted(range(len(cents)), key=lambda i: -cents[i][2])
    ys = np.array([cents[i][0] for i in order], dtype=np.float64)
    xs = np.array([cents[i][1] for i in order], dtype=np.float64)
    areas = np.array([cents[i][2] for i in order], dtype=np.float64)
    alive = np.ones(len(order), dtype=bool)
    out: list[tuple[float, float, int]] = []
    md2 = float(min_dist) * float(min_dist)
    for i in range(len(order)):
        if not alive[i]:
            continue
        mask = alive.copy()
        mask[i] = False
        if np.any(mask):
            dy = ys[mask] - ys[i]
            dx = xs[mask] - xs[i]
            near = (dy * dy + dx * dx) <= md2
            idx = np.flatnonzero(mask)[near]
        else:
            idx = np.empty(0, dtype=np.int64)
        wsum = float(areas[i])
        y = float(ys[i]) * wsum
        x = float(xs[i]) * wsum
        for j in idx:
            w = float(areas[j])
            y += float(ys[j]) * w
            x += float(xs[j]) * w
            wsum += w
            alive[j] = False
        alive[i] = False
        out.append((y / wsum, x / wsum, int(round(wsum))))
    return out


def _dual_lattice_quality(large: list[tuple[float, float]]) -> tuple[float, int]:
    """
    大花晶格乾淨度：近重複越少、行列越清楚越好。
    評分前先合併過近點，避免碎裂雙心拖垮分數。
    """
    if len(large) < 4:
        return -1e9, len(large)
    cents = [(cy, cx, 1000) for cy, cx in large]
    # 先用寬鬆距離合併；若 NN 仍很小再併一次
    merged = _merge_nearby_centroids(cents, 28.0)
    pts = [(cy, cx) for cy, cx, _ in merged]
    n = len(pts)
    if n < 4:
        return -1e9, n
    nn = _nn_median_spacing(pts)
    sep = max(40.0, nn * 0.35 if nn >= 16 else 40.0)
    xs = [cx for _, cx in pts]
    ys = [cy for _, cy in pts]
    ux = _unique_axis_count(xs, sep)
    uy = _unique_axis_count(ys, sep)
    expected = max(ux * max(uy, 1) * 0.5, float(max(ux, uy)), 1.0)
    excess = max(0.0, n / expected - 1.2)
    # 中位面積代理：合併後點數適中加分
    score = ux * 3.0 + uy * 3.0 - excess * 20.0 + min(n, 30) * 0.4
    return score, n


def _fg_mask_merged(
    arr: np.ndarray,
    bg: Sequence[int],
    threshold: float,
    *,
    max_close: int = 21,
) -> np.ndarray:
    """
    閉運算合併碎裂前景（花圈線條／水彩筆觸常被切成多塊）。
    僅用於質心／週期估測；切開硬門禁仍用原始 _fg_mask。
    花圈類會碎成數十塊「假大花」，不可因 raw 已偵測到雙尺度就提前返回。
    """
    raw = _fg_mask(arr, bg, threshold)
    med0 = _interior_median_motif_area(raw)
    n0, _, stats0, _ = cv2.connectedComponentsWithStats(
        raw.astype(np.uint8), connectivity=4
    )
    n_big = sum(
        1
        for i in range(1, n0)
        if int(stats0[i, cv2.CC_STAT_AREA]) >= max(40, int(med0 * 0.08) if med0 else 40)
    )
    # 已經是大塊或數量不多 → 不必合併
    if med0 >= 800 or n_big < 40:
        return raw

    best = raw
    best_score = -1e18
    # 碎裂點綴（很多中小塊）：raw 雙尺度常是「假大花」，不可直接勝出
    force_close = med0 < 250 and n_big >= 80
    raw_cents = _fg_centroids(raw, min_area=max(40, int(med0 * 0.08) if med0 else 40))
    raw_dual = _dual_scale_groups(raw_cents)
    if raw_dual is not None and not force_close:
        best_score, _ = _dual_lattice_quality(raw_dual[0])

    for k in (5, 9, 15, 21):
        if k > max_close:
            break
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        closed = cv2.morphologyEx(
            raw.astype(np.uint8), cv2.MORPH_CLOSE, ker
        ).astype(bool)
        med = _interior_median_motif_area(closed)
        cents = _fg_centroids(
            closed, min_area=max(40, int(med * 0.08) if med else 40), skip_edge=True
        )
        if len(cents) < 8:
            cents = _fg_centroids(
                closed, min_area=max(40, int(med * 0.08) if med else 40)
            )
        dual = _dual_scale_groups(cents)
        if dual is not None and len(dual[0]) >= 8:
            score, _ = _dual_lattice_quality(dual[0])
            # 獎更大閉合與更大中位面積（花圈輪廓併成一塊）
            score += min(k, 21) * 0.4 + min(med, 2000.0) / 120.0
            if force_close:
                score += min(med, 2000.0) / 80.0
            if score > best_score:
                best_score = score
                best = closed
            continue
        # 閉合後雙尺度暫時消失（碎花併進底）：仍可用面積↑／塊數↓挑選
        if med >= 300 and len(cents) >= 16:
            score = med / 40.0 - len(cents) * 0.08 + min(k, 21) * 0.8
            if force_close:
                score += 30.0
            if score > best_score:
                best_score = score
                best = closed
            continue
        if med > med0 * 1.8 and len(cents) >= 12:
            score = med / 100.0 + len(cents) * 0.01
            if score > best_score:
                best_score = score
                best = closed
                med0 = med
    return best


def _motif_centroids_for_gap(
    arr: np.ndarray,
    bg: Sequence[int],
    threshold: float,
    med: float | None = None,
    fg_merged: np.ndarray | None = None,
) -> tuple[list[tuple[float, float, int]], float, np.ndarray]:
    """
    間距／節奏用質心：合併碎塊後，雙尺度只取大花。
    避免小碎花把跨縫間距誤判成 ≈2。
    第三回傳值為合併後 mask，供 _rank_tile 重用。
    fg_merged：可傳入已算好的合併 mask（裁切搜尋時避免對每個候選重跑閉運算）。
    """
    fg = fg_merged if fg_merged is not None else _fg_mask_merged(arr, bg, threshold)
    if med is None:
        med = _interior_median_motif_area(fg)
    cents = _fg_centroids(
        fg, min_area=max(40, int(med * 0.08) if med else 40), skip_edge=True
    )
    if len(cents) < 6:
        cents = _fg_centroids(fg, min_area=max(40, int(med * 0.08) if med else 40))
    if len(cents) < 6:
        return cents, med if med else 0.0, fg
    groups = _dual_scale_groups(cents)
    if groups is not None:
        large, _small = groups
        meds = _area_cluster_medians([a for _, _, a in cents])
        med_for_merge = float(med)
        if len(meds) >= 2:
            split = float(np.sqrt(meds[0] * meds[-1]))
            large_areas = [a for _, _, a in cents if a >= split]
            if large_areas:
                med_for_merge = float(np.median(large_areas))
        merge_dist = max(24.0, float(np.sqrt(max(med_for_merge, 1.0))) * 0.45)
        merged = _merge_nearby_centroids(
            [(cy, cx, 1000) for cy, cx in large], merge_dist
        )
        # 回傳整體 med 給切開檢查；勿用大花面積（否則切開門檻失真）
        return merged, float(med), fg
    merge_dist = max(24.0, float(np.sqrt(max(med, 1.0))) * 0.45)
    return _merge_nearby_centroids(cents, merge_dist), float(med), fg



def _erode_bool(mask: np.ndarray, iterations: int = 2) -> np.ndarray:
    kernel = np.ones((3, 3), np.uint8)
    return cv2.erode(
        mask.astype(np.uint8), kernel, iterations=max(1, iterations)
    ).astype(bool)


def _interior_median_motif_area(fg: np.ndarray) -> float:
    h, w = fg.shape
    m = max(2, min(h, w) // 40)
    interior = fg[m : h - m, m : w - m] if h > 2 * m and w > 2 * m else fg
    n, _, stats, _ = cv2.connectedComponentsWithStats(
        interior.astype(np.uint8), connectivity=4
    )
    areas = [int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, n) if int(stats[i, cv2.CC_STAT_AREA]) >= 40]
    if not areas:
        return 0.0
    return float(np.median(areas))


def _fg_centroids(
    fg: np.ndarray, min_area: int = 40, *, skip_edge: bool = False
) -> list[tuple[float, float, int]]:
    h, w = fg.shape
    n, _, stats, cents = cv2.connectedComponentsWithStats(
        fg.astype(np.uint8), connectivity=4
    )
    out: list[tuple[float, float, int]] = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x, y, bw, bh = (int(stats[i, j]) for j in range(4))
        if skip_edge and (x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1):
            continue
        out.append((float(cents[i, 1]), float(cents[i, 0]), area))
    return out


def _area_cluster_medians(areas: list[int]) -> list[float]:
    if not areas:
        return []
    arr = np.asarray(sorted(areas), dtype=np.float64)
    if len(arr) < 6:
        return [float(np.median(arr))]
    gaps = arr[1:] / np.maximum(arr[:-1], 1.0)
    i = int(np.argmax(gaps))
    # 1.65：水彩暈開／半隻動物常把 1.8 缺口填到 1.79，仍應切開大圖章
    if gaps[i] >= 1.65 and 2 <= i <= len(arr) - 3:
        return [float(np.median(arr[: i + 1])), float(np.median(arr[i + 1 :]))]
    mx = float(arr[-1])
    med = float(np.median(arr))
    # 動物＋骨頭／色塊：中間可能有過渡面積，改用「接近最大塊」當大圖章
    if mx >= max(med * 6.0, 5000.0) and len(arr) >= 8:
        thr = mx * 0.55
        large = arr[arr >= thr]
        small = arr[arr < thr]
        if len(large) >= 4 and len(small) >= 4:
            return [float(np.median(small)), float(np.median(large))]
    return [med]


def _tile_seam_scores(arr: np.ndarray) -> tuple[float, float]:
    a = arr.astype(np.float64)
    return float(np.mean(np.abs(a[:, -1] - a[:, 0]))), float(
        np.mean(np.abs(a[-1] - a[0]))
    )


def _centroid_seam_offsets(
    cents: list[tuple[float, float, int]],
    h: int,
    w: int,
    med: float,
) -> tuple[int, int, float, float]:
    min_sep = max(10.0, float(np.sqrt(max(med, 1.0))) * 0.35)

    def _axis(coords: list[float], length: int) -> tuple[int, float]:
        if len(coords) < 3 or length < 16:
            return 0, 0.0
        pts = sorted(coords)
        uniq: list[float] = [pts[0]]
        for p in pts[1:]:
            if p - uniq[-1] >= min_sep * 0.6:
                uniq.append(p)
        if len(uniq) < 2:
            return 0, 0.0
        gaps: list[tuple[float, float]] = []
        for i in range(len(uniq) - 1):
            gaps.append((float(uniq[i + 1] - uniq[i]), float(uniq[i])))
        wrap = float(uniq[0] + (length - uniq[-1]))
        interior = [g for g, _ in gaps]
        pitch = float(np.median(interior)) if interior else float(wrap)
        if pitch < 8:
            return 0, pitch
        # 種子相位：用「典型內部空隙」中點，勿用異常大的 wrap
        # （原圖若尚未四方連續，wrap 常大到 1.5×～2×，會把縫放到切圖案處）
        use_gaps = [g for g in gaps if 0.7 * pitch <= g[0] <= 1.3 * pitch]
        if not use_gaps:
            use_gaps = gaps if gaps else [(wrap, float(uniq[-1]))]
        size, start = use_gaps[len(use_gaps) // 2]
        mid = (start + size * 0.5) % length
        # 若 wrap 合理，也允許以 wrap 中點作為候選（取 mod pitch 更穩）
        if 0.7 * pitch <= wrap <= 1.3 * pitch:
            wrap_mid = (uniq[-1] + wrap * 0.5) % length
            # 兩種相位應接近；以內部為準
            _ = wrap_mid
        return int(round(mid)) % length, pitch

    ox, px = _axis([cx for _, cx, _ in cents], w)
    oy, py = _axis([cy for cy, _, _ in cents], h)
    return ox, oy, px, py


def _centroid_axis_gap_cv(
    coords: list[float], length: int, min_sep: float
) -> float:
    """軸向間距變異係數；規則晶格偏低，不規則散點偏高。"""
    if len(coords) < 4 or length < 16:
        return 99.0
    pts = sorted(coords)
    uniq: list[float] = [pts[0]]
    for p in pts[1:]:
        if p - uniq[-1] >= min_sep * 0.6:
            uniq.append(p)
    if len(uniq) < 4:
        return 99.0
    gaps = list(np.diff(uniq))
    gaps.append(float(uniq[0] + (length - uniq[-1])))
    med_g = float(np.median(gaps))
    if med_g < 8:
        return 99.0
    return float(np.std(gaps) / med_g)


def _nn_median_spacing(points: list[tuple[float, float]]) -> float:
    """質心最近鄰距離中位數（交錯格上比軸投影更穩）。"""
    n = len(points)
    if n < 4:
        return 0.0
    dists: list[float] = []
    for i, (cy, cx) in enumerate(points):
        best = 1e18
        for j, (cy2, cx2) in enumerate(points):
            if i == j:
                continue
            d = float(np.hypot(cy - cy2, cx - cx2))
            if d < best:
                best = d
        if best < 1e17:
            dists.append(best)
    if len(dists) < 3:
        return 0.0
    return float(np.median(dists))


def _nn_spacing_cv(points: list[tuple[float, float]]) -> float:
    n = len(points)
    if n < 4:
        return 99.0
    dists: list[float] = []
    for i, (cy, cx) in enumerate(points):
        best = 1e18
        for j, (cy2, cx2) in enumerate(points):
            if i == j:
                continue
            d = float(np.hypot(cy - cy2, cx - cx2))
            if d < best:
                best = d
        if best < 1e17:
            dists.append(best)
    if len(dists) < 4:
        return 99.0
    mean = float(np.mean(dists))
    if mean < 8:
        return 99.0
    return float(np.std(dists) / mean)


def _correct_half_pitch(
    pitch_x: float,
    pitch_y: float,
    points: list[tuple[float, float]],
) -> tuple[float, float]:
    """
    交錯／磚縫格把兩列投影疊在一起時，軸向 pitch 常變成真實週期的一半。
    用最近鄰距離校正：若 NN ≈ 2×pitch，則加倍；並避免 1/4 週期殘留。
    """
    nn = _nn_median_spacing(points)
    if nn < 16:
        return pitch_x, pitch_y
    out_x, out_y = pitch_x, pitch_y

    def _boost(p: float) -> float:
        # 反覆加倍到接近 NN（碎花質心常估出 1/4～1/3 真週期）
        cur = p
        for _ in range(3):
            if cur < 8:
                break
            if 1.70 * cur <= nn <= 2.45 * cur:
                cur *= 2.0
                continue
            # NN 遠大於 pitch（>2.45×）也視為低估，最多拉到 ≥0.55×NN
            if nn > 2.45 * cur and cur * 2.0 <= nn * 1.15:
                cur *= 2.0
                continue
            break
        return cur

    out_x = _boost(out_x)
    out_y = _boost(out_y)
    # 單軸半週期：NN 接近另一軸的完整 pitch
    if out_x == pitch_x and pitch_x >= 8 and out_y >= 16:
        if 1.5 * pitch_x <= out_y <= 2.6 * pitch_x and abs(nn - out_y) / out_y < 0.4:
            out_x = pitch_x * 2.0
    if out_y == pitch_y and pitch_y >= 8 and out_x >= 16:
        if 1.5 * pitch_y <= out_x <= 2.6 * pitch_y and abs(nn - out_x) / out_x < 0.4:
            out_y = pitch_y * 2.0
    return out_x, out_y


def looks_like_regular_lattice(
    arr: np.ndarray,
    bg: Sequence[int],
    threshold: float = 40.0,
    *,
    max_cv: float = 0.55,
    fg_merged: np.ndarray | None = None,
) -> bool:
    """
    質心是否近似規則晶格。不規則散點（已手繪四方連續的動物等）應為 False，
    避免硬走晶格裁切後 FAIL。
    雙尺度（大花環＋小碎花）只看大花；交錯格改用最近鄰 CV（軸投影 CV 會虛高）。
    """
    # 超大圖用縮圖判斷規則性（質心幾何與縮放無關）
    if fg_merged is None and max(arr.shape[0], arr.shape[1]) > _DISCRETE_FULLRES_TRIGGER:
        arr, _ = _downscale_for_discrete_analysis(arr)
    # 花圈等碎裂圖案先閉運算再估質心，否則全是小碎花、雙尺度偵測不到
    fg = fg_merged if fg_merged is not None else _fg_mask_merged(arr, bg, threshold)
    med = _interior_median_motif_area(fg)
    if med < 40:
        return False
    cents = _fg_centroids(fg, min_area=max(40, int(med * 0.08)), skip_edge=True)
    if len(cents) < 8:
        cents = _fg_centroids(fg, min_area=max(40, int(med * 0.08)))
    if len(cents) < 8:
        return False
    h, w = fg.shape
    groups = _dual_scale_groups(cents)
    if groups is not None:
        points = groups[0]
        meds = _area_cluster_medians([a for _, _, a in cents])
        med_pitch = med
        if len(meds) >= 2:
            split = float(np.sqrt(meds[0] * meds[-1]))
            large_areas = [a for _, _, a in cents if a >= split]
            if large_areas:
                med_pitch = float(np.median(large_areas))
        min_sep = max(10.0, float(np.sqrt(max(med_pitch, 1.0))) * 0.35)
        cvx = _centroid_axis_gap_cv([cx for _, cx in points], w, min_sep)
        cvy = _centroid_axis_gap_cv([cy for cy, _ in points], h, min_sep)
        nn_cv = _nn_spacing_cv(points)
        # 交錯格軸向 CV 常 >0.55，但 NN CV 仍低
        if nn_cv <= max_cv and len(points) >= 8:
            return True
        # 大圖章只有 6～7 顆時，間距必須很穩（恐龍）；罌粟連枝勿誤判
        if nn_cv <= 0.20 and len(points) >= 6:
            return True
        if cvx <= max_cv and cvy <= max_cv:
            return True
        # 大花過少時改看全部質心

    min_sep = max(10.0, float(np.sqrt(max(med, 1.0))) * 0.35)
    cvx = _centroid_axis_gap_cv([cx for _, cx, _ in cents], w, min_sep)
    cvy = _centroid_axis_gap_cv([cy for cy, _, _ in cents], h, min_sep)
    if cvx <= max_cv and cvy <= max_cv:
        return True
    nn_cv = _nn_spacing_cv([(cy, cx) for cy, cx, _ in cents])
    return nn_cv <= max_cv * 0.50 and len(cents) >= 12


def _detect_stagger(
    cents: list[tuple[float, float, int]], med: float
) -> bool:
    if len(cents) < 6:
        return False
    min_sep = max(10.0, float(np.sqrt(max(med, 1.0))) * 0.35)
    ys_all = sorted(cy for cy, _, _ in cents)
    row_ys: list[float] = [ys_all[0]]
    for v in ys_all[1:]:
        if v - row_ys[-1] >= min_sep * 0.6:
            row_ys.append(v)
    if len(row_ys) < 3:
        return False
    row_pitch = float(np.median(np.diff(row_ys)))
    bw = max(12.0, row_pitch * 0.4)
    rows: dict[int, list[float]] = {}
    for cy, cx, _ in cents:
        key = int(round(cy / max(bw, 1.0)))
        rows.setdefault(key, []).append(cx)
    keys = sorted(rows.keys())
    offsets: list[float] = []
    for i in range(len(keys) - 1):
        a = sorted(rows[keys[i]])
        b = sorted(rows[keys[i + 1]])
        if len(a) < 2 or len(b) < 2:
            continue
        pitch_x = float(np.median(np.diff(a))) if len(a) >= 3 else 0.0
        if pitch_x < 12:
            pitch_x = float(np.median(np.diff(b))) if len(b) >= 3 else 0.0
        if pitch_x < 12:
            continue
        # 勿用整列中位數：奇數個點時磚縫兩列中位數會碰巧對齊
        d = abs(a[0] - b[0]) % pitch_x
        d = min(d, pitch_x - d)
        offsets.append(d / pitch_x)
    if offsets and float(np.median(offsets)) > 0.28:
        return True
    # 備援：NN ≈ 對角（磚縫／交錯）
    pts = [(cy, cx) for cy, cx, _ in cents]
    nn = _nn_median_spacing(pts)
    if nn < 16 or row_pitch < 16:
        return False
    xs = sorted({cx for _, cx, _ in cents})
    if len(xs) < 3:
        return False
    pitch_x = float(np.median(np.diff(xs)))
    if pitch_x < 16:
        return False
    diag = float(np.hypot(pitch_x, row_pitch))
    return bool(0.7 * diag <= nn <= 1.3 * diag)


def _detect_color_period_2x(
    arr: np.ndarray,
    cents: list[tuple[float, float, int]],
    bg: Sequence[int],
    threshold: float,
) -> tuple[bool, bool]:
    """相鄰質心顏色差大 → 該軸需要 2× 週期（雙色棋盤）。回傳 (need_2x_x, need_2x_y)。"""
    if len(cents) < 8:
        return False, False
    fg = _fg_mask(arr, bg, threshold)
    h, w = arr.shape[:2]
    colored: list[tuple[float, float, np.ndarray]] = []
    for cy, cx, _ in cents:
        y0 = min(max(int(round(cy)), 5), h - 6)
        x0 = min(max(int(round(cx)), 5), w - 6)
        patch = arr[y0 - 5 : y0 + 6, x0 - 5 : x0 + 6].astype(np.float64)
        m = fg[y0 - 5 : y0 + 6, x0 - 5 : x0 + 6]
        if int(m.sum()) < 5:
            continue
        colored.append((cy, cx, patch[m].mean(0)))
    if len(colored) < 8:
        return False, False

    # 顏色雙峰：通道高方差，或近鄰質心色差大 → 雙色棋盤／隔行
    rs = np.array([c[2][0] for c in colored])
    gs = np.array([c[2][1] for c in colored])
    bs = np.array([c[2][2] for c in colored])
    dualish = max(float(rs.std()), float(gs.std()), float(bs.std())) > 28.0
    if not dualish:
        # 粉／橙等相近色：看近鄰色差
        nn_diffs: list[float] = []
        for i, (cy, cx, col) in enumerate(colored):
            best_d = 1e9
            best_cd = 0.0
            for j, (cy2, cx2, col2) in enumerate(colored):
                if i == j:
                    continue
                dist = float(np.hypot(cy2 - cy, cx2 - cx))
                if 25.0 < dist < best_d:
                    best_d = dist
                    best_cd = float(np.mean(np.abs(col - col2)))
            if best_d < 1e8:
                nn_diffs.append(best_cd)
        if len(nn_diffs) >= 8 and float(np.median(nn_diffs)) > 28.0:
            dualish = True
    # 多物種圖章色方差也會高，不可直接 2×；改看軸向上是否真的隔顆換色

    def _axis_need(along_x: bool) -> bool:
        diffs: list[float] = []
        for i, (cy, cx, col) in enumerate(colored):
            best_d = 1e9
            best_cd = 0.0
            for j, (cy2, cx2, col2) in enumerate(colored):
                if i == j:
                    continue
                if along_x:
                    if abs(cy2 - cy) > 40:
                        continue
                    d = cx2 - cx
                else:
                    if abs(cx2 - cx) > 40:
                        continue
                    d = cy2 - cy
                if 40 < d < best_d:
                    best_d = d
                    best_cd = float(np.mean(np.abs(col - col2)))
            if best_d < 1e8:
                diffs.append(best_cd)
        if len(diffs) < 4:
            return False
        return float(np.median(diffs)) > 22.0

    return _axis_need(True), _axis_need(False)


def _period_candidates(
    pitch_x: float,
    pitch_y: float,
    h: int,
    w: int,
    stagger: bool,
    color_2x_x: bool,
    color_2x_y: bool,
) -> list[tuple[int, int]]:
    """只產生 pitch / 2×pitch 附近，禁止亂入 4×。"""

    def _expand(pitch: float, full: int, force_2x: bool) -> list[int]:
        out: list[int] = []
        if pitch < 16:
            return out
        bases = [int(round(pitch))]
        if force_2x or stagger:
            bases.append(int(round(pitch * 2)))
        # 必須保留 round(pitch)；覆蓋率只作次要排序
        scored: list[tuple[float, int]] = []
        for base in bases:
            for d in (-2, -1, 0, 1, 2):
                v = base + d
                if 24 <= v <= full // 2:
                    cover = (full // v) * v / full
                    # 貼近估測週期為主，覆蓋率為輔（避免 82 勝過真正的 85）
                    scored.append((abs(v - pitch) * 2.0 - cover * 3.0, v))
        scored.sort()
        # round(pitch) 置頂
        primary = int(round(pitch))
        if 24 <= primary <= full // 2:
            out.append(primary)
        if force_2x or stagger:
            p2 = int(round(pitch * 2))
            if 24 <= p2 <= full // 2 and p2 not in out:
                out.append(p2)
        for _, v in scored:
            if v not in out:
                out.append(v)
        return out[:8]

    pxs = _expand(pitch_x, w, color_2x_x)
    pys = _expand(pitch_y, h, color_2x_y or stagger)
    if not pxs or not pys:
        return []
    # 雙色時單倍與 2× 都保留（隔行換色需要奇數行高，常是單倍 pitch）
    pairs: list[tuple[int, int]] = []
    if color_2x_x or color_2x_y or stagger:
        # 雙色：先 2×2（完整色週期），再 2×1／1×1
        xs2 = [p for p in pxs if abs(p - pitch_x * 2) <= 4] or pxs[:3]
        ys2 = [p for p in pys if abs(p - pitch_y * 2) <= 4] or pys[:3]
        xs1 = [p for p in pxs if abs(p - pitch_x) <= 2] or pxs[:3]
        ys1 = [p for p in pys if abs(p - pitch_y) <= 2] or pys[:3]
        for px in xs2[:3]:
            for py in ys2[:3]:
                pairs.append((px, py))
        for px in xs2[:3]:
            for py in ys1[:3]:
                pairs.append((px, py))
        for px in xs1[:3]:
            for py in ys1[:3]:
                pairs.append((px, py))
    else:
        for px in pxs[:4]:
            for py in pys[:4]:
                pairs.append((px, py))
    # 去重保序
    uniq: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for pr in pairs:
        if pr not in seen:
            seen.add(pr)
            uniq.append(pr)
    return uniq[:16]


def _wrap_gap_ratios(
    arr: np.ndarray,
    bg: Sequence[int],
    threshold: float,
    med: float | None = None,
    cents: list[tuple[float, float, int]] | None = None,
) -> tuple[float, float]:
    if cents is None:
        cents, med_g, _fg_m = _motif_centroids_for_gap(arr, bg, threshold, med)
        if med is None:
            med = med_g
    if med is None or med < 40:
        fg = _fg_mask(arr, bg, threshold)
        med = _interior_median_motif_area(fg)
    if med < 40:
        return 1.0, 1.0
    if len(cents) < 6:
        return 1.0, 1.0
    h, w = arr.shape[:2]
    min_sep = max(10.0, float(np.sqrt(max(med, 1.0))) * 0.35)

    def _band_ratios(pts: list[tuple[float, float]], length: int) -> list[float]:
        if len(pts) < 4:
            return []
        across = sorted({p[1] for p in pts})
        if len(across) < 2:
            bands = [pts]
        else:
            pitch = float(np.median(np.diff(across))) if len(across) > 1 else 40.0
            bw = max(12.0, pitch * 0.4)
            grouped: dict[int, list[float]] = {}
            for along, ac in pts:
                key = int(round(ac / max(bw, 1.0)))
                grouped.setdefault(key, []).append(along)
            bands = [[(a, 0.0) for a in xs] for xs in grouped.values() if len(xs) >= 2]
        ratios: list[float] = []
        for band in bands:
            coords = sorted(p[0] for p in band)
            uniq: list[float] = [coords[0]]
            for p in coords[1:]:
                if p - uniq[-1] >= min_sep * 0.6:
                    uniq.append(p)
            if len(uniq) < 2:
                continue
            med_g = (
                float(uniq[1] - uniq[0])
                if len(uniq) == 2
                else float(np.median(np.diff(uniq)))
            )
            if med_g < 8:
                continue
            wrap = float(uniq[0] + (length - uniq[-1]))
            ratios.append(wrap / med_g)
        return ratios

    xy = [(cx, cy) for cy, cx, _ in cents]
    yx = [(cy, cx) for cy, cx, _ in cents]
    rx_list = _band_ratios(xy, w)
    ry_list = _band_ratios(yx, h)
    return (
        float(np.median(rx_list)) if rx_list else 1.0,
        float(np.median(ry_list)) if ry_list else 1.0,
    )


def _large_motif_median(
    arr: np.ndarray,
    bg: Sequence[int],
    threshold: float,
    med: float,
) -> float:
    """雙尺度時切開檢查改用大圖章中位面積，否則半隻狗仍大於骨頭／色點。"""
    if med < 40:
        return med
    fg = _fg_mask(arr, bg, threshold)
    cents = _fg_centroids(
        fg, min_area=max(40, int(med * 0.08)), skip_edge=True
    )
    if len(cents) < 6:
        return med
    meds = _area_cluster_medians([a for *_, a in cents])
    if len(meds) < 2 or meds[1] < meds[0] * 4.0:
        return med
    split = float(np.sqrt(meds[0] * meds[1]))
    if meds[1] >= meds[0] * 6.0:
        split = max(split, float(meds[1]) * 0.55)
    large_areas = [a for *_, a in cents if a >= split]
    if len(large_areas) < 4:
        return med
    return max(med, float(np.median(large_areas)))


def _cross_seam_cut_count(
    arr: np.ndarray,
    bg: Sequence[int],
    threshold: float,
    med: float | None = None,
    area_ratio_max: float = 0.55,
    *,
    adjust_large_med: bool = True,
    fg: np.ndarray | None = None,
) -> int:
    """
    腐蝕後中縫殘片數（只計明顯小於完整圖案的碎片）。
    不做質量平衡／色差硬判——那些會把正確跨縫對接誤判成切開。
    adjust_large_med=False：med 已是切開用面積（含大圖章中位），跳過昂貴重估。
    """
    if fg is None:
        fg = _fg_mask(arr, bg, threshold)
    if med is None:
        med = _interior_median_motif_area(fg)
    if adjust_large_med:
        med = _large_motif_median(arr, bg, threshold, med)
    if med < 40:
        return 0
    body = _erode_bool(fg, 2)
    cuts = 0
    for axis in (0, 1):
        double = np.concatenate([body, body], axis=axis)
        seam = body.shape[axis]
        n_labels, _, stats, _ = cv2.connectedComponentsWithStats(
            double.astype(np.uint8), connectivity=4
        )
        for i in range(1, n_labels):
            x, y, bw, bh, area = (int(stats[i, j]) for j in range(5))
            if area < max(12, int(med * 0.05)):
                continue
            touches = (x <= seam <= x + bw) if axis == 1 else (y <= seam <= y + bh)
            if touches and area < med * area_ratio_max:
                cuts += 1
    return cuts


def _discrete_integrity_ok(
    arr: np.ndarray,
    bg: Sequence[int],
    threshold: float,
    med: float | None = None,
    *,
    max_cuts: int = 0,
    gap_lo: float = 0.90,
    gap_hi: float = 1.15,
) -> bool:
    if med is None:
        med = _interior_median_motif_area(_fg_mask(arr, bg, threshold))
    if med < 40:
        return True
    if _cross_seam_cut_count(arr, bg, threshold, med=med) > max_cuts:
        return False
    gx, gy = _wrap_gap_ratios(arr, bg, threshold, med=med)
    return gap_lo <= gx <= gap_hi and gap_lo <= gy <= gap_hi


def _dual_scale_groups(
    cents: list[tuple[float, float, int]],
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]] | None:
    areas = [a for _, _, a in cents]
    meds = _area_cluster_medians(areas)
    if len(meds) < 2:
        return None
    split = float(np.sqrt(meds[0] * meds[1]))
    # 動物 vs 小填充面積差極大時，幾何平均會把中等色塊算進大圖章
    if meds[1] >= meds[0] * 6.0:
        split = max(split, float(meds[1]) * 0.55)
    large = [(cy, cx) for cy, cx, a in cents if a >= split]
    small = [(cy, cx) for cy, cx, a in cents if a < split]
    if len(large) < 6 or len(small) < 4:
        return None
    return large, small


def _interstitial_star_mismatch(
    large: list[tuple[float, float]],
    small: list[tuple[float, float]],
    h: int,
    w: int,
) -> int:
    def _axis_delta(
        majors: list[tuple[float, float]],
        minors: list[tuple[float, float]],
        length: int,
    ) -> int:
        if len(majors) < 4 or len(minors) < 2:
            return 0
        across = sorted({m[1] for m in majors})
        if len(across) < 2:
            return 0
        pitch = float(np.median(np.diff(across)))
        bw = max(12.0, pitch * 0.35)
        bands: dict[int, list[float]] = {}
        for along, ac in majors:
            key = int(round(ac / max(bw, 1.0)))
            bands.setdefault(key, []).append(along)
        int_counts: list[int] = []
        wrap_counts: list[int] = []
        for key, xs in bands.items():
            if len(xs) < 2:
                continue
            xs = sorted(xs)
            ac0 = key * bw
            gaps = list(np.diff(xs))
            med_g = float(np.median(gaps)) if gaps else 0.0
            if med_g < 12:
                continue
            for i in range(len(xs) - 1):
                a0, a1 = xs[i], xs[i + 1]
                cnt = sum(
                    1
                    for s_along, s_ac in minors
                    if a0 + med_g * 0.12 < s_along < a1 - med_g * 0.12
                    and abs(s_ac - ac0) <= pitch * 0.55
                )
                int_counts.append(cnt)
            a0 = xs[-1]
            cnt_w = sum(
                1
                for s_along, s_ac in minors
                if (s_along > a0 + med_g * 0.12 or s_along < xs[0] - med_g * 0.12)
                and abs(s_ac - ac0) <= pitch * 0.55
            )
            wrap_counts.append(cnt_w)
        if not int_counts or not wrap_counts:
            return 0
        exp = int(round(float(np.median(int_counts))))
        return int(sum(abs(c - exp) for c in wrap_counts))

    dx = _axis_delta([(cx, cy) for cy, cx in large], [(cx, cy) for cy, cx in small], w)
    dy = _axis_delta([(cy, cx) for cy, cx in large], [(cy, cx) for cy, cx in small], h)
    return dx + dy


def _period_consistency_error(tile: np.ndarray, px: int, py: int) -> float:
    """單元應滿足平移 px/py 後重合；否則週期估錯會在接縫累積錯位。"""
    h, w = tile.shape[:2]
    a = tile.astype(np.float64)
    err = 0.0
    n = 0
    if 8 <= px < w // 2:
        err += float(np.mean(np.abs(a[:, :-px] - a[:, px:])))
        n += 1
    if 8 <= py < h // 2:
        err += float(np.mean(np.abs(a[:-py, :] - a[py:, :])))
        n += 1
    return err / max(n, 1)


def _edge_profile_mismatch(fg: np.ndarray) -> float:
    """
    對邊前景剖面錯位（1px 漂移／半週期撞色切開）懲罰。
    縫落空隙（兩邊都幾乎無前景）不罰。
    """
    h, w = fg.shape
    if h < 16 or w < 16:
        return 0.0
    pen = 0.0
    band = max(2, min(5, min(h, w) // 80))

    def _axis_pen(along_x: bool) -> float:
        if along_x:
            a = fg[:, :band].mean(axis=1)
            b = fg[:, w - band :].mean(axis=1)
        else:
            a = fg[:band, :].mean(axis=0)
            b = fg[h - band :, :].mean(axis=0)
        if float(a.mean()) < 0.05 and float(b.mean()) < 0.05:
            return 0.0
        if float(a.std()) < 0.02 and float(b.std()) < 0.02:
            return 0.0
        a0 = a - a.mean()
        b0 = b - b.mean()
        denom = float(np.sqrt(np.sum(a0**2) * np.sum(b0**2))) + 1e-9
        best = -1.0
        for lag in range(-4, 5):
            if lag < 0:
                aa, bb = a0[-lag:], b0[: len(b0) + lag]
            elif lag > 0:
                aa, bb = a0[: len(a0) - lag], b0[lag:]
            else:
                aa, bb = a0, b0
            if len(aa) < 8:
                continue
            corr = float(np.dot(aa, bb) / (np.sqrt(np.sum(aa**2) * np.sum(bb**2)) + 1e-9))
            best = max(best, corr)
        if best < 0.55:
            return (0.55 - max(best, -0.2)) * 50.0
        return 0.0

    pen += _axis_pen(True)
    pen += _axis_pen(False)
    return pen


def _edge_band_color(
    tile: np.ndarray,
    fg: np.ndarray,
    side: str,
    band: int,
) -> np.ndarray | None:
    h, w = fg.shape
    b = max(2, min(band, h // 8, w // 8))
    if side == "top":
        m = fg[:b, :]
        patch = tile[:b, :]
    elif side == "bot":
        m = fg[h - b :, :]
        patch = tile[h - b :, :]
    elif side == "left":
        m = fg[:, :b]
        patch = tile[:, :b]
    else:
        m = fg[:, w - b :]
        patch = tile[:, w - b :]
    if float(m.mean()) < 0.012 or int(m.sum()) < 8:
        return None
    return patch[m].astype(np.float64).mean(0)


def _color_wrap_penalty(
    tile: np.ndarray,
    bg: Sequence[int],
    threshold: float,
) -> float:
    """
    局部對邊比色：同一列／行兩側都有前景且主色差大 → 跨縫撞色。
    不用整邊均色（交錯雙色時左右整邊常是不同色但局部對接仍正確）。
    """
    fg = _fg_mask(tile, bg, threshold)
    h, w = fg.shape
    if h < 32 or w < 32:
        return 0.0
    a = tile.astype(np.float64)
    pen = 0.0
    bad = 0
    checked = 0
    band = max(4, min(10, min(h, w) // 60))
    win = max(16, min(40, min(h, w) // 20))

    for y0 in range(0, h - win + 1, win // 2):
        l = fg[y0 : y0 + win, :band]
        r = fg[y0 : y0 + win, w - band :]
        if float(l.mean()) < 0.06 or float(r.mean()) < 0.06:
            continue
        checked += 1
        lc = a[y0 : y0 + win, :band][l].mean(0)
        rc = a[y0 : y0 + win, w - band :][r].mean(0)
        if float(np.mean(np.abs(lc - rc))) > 40.0:
            bad += 1

    for x0 in range(0, w - win + 1, win // 2):
        t = fg[:band, x0 : x0 + win]
        b = fg[h - band :, x0 : x0 + win]
        if float(t.mean()) < 0.06 or float(b.mean()) < 0.06:
            continue
        checked += 1
        tc = a[:band, x0 : x0 + win][t].mean(0)
        bc = a[h - band :, x0 : x0 + win][b].mean(0)
        if float(np.mean(np.abs(tc - bc))) > 40.0:
            bad += 1

    if checked == 0:
        return 0.0
    if bad / checked >= 0.35:
        pen = 55.0
    elif bad >= 3:
        pen = 30.0
    return pen


def _rank_tile_light(
    tile: np.ndarray,
    bg: Sequence[int],
    threshold: float,
    med: float,
    px: int = 0,
    py: int = 0,
    *,
    cut_med: float | None = None,
) -> tuple[int, float]:
    """搜尋用輕量評分：切開數 + 邊緣前景 + 週期一致性（不做閉運算合併）。"""
    fg = _fg_mask(tile, bg, threshold)
    med_cut = cut_med if cut_med is not None else med
    cuts = _cross_seam_cut_count(
        tile,
        bg,
        threshold,
        med=med_cut,
        adjust_large_med=cut_med is None,
        fg=fg,
    )
    score = cuts * 100.0
    score += _edge_fg_score(fg, max(2, min(6, min(tile.shape[0], tile.shape[1]) // 80))) * 8.0
    score += _edge_profile_mismatch(fg)
    score += _join_continuity_penalty(tile)
    if px > 0 and py > 0:
        score += _period_consistency_error(tile, px, py) * 2.0
    tier = 0 if cuts == 0 else 2
    return tier, score


def _rank_tile(
    tile: np.ndarray,
    bg: Sequence[int],
    threshold: float,
    med: float,
    px: int = 0,
    py: int = 0,
    *,
    cut_med: float | None = None,
    fg_merged: np.ndarray | None = None,
) -> tuple[int, float, float, float]:
    """回傳 (tier, score, gap_x, gap_y)。搜尋端可直接用 gap，避免再算一次 wrap_gap。"""
    fg = _fg_mask(tile, bg, threshold)
    med_cut = cut_med if cut_med is not None else med
    cuts = _cross_seam_cut_count(
        tile,
        bg,
        threshold,
        med=med_cut,
        adjust_large_med=cut_med is None,
        fg=fg,
    )
    gap_cents, gap_med, fg_merged = _motif_centroids_for_gap(
        tile, bg, threshold, med, fg_merged=fg_merged
    )
    gx, gy = _wrap_gap_ratios(
        tile, bg, threshold, med=gap_med if gap_med >= 40 else med, cents=gap_cents
    )
    gap_ok = 0.90 <= gx <= 1.15 and 0.90 <= gy <= 1.15
    cpen = _color_wrap_penalty(tile, bg, threshold)
    # 軟通過也必須 gap 合理；否則整圖常以「cuts=0 + 壞 gap」打敗正確裁切
    soft_ok = cuts == 0 and gap_ok
    hard = soft_ok and cpen < 40.0
    tier = 0 if hard else (1 if soft_ok else 2)
    # 點綴不以邊緣像素差為主（對半拼接時對邊本來就不像）
    score = (
        cuts * 100.0
        + abs(gx - 1.0) * 80.0
        + abs(gy - 1.0) * 80.0
        + cpen
    )
    if px > 0 and py > 0:
        cons = _period_consistency_error(tile, px, py)
        score += cons * 2.0
    # 輕量邊緣前景：縫落空隙略優於切開，但不壓過完整性
    score += _edge_fg_score(fg, max(2, min(6, min(tile.shape[0], tile.shape[1]) // 80))) * 8.0
    score += _edge_profile_mismatch(fg)
    score += _join_continuity_penalty(tile)
    groups = _dual_scale_groups(gap_cents) if gap_cents else None
    # gap_cents 若已是大花-only，再 dual 會失敗；改用合併 mask 全質心（重用已算 mask）
    if groups is None:
        med_m = _interior_median_motif_area(fg_merged)
        cents_m = _fg_centroids(
            fg_merged, min_area=max(40, int(med_m * 0.08) if med_m else 40)
        )
        groups = _dual_scale_groups(cents_m)
    if groups is not None:
        # 双尺度：優先縫落空隙，避免大切過大花造成 1px 錯位感
        score += _edge_fg_score(fg, max(2, min(6, min(tile.shape[0], tile.shape[1]) // 80))) * 25.0
        miss = _interstitial_star_mismatch(
            groups[0], groups[1], tile.shape[0], tile.shape[1]
        )
        score += miss * 30.0
        if hard and miss > 0:
            tier = 1
    return tier, score, gx, gy


def _edge_fg_score(fg: np.ndarray, band: int) -> float:
    h, w = fg.shape
    b = max(1, min(band, h // 8, w // 8))
    return float(
        fg[:, :b].mean()
        + fg[:, w - b :].mean()
        + fg[:b, :].mean()
        + fg[h - b :, :].mean()
    )


def _float_period_sizes(full: int, pitch: float, min_cover: float = 0.72) -> list[tuple[int, int]]:
    """回傳 (size, n_repeats)。優先 n×round(pitch)，保證 size 為整數週期倍數。"""
    if pitch < 16:
        return []
    out: list[tuple[int, int]] = []
    n0 = max(2, int(full * min_cover / pitch))
    n1 = max(n0, int(full / pitch))
    scored: list[tuple[float, int, int]] = []
    ip = max(24, int(round(pitch)))
    for n in range(n0, n1 + 1):
        # 只收精確整數倍，避免 round(n*pitch) 造成 size≠n×period
        for size in {n * ip}:
            if size > full or size < int(full * min_cover):
                continue
            err = abs(size / n - pitch)
            cover = size / full
            scored.append((err * 10.0 - cover, size, n))
        # 次選：round(n*pitch) 但立刻 snap 回最近整數倍
        approx = int(round(n * pitch))
        if approx <= full and approx >= int(full * min_cover):
            n2 = max(1, int(round(approx / ip)))
            size2 = n2 * ip
            if size2 <= full and size2 >= int(full * min_cover):
                err = abs(size2 / n2 - pitch)
                cover = size2 / full
                scored.append((err * 10.0 + 0.2 - cover, size2, n2))
    scored.sort()
    seen: set[int] = set()
    for _, size, n in scored:
        if size in seen:
            continue
        # 硬保證整除
        if size % max(1, int(round(size / max(n, 1)))) != 0:
            n = max(1, int(round(size / ip)))
            size = n * ip
            if size > full or size < int(full * min_cover) or size in seen:
                continue
        seen.add(size)
        out.append((size, n))
        if len(out) >= 5:
            break
    alt = (full // ip) * ip
    if alt >= int(full * min_cover) and alt <= full and alt not in seen:
        out.append((alt, max(1, alt // ip)))
    return out


def _join_continuity_penalty(tile: np.ndarray) -> float:
    """對邊拼接處梯度跳變相對內部的倍數；越大越不連續。"""
    a = tile.astype(np.float64)
    h, w = a.shape[:2]
    if h < 16 or w < 16:
        return 0.0
    dbl = np.concatenate([a, a], axis=1)
    jx = float(np.mean(np.abs(dbl[:, w] - dbl[:, w - 1])))
    ix = float(np.mean(np.abs(dbl[:, w // 2] - dbl[:, w // 2 - 1]))) + 1e-6
    dbl2 = np.concatenate([a, a], axis=0)
    jy = float(np.mean(np.abs(dbl2[h] - dbl2[h - 1])))
    iy = float(np.mean(np.abs(dbl2[h // 2] - dbl2[h // 2 - 1]))) + 1e-6
    return max(0.0, jx / ix - 1.3) * 12.0 + max(0.0, jy / iy - 1.3) * 12.0


def _rolled_interior_join_penalty(tile: np.ndarray, ox: int, oy: int) -> float:
    """np.roll 後原圖對邊接到畫面中間；量那條內部接縫相對內部的跳變。"""
    a = tile.astype(np.float64)
    h, w = a.shape[:2]
    if h < 16 or w < 16:
        return 0.0
    pen = 0.0
    if oy % h != 0:
        yj = (-int(oy)) % h
        jy = float(np.mean(np.abs(a[yj] - a[(yj - 1) % h])))
        iy = float(np.mean(np.abs(a[h // 2] - a[h // 2 - 1]))) + 1e-6
        pen += max(0.0, jy / iy - 1.3) * 12.0
    if ox % w != 0:
        xj = (-int(ox)) % w
        jx = float(np.mean(np.abs(a[:, xj] - a[:, (xj - 1) % w])))
        ix = float(np.mean(np.abs(a[:, w // 2] - a[:, w // 2 - 1]))) + 1e-6
        pen += max(0.0, jx / ix - 1.3) * 12.0
    return pen


def try_discrete_lattice_crop(
    arr: np.ndarray,
    bg: Sequence[int],
    threshold: float = 40.0,
    *,
    fg_merged: np.ndarray | None = None,
    med: float | None = None,
    cut_med: float | None = None,
    log: Callable[[str], None] | None = None,
) -> tuple[np.ndarray | None, str, int, tuple[int, int, int, int, int, int] | None]:
    """
    回傳 (tile, detail, tier, box)。
    box = (dx, dy, cw, ch, px, py)，失敗時為 None。
    """
    def _lg(msg: str) -> None:
        if log is not None:
            log(msg)

    h, w = arr.shape[:2]
    fg_raw = _fg_mask(arr, bg, threshold)
    if float(fg_raw.mean()) < 0.02:
        return None, "前景過少", 2, None

    # 週期／質心用合併後 mask；切開檢查仍看原始色差前景
    fg = fg_merged if fg_merged is not None else _fg_mask_merged(arr, bg, threshold)
    if med is None:
        med = _interior_median_motif_area(fg)
    if med < 40:
        return None, "無法估圖案尺寸", 2, None
    # 切開門檻只估一次（源圖），搜尋候選不再重跑大圖章質心
    if cut_med is None:
        cut_med = _large_motif_median(arr, bg, threshold, med)

    cents_all = _fg_centroids(fg, min_area=max(40, int(med * 0.08)), skip_edge=True)
    if len(cents_all) < 6:
        cents_all = _fg_centroids(fg, min_area=max(40, int(med * 0.08)))
    if len(cents_all) < 6:
        return None, "質心過少", 2, None

    groups = _dual_scale_groups(cents_all)
    if groups is not None:
        large, _small = groups
        # 大花過少（柯基+碎星心）：勿只拿很少點估週期，否則易 2× 半幅假週期
        # 恐龍等 6～11 個大圖章仍足以估磚縫週期
        if len(large) < 6:
            groups = None
            cents_pitch = cents_all
            med_pitch = med
        else:
            meds = _area_cluster_medians([a for _, _, a in cents_all])
            split = float(np.sqrt(meds[0] * meds[-1]))
            large_areas = [a for _, _, a in cents_all if a >= split]
            med_pitch = float(np.median(large_areas)) if large_areas else med
            cents_pitch = [(cy, cx, 1000) for cy, cx in large]
    if groups is None:
        cents_pitch = cents_all
        med_pitch = med

    # 合併花圈碎裂雙心，再估週期／相位
    # 合併半徑取「小於半週期間距」：先用面積估，再用 NN 收斂
    merge_dist = max(36.0, float(np.sqrt(max(med_pitch, 1.0))) * 1.1)
    cents_pitch = _merge_nearby_centroids(cents_pitch, merge_dist)
    pitch_pts0 = [(cy, cx) for cy, cx, _ in cents_pitch]
    nn0 = _nn_median_spacing(pitch_pts0)
    if nn0 >= 40:
        # 若仍有明顯亞晶格近點，再併一次（不超過 0.4×NN）
        cents_pitch = _merge_nearby_centroids(
            cents_pitch, max(merge_dist, min(nn0 * 0.4, 90.0))
        )
    if len(cents_pitch) < 4:
        return None, "質心過少", 2, None

    ox0, oy0, pitch_x, pitch_y = _centroid_seam_offsets(cents_pitch, h, w, med_pitch)
    # 交錯／磚縫：軸投影半週期 → 用最近鄰距離校正
    pitch_pts = [(cy, cx) for cy, cx, _ in cents_pitch]
    pitch_x, pitch_y = _correct_half_pitch(pitch_x, pitch_y, pitch_pts)
    nn_pitch = _nn_median_spacing(pitch_pts)
    # 須先判交錯：六角圓點列距常 ≈ 0.69×NN，不可改成對角線距離
    stagger = _detect_stagger(cents_pitch, med_pitch)
    # 軸向週期仍遠小於最近鄰：改用 NN（骨頭／色點會把軸投影拉成 20～40px）
    if nn_pitch >= 64.0 and not stagger:
        if pitch_x > 0 and pitch_x < nn_pitch * 0.72:
            pitch_x = nn_pitch
        if pitch_y > 0 and pitch_y < nn_pitch * 0.72:
            pitch_y = nn_pitch
    # 絕對／相對下限：過短週期幾乎一定是碎花亞晶格，硬通過會切花
    min_pitch = max(64.0, float(np.sqrt(max(med_pitch, 1.0))) * 1.5)
    if nn_pitch >= 40:
        min_pitch = max(min_pitch, nn_pitch * 0.6)
    # 交錯格列距／半偏移略低於 min_pitch 時不要加倍；矩形單元改走 2×
    if not stagger:
        if pitch_x > 0 and pitch_x < min_pitch:
            # 拉到接近 NN，避免 ceil(min/p)*p 過頭（例 170→340 但 NN=267）
            target = nn_pitch if nn_pitch >= min_pitch else min_pitch
            mul = max(1, int(round(target / max(pitch_x, 1e-6))))
            cand = pitch_x * float(mul)
            if cand < min_pitch * 0.85:
                mul = max(2, int(np.ceil(min_pitch / max(pitch_x, 1e-6))))
                cand = pitch_x * float(mul)
            pitch_x = cand
        if pitch_y > 0 and pitch_y < min_pitch:
            target = nn_pitch if nn_pitch >= min_pitch else min_pitch
            mul = max(1, int(round(target / max(pitch_y, 1e-6))))
            cand = pitch_y * float(mul)
            if cand < min_pitch * 0.85:
                mul = max(2, int(np.ceil(min_pitch / max(pitch_y, 1e-6))))
                cand = pitch_y * float(mul)
            pitch_y = cand
    c2x, c2y = _detect_color_period_2x(arr, cents_all, bg, threshold)
    # 週期已接近半幅時禁止再 2×（不規則散點／已是完整色週期）
    if c2x or c2y:
        if pitch_x * 2.0 <= w * 0.48 and pitch_y * 2.0 <= h * 0.48:
            c2x, c2y = True, True
            pitch_x = pitch_x * 2.0
            pitch_y = pitch_y * 2.0
        else:
            c2x, c2y = False, False

    sizes_x = _float_period_sizes(w, pitch_x)
    sizes_y = _float_period_sizes(h, pitch_y)
    sizes_x2: list[tuple[int, int]] = []
    sizes_y2: list[tuple[int, int]] = []
    # 交錯或雙尺度大花：一律保留 2× 週期候選，避免半週期假通過
    if (stagger or groups is not None) and not (c2x or c2y):
        # 大 pitch 的 2 週期常略低於預設 0.72 覆蓋；交錯仍必須用 2× 矩形單元
        cover2 = 0.50 if stagger else 0.72
        sizes_x2 = _float_period_sizes(w, pitch_x * 2.0, min_cover=cover2)
        sizes_y2 = _float_period_sizes(h, pitch_y * 2.0, min_cover=cover2)
    if not sizes_x or not sizes_y:
        if sizes_x2 and sizes_y2:
            sizes_x, sizes_y = sizes_x2, sizes_y2
        else:
            return None, "無週期候選", 2, None

    # 質心空隙作種子偏移（用浮點 pitch 取模，避免 round 後差 3～5px 切到圖案）
    ipx = max(1, int(round(pitch_x if not (c2x or c2y) else pitch_x)))
    ipy = max(1, int(round(pitch_y)))
    seed_ox = int(round(float(ox0) % float(pitch_x))) % ipx
    seed_oy = int(round(float(oy0) % float(pitch_y))) % ipy
    band = max(2, int(round(np.sqrt(max(med, 1.0))) * 0.12))
    min_side = max(int(min(h, w) * 0.50), 240)
    fg = _fg_mask(arr, bg, threshold)

    # 交錯時 2× 尺寸優先，避免 1× 半週期「假硬通過」
    size_pairs: list[tuple[int, int, int, int]] = []

    def _append_pairs(sx: list[tuple[int, int]], sy: list[tuple[int, int]]) -> None:
        # 少重複優先（2×、3×），大覆蓋放後面
        scored: list[tuple[float, int, int, int, int]] = []
        for cw, nx in sx[:4]:
            for ch, ny in sy[:4]:
                if cw < min_side or ch < min_side:
                    continue
                px = max(1, int(round(cw / max(nx, 1))))
                py = max(1, int(round(ch / max(ny, 1))))
                cw2, ch2 = nx * px, ny * py
                if cw2 > w or ch2 > h or cw2 < min_side or ch2 < min_side:
                    continue
                scored.append((float(nx + ny), cw2, ch2, px, py))
        scored.sort()
        for _, cw2, ch2, px, py in scored:
            size_pairs.append((cw2, ch2, px, py))

    if sizes_x2 and sizes_y2:
        _append_pairs(sizes_x2, sizes_y2)
    # 交錯已有 2× 矩形單元時不要再搜 1×（半週期會把交錯列拼成方格／花生點）
    if not (stagger and sizes_x2 and sizes_y2):
        _append_pairs(sizes_x, sizes_y)
    # 去重
    uniq_pairs: list[tuple[int, int, int, int]] = []
    seen_sz: set[tuple[int, int]] = set()
    for pr in size_pairs:
        if (pr[0], pr[1]) not in seen_sz:
            seen_sz.add((pr[0], pr[1]))
            uniq_pairs.append(pr)
    size_pairs = uniq_pairs[:6]
    if not size_pairs:
        return None, "無可用裁切尺寸", 2, None

    _lg(f"  → 點綴晶格：搜尋 {len(size_pairs)} 組裁切尺寸…")
    # 積分圖加速邊緣對稱分；質心陣列加速空隙距離
    fg_integ = cv2.integral(fg.astype(np.float32))
    cents_xy = (
        np.array([(cx, cy) for cy, cx, _ in cents_pitch], dtype=np.float64)
        if cents_pitch
        else np.zeros((0, 2), dtype=np.float64)
    )

    def _rect_mean(y0: int, x0: int, y1: int, x1: int) -> float:
        area = max(1, (y1 - y0) * (x1 - x0))
        s = (
            float(fg_integ[y1, x1])
            - float(fg_integ[y0, x1])
            - float(fg_integ[y1, x0])
            + float(fg_integ[y0, x0])
        )
        return s / area

    def _sym_score_at(dx: int, dy: int, cw: int, ch: int) -> float:
        b = max(1, min(band, ch // 8, cw // 8))
        left = _rect_mean(dy, dx, dy + ch, dx + b)
        right = _rect_mean(dy, dx + cw - b, dy + ch, dx + cw)
        top = _rect_mean(dy, dx, dy + b, dx + cw)
        bot = _rect_mean(dy + ch - b, dx, dy + ch, dx + cw)
        return (
            left
            + right
            + top
            + bot
            + abs(left - right) * 3.0
            + abs(top - bot) * 3.0
        )

    def _gap_clearance_fast(dx: int, dy: int, cw: int, ch: int) -> float:
        if cents_xy.shape[0] == 0:
            return 0.0
        cx = cents_xy[:, 0]
        cy = cents_xy[:, 1]
        inside = (
            (cx >= dx - 2)
            & (cx < dx + cw + 2)
            & (cy >= dy - 2)
            & (cy < dy + ch + 2)
        )
        if not np.any(inside):
            return 0.0
        cx = cx[inside]
        cy = cy[inside]
        rx = np.minimum(np.abs(cx - dx), np.abs(cx - (dx + cw)))
        ry = np.minimum(np.abs(cy - dy), np.abs(cy - (dy + ch)))
        return float(np.min(np.minimum(rx, ry)))

    coarse: list[tuple[tuple[int, float], int, int, int, int, np.ndarray, str]] = []
    min_px_i = max(48, int(round(min_pitch)))
    for si, (cw, ch, px, py) in enumerate(size_pairs):
        _lg(
            f"  → 點綴晶格：尺寸組 {si + 1}/{len(size_pairs)} "
            f"週期 {px}×{py} → 單元 {cw}×{ch}"
        )
        # 雙重保險：非整數倍週期直接跳過；過短週期禁止硬通過候選
        if cw % px != 0 or ch % py != 0:
            continue
        if px < min_px_i or py < min_px_i:
            continue
        step = 2 if max(px, py) >= 70 else 1
        if max(px, py) >= 140:
            step = max(step, 3)
        seeds = [
            (0, 0),
            (seed_ox % px, seed_oy % py),
            (px // 2, py // 2),
            (px // 4, py // 4),
            (px // 2, 0),
            (0, py // 2),
        ]
        # 大花質心空隙中點（正交格用 1/2 週期；磚縫／交錯用 1/4）
        gap_frac = 0.25 if stagger else 0.5
        for cy, cx, _ in cents_pitch[:10]:
            seeds.append(
                (
                    int(round(cx - px * gap_frac)) % px,
                    int(round(cy - py * gap_frac)) % py,
                )
            )
            if stagger:
                seeds.append(
                    (
                        int(round(cx - px * 0.75)) % px,
                        int(round(cy - py * 0.75)) % py,
                    )
                )
        scored: list[tuple[float, int, int]] = []
        seen_xy: set[tuple[int, int]] = set()

        def _add_coarse(dx: int, dy: int) -> None:
            if dx < 0 or dy < 0 or dx + cw > w or dy + ch > h:
                return
            key = (dx, dy)
            if key in seen_xy:
                return
            seen_xy.add(key)
            scored.append((_sym_score_at(dx, dy, cw, ch), dx, dy))

        rad = max(6, min(px, py) // 5)
        for sx, sy in seeds:
            for dx in range(max(0, sx - rad), min(px, sx + rad + 1), step):
                for dy in range(max(0, sy - rad), min(py, sy + rad + 1), step):
                    _add_coarse(dx, dy)
        # 種子鄰域精修（浮點取模後常需 ±1～2）
        for dx in range(max(0, seed_ox - 2), min(px, seed_ox + 3)):
            for dy in range(max(0, seed_oy - 2), min(py, seed_oy + 3)):
                _add_coarse(dx, dy)
        gx_step = max(step, max(2, px // 4))
        gy_step = max(step, max(2, py // 4))
        for dx in range(0, px, gx_step):
            for dy in range(0, py, gy_step):
                _add_coarse(dx, dy)
        # 相位也可超過單週期（裁切原點）
        for dx in range(0, max(1, w - cw + 1), max(step, px // 2 or 1)):
            for dy in range(0, max(1, h - ch + 1), max(step, py // 2 or 1)):
                _add_coarse(dx, dy)

        # 粗分 Top-N 再加空隙距離重排
        scored.sort(key=lambda t: t[0])
        rescored: list[tuple[float, int, int]] = []
        for e0, dx, dy in scored[:32]:
            e = e0 - _gap_clearance_fast(dx, dy, cw, ch) * 0.2
            rescored.append((e, dx, dy))
        rescored.sort(key=lambda t: t[0])
        scored = rescored

        local: tuple[tuple[int, float], int, int, np.ndarray] | None = None
        seen: set[tuple[int, int]] = set()
        # 搜尋用輕量評分；完整 _rank_tile 只用於局部優勝者
        light_hits: list[tuple[tuple[int, float], int, int, np.ndarray]] = []
        for _, dx, dy in scored[:6]:
            if (dx, dy) in seen:
                continue
            seen.add((dx, dy))
            tile = arr[dy : dy + ch, dx : dx + cw]
            key = _rank_tile_light(
                tile, bg, threshold, med, px=px, py=py, cut_med=cut_med
            )
            clr = _gap_clearance_fast(dx, dy, cw, ch)
            key2 = (key[0], key[1] - clr * 3.0)
            light_hits.append((key2, dx, dy, tile))
        light_hits.sort(key=lambda t: t[0])
        for key2, dx, dy, tile in light_hits[:3]:
            fg_t = fg[dy : dy + ch, dx : dx + cw]
            tier, sc, gx, gy = _rank_tile(
                tile,
                bg,
                threshold,
                med,
                px=px,
                py=py,
                cut_med=cut_med,
                fg_merged=fg_t,
            )
            key = (tier, sc)
            clr = _gap_clearance_fast(dx, dy, cw, ch)
            gap_pen = (abs(gx - 1.0) + abs(gy - 1.0)) * 80.0
            key_full = (key[0], key[1] + gap_pen - clr * 3.0)
            if local is None or key_full < local[0]:
                local = (key_full, dx, dy, tile.copy())
            if (
                key[0] == 0
                and key[1] < 12.0
                and clr >= max(6.0, min(px, py) * 0.12)
                and abs(gx - 1.0) < 0.15
                and abs(gy - 1.0) < 0.15
            ):
                break
        if local is None:
            continue
        key0, best_dx, best_dy, tile0 = local
        if key0[0] != 0 or key0[1] > 8.0:
            for dx in range(max(0, best_dx - 1), min(w - cw + 1, best_dx + 2)):
                for dy in range(max(0, best_dy - 1), min(h - ch + 1, best_dy + 2)):
                    tile = arr[dy : dy + ch, dx : dx + cw]
                    fg_t = fg[dy : dy + ch, dx : dx + cw]
                    tier, sc, gx, gy = _rank_tile(
                        tile,
                        bg,
                        threshold,
                        med,
                        px=px,
                        py=py,
                        cut_med=cut_med,
                        fg_merged=fg_t,
                    )
                    key = (tier, sc)
                    clr = _gap_clearance_fast(dx, dy, cw, ch)
                    gap_pen = (abs(gx - 1.0) + abs(gy - 1.0)) * 80.0
                    key2 = (key[0], key[1] + gap_pen - clr * 3.0)
                    if key2 < key0:
                        key0, best_dx, best_dy, tile0 = key2, dx, dy, tile.copy()
                    if key0[0] == 0 and key0[1] < 6.0 and clr >= 6.0:
                        break
                if key0[0] == 0 and key0[1] < 6.0:
                    break
        detail = f"晶格週期 {px}×{py}px → 單元 {cw}×{ch}"
        coarse.append((key0, best_dx, best_dy, px, py, tile0, detail))
        # 已有很好的硬通過：不必再掃其餘尺寸
        if key0[0] == 0 and key0[1] < 6.0:
            break
        if key0[0] == 0 and key0[1] < 8.0:
            # 繼續試其他尺寸，取更好硬通過
            continue

    if not coarse:
        return None, "無可用裁切", 2, None

    # 補試：2×／3× 週期 × 質心空隙相位（正交 1/2；磚縫 1/4）
    # 已有硬通過時略過，大幅縮短搜尋
    has_hard = any(t[0][0] == 0 and t[0][1] < 20.0 for t in coarse)
    if not has_hard:
        gap_frac = 0.25 if stagger else 0.5
        for px, py in {(pr[2], pr[3]) for pr in size_pairs}:
            for nx, ny in ((2, 2), (2, 3), (3, 2)):
                cw, ch = nx * px, ny * py
                if cw > w or ch > h or cw < min_side or ch < min_side:
                    continue
                for cy, cx, _ in cents_pitch[:4]:
                    fracs = (gap_frac, 0.75) if stagger else (gap_frac,)
                    for frac in fracs:
                        ox = int(round(cx - px * frac))
                        oy = int(round(cy - py * frac))
                        for dx, dy in (
                            (ox, oy),
                            (ox - 2, oy),
                            (ox + 2, oy),
                            (ox, oy - 2),
                            (ox, oy + 2),
                        ):
                            if dx < 0 or dy < 0 or dx + cw > w or dy + ch > h:
                                continue
                            tile = arr[dy : dy + ch, dx : dx + cw]
                            # 先輕量過濾
                            if _rank_tile_light(
                                tile, bg, threshold, med, px=px, py=py, cut_med=cut_med
                            )[0] > 0:
                                continue
                            fg_t = fg[dy : dy + ch, dx : dx + cw]
                            tier, sc, gx, gy = _rank_tile(
                                tile,
                                bg,
                                threshold,
                                med,
                                px=px,
                                py=py,
                                cut_med=cut_med,
                                fg_merged=fg_t,
                            )
                            key = (tier, sc)
                            gap_pen = (abs(gx - 1.0) + abs(gy - 1.0)) * 80.0
                            clr = _gap_clearance_fast(dx, dy, cw, ch)
                            key2 = (key[0], key[1] + gap_pen - clr * 3.0)
                            detail = f"晶格週期 {px}×{py}px → 單元 {cw}×{ch}"
                            coarse.append((key2, dx, dy, px, py, tile.copy(), detail))
                            if key[0] == 0 and gap_pen < 8.0 and key[1] < 40.0:
                                break
                        else:
                            continue
                        break
                    else:
                        continue
                    break

    # 同 tier：偏好間距分低、週期重複次數少（磚縫單元常是 2×2 格，勿被 3×3 大圖蓋過）
    def _sel_key(t: tuple) -> tuple:
        key, _dx, _dy, px, py, tile, _d = t
        th, tw = tile.shape[:2]
        nrep = tw / max(float(px), 1.0) + th / max(float(py), 1.0)
        cov = (th * tw) / float(h * w)
        return (key[0], key[1] + nrep * 5.0 - cov * 2.0)

    coarse.sort(key=_sel_key)
    # 最終複核：嚴格間距；切開檢查用整體 med（勿用大花面積把切開漏掉）
    for key, dx, dy, px, py, tile, detail in coarse[:6]:
        th, tw = tile.shape[:2]
        rk_t, rk_s, gx, gy = _rank_tile(
            tile,
            bg,
            threshold,
            med,
            px=px,
            py=py,
            cut_med=cut_med,
            fg_merged=fg[dy : dy + th, dx : dx + tw],
        )
        rk = (rk_t, rk_s)
        gap_ok = 0.90 <= gx <= 1.15 and 0.90 <= gy <= 1.15
        if rk[0] <= 1 and gap_ok:
            return tile, detail, rk[0], (dx, dy, tw, th, px, py)
    best_key, bdx, bdy, bpx, bpy, best_tile, best_detail = coarse[0]
    bth, btw = best_tile.shape[:2]
    return best_tile, best_detail, best_key[0], (bdx, bdy, btw, bth, bpx, bpy)


def try_make_discrete_seamless(
    arr: np.ndarray,
    bg: Sequence[int],
    threshold: float = 40.0,
    log: Callable[[str], None] | None = None,
) -> tuple[np.ndarray, str]:
    """構造裁切為主；禁止邊緣相關備選勝出。

    超大圖會在縮圖上搜尋晶格，再映射回原圖裁切並用原圖接縫評分（不放寬標準）。
    """
    def _lg(msg: str) -> None:
        if log is not None:
            log(msg)

    full_h, full_w = arr.shape[:2]
    work = arr
    scale = 1.0
    if max(full_h, full_w) > _DISCRETE_FULLRES_TRIGGER:
        work, scale = _downscale_for_discrete_analysis(arr)
        wh, ww = work.shape[:2]
        _lg(
            f"  → 點綴晶格：原圖 {full_w}×{full_h} 過大，"
            f"先以 {ww}×{wh} 搜尋再映射回原圖（接縫仍按原圖評）"
        )

    t_fg = time.perf_counter()
    _lg("  → 點綴晶格：前景／質心分析…")
    fg_src = _fg_mask_merged(work, bg, threshold)
    med = _interior_median_motif_area(fg_src)
    cut_med = _large_motif_median(work, bg, threshold, med) if med >= 40 else med
    _lg(f"  → 點綴晶格：前景完成（{time.perf_counter() - t_fg:.1f}s）")

    med_rank = med * scale * scale
    cut_rank = (
        (cut_med * scale * scale) if (cut_med is not None and med >= 40) else None
    )

    def _rk(
        tile: np.ndarray,
        px: int = 0,
        py: int = 0,
        fg_merged: np.ndarray | None = None,
        *,
        med_use: float | None = None,
        cut_use: float | None = None,
    ) -> tuple[int, float]:
        m = med if med_use is None else med_use
        c = (cut_med if med >= 40 else None) if cut_use is None else cut_use
        t, s, _, _ = _rank_tile(
            tile,
            bg,
            threshold,
            m,
            px=px,
            py=py,
            cut_med=c,
            fg_merged=fg_merged,
        )
        return t, s

    # 源圖接縫已極低：直接保留，跳過昂貴晶格搜尋。
    # JPEG 爪印等邊緣碎點常被誤判切開；色差接縫夠低時以接縫為準。
    sv0, sh0 = _tile_seam_scores(work)
    if (sv0 + sh0) <= 3.5 and max(sv0, sh0) <= 2.5:
        cuts0 = _cross_seam_cut_count(
            work, bg, threshold, med=cut_med, adjust_large_med=False
        )
        if cuts0 == 0 or (sv0 + sh0) <= 3.0:
            return arr.copy(), "原圖（接縫已低）"

    box: tuple[int, int, int, int, int, int] | None = None
    # 非規則散點：跳過昂貴晶格相位搜尋（仍可走質心滾動／原圖）
    if looks_like_regular_lattice(work, bg, threshold, fg_merged=fg_src):
        cropped, detail, _tier, box = try_discrete_lattice_crop(
            work,
            bg,
            threshold,
            fg_merged=fg_src,
            med=med,
            cut_med=cut_med if med >= 40 else None,
            log=log,
        )
    else:
        cropped, detail = None, "非規則晶格"
        _lg("  → 點綴晶格：判定非規則，跳過晶格裁切搜尋")

    candidates: list[tuple[tuple[int, float], np.ndarray, str]] = []
    if cropped is not None and box is not None:
        dx_s, dy_s, cw_s, ch_s, px_s, py_s = box
        if scale != 1.0:
            dx, dy, cw, ch, px_r, py_r = _map_lattice_box_to_full(
                dx_s, dy_s, cw_s, ch_s, px_s, py_s, scale, full_h, full_w
            )
            # 大裁切禁止對每個偏移做完整 _rank_tile（會卡死）；
            # 先用接縫色差輕量選相位，再對優勝者做一次完整評分。
            rad = 1 if max(cw, ch) >= 4000 else max(1, min(3, int(round(scale))))
            _lg(
                f"  → 點綴晶格：映射原圖裁切 {cw}×{ch} @ ({dx},{dy})，"
                f"週期 {px_r}×{py_r}，接縫精修 ±{rad}px…"
            )
            best_seam: float | None = None
            best_xy = (dx, dy)
            for ddx in range(-rad, rad + 1):
                for ddy in range(-rad, rad + 1):
                    x = min(max(0, dx + ddx), max(0, full_w - cw))
                    y = min(max(0, dy + ddy), max(0, full_h - ch))
                    sv, shs = _tile_seam_scores(arr[y : y + ch, x : x + cw])
                    seam = sv + shs
                    if best_seam is None or seam < best_seam:
                        best_seam = seam
                        best_xy = (x, y)
            x, y = best_xy
            cropped = arr[y : y + ch, x : x + cw].copy()
            _lg("  → 點綴晶格：原圖裁切評分…")
            if max(cw, ch) >= 4000:
                # 超大單元避免完整連通域評分卡死；輕量分 + 接縫色差
                key = _rank_tile_light(
                    cropped,
                    bg,
                    threshold,
                    med_rank,
                    px=px_r,
                    py=py_r,
                    cut_med=cut_rank,
                )
                sv, shs = _tile_seam_scores(cropped)
                key = (key[0], key[1] + (sv + shs) * 0.25)
            else:
                key = _rk(
                    cropped,
                    px_r,
                    py_r,
                    med_use=med_rank,
                    cut_use=cut_rank,
                )
            detail = (
                f"晶格週期 {px_r}×{py_r}px → 單元 {cw}×{ch}（大圖縮放分析）"
            )
        else:
            px_r, py_r = px_s, py_s
            key = _rk(cropped, px_r, py_r)

        if px_r and py_r and (px_r < 80 or py_r < 80):
            key = (max(key[0], 1) + 1, key[1] + 80.0)
        if px_r and py_r:
            cents_n = _fg_centroids(
                fg_src,
                min_area=max(40, int(med * 0.08) if med else 40),
                skip_edge=True,
            )
            groups_n = _dual_scale_groups(cents_n)
            pts_n = groups_n[0] if groups_n is not None else [
                (cy, cx) for cy, cx, _ in cents_n
            ]
            nn_n = _nn_median_spacing(pts_n) * scale
            if nn_n >= 80 and (px_r < nn_n * 0.55 or py_r < nn_n * 0.55):
                key = (max(key[0], 1) + 1, key[1] + 90.0)
        candidates.append((key, cropped, f"點綴晶格({detail})"))
        if key[0] == 0:
            cv, chs = _tile_seam_scores(cropped)
            ov, oh = _tile_seam_scores(arr if scale != 1.0 else work)
            seam_ok = (cv + chs) <= (ov + oh) * 1.02 + 1.0
            abs_ok = (cv + chs) <= 28.0 and max(cv, chs) <= 20.0
            if seam_ok and abs_ok:
                return cropped, f"點綴晶格({detail})"

    # 僅質心滾動備選。原圖已有碰邊殘片時禁止：滾動只會把半隻動物捲到畫面中央。
    orig_cuts = _cross_seam_cut_count(
        work, bg, threshold, med=cut_med, adjust_large_med=False
    )
    if med >= 40 and orig_cuts == 0:
        cents = _fg_centroids(fg_src, min_area=max(40, int(med * 0.08)), skip_edge=True)
        if len(cents) < 4:
            cents = _fg_centroids(fg_src, min_area=max(40, int(med * 0.08)))
        groups_ph = _dual_scale_groups(cents)
        if groups_ph is not None:
            cents = [(cy, cx, 1000) for cy, cx in groups_ph[0]]
        if len(cents) >= 4:
            _lg("  → 點綴晶格：質心相位搜尋…")
            wh, ww = work.shape[:2]
            ox, oy, px, py = _centroid_seam_offsets(cents, wh, ww, med)
            best_r = np.roll(np.roll(work, -oy, 0), -ox, 1)
            best_light = _rank_tile_light(
                best_r, bg, threshold, med, cut_med=cut_med
            )
            best_ox, best_oy = ox, oy
            rad = max(3, int(round(max(px, py, 16) * 0.10)))
            for dx in range(-rad, rad + 1, 2):
                cand = np.roll(np.roll(work, -oy, 0), -(ox + dx), 1)
                k = _rank_tile_light(cand, bg, threshold, med, cut_med=cut_med)
                if k < best_light:
                    best_light, best_ox = k, (ox + dx) % ww
                    best_r = cand
            for dy in range(-rad, rad + 1, 2):
                cand = np.roll(np.roll(work, -(oy + dy), 0), -best_ox, 1)
                k = _rank_tile_light(cand, bg, threshold, med, cut_med=cut_med)
                if k < best_light:
                    best_light = k
                    best_oy = (oy + dy) % wh
                    best_r = cand
            if scale != 1.0:
                ox_f = int(round(best_ox * scale)) % full_w
                oy_f = int(round(best_oy * scale)) % full_h
                best_full = np.roll(np.roll(arr, -oy_f, 0), -ox_f, 1)
                best_k = _rk(
                    best_full, med_use=med_rank, cut_use=cut_rank
                )
                join_pen = _rolled_interior_join_penalty(best_full, ox_f, oy_f)
                if join_pen >= 6.0:
                    best_k = (max(best_k[0], 1) + 1, best_k[1] + join_pen * 4.0)
                candidates.append((best_k, best_full, "點綴質心相位"))
            else:
                best_k = _rk(best_r)
                join_pen = _rolled_interior_join_penalty(best_r, best_ox, best_oy)
                if join_pen >= 6.0:
                    best_k = (max(best_k[0], 1) + 1, best_k[1] + join_pen * 4.0)
                candidates.append((best_k, best_r, "點綴質心相位"))

    # 原圖候選：大圖用縮圖打分，勝出仍回傳原解析度
    if scale != 1.0:
        candidates.append(
            (_rk(work, fg_merged=fg_src), arr.copy(), "原圖")
        )
    else:
        candidates.append((_rk(arr, fg_merged=fg_src), arr.copy(), "原圖"))

    ov, oh = _tile_seam_scores(arr if scale != 1.0 else work)
    base_seam = ov + oh
    adjusted: list[tuple[tuple[int, float], np.ndarray, str]] = []
    for key, tile, msg in candidates:
        cv, chs = _tile_seam_scores(tile)
        seam = cv + chs
        k0, k1 = key
        if seam > base_seam * 1.15 + 3.0:
            k0 = max(k0, 1) + 1
            k1 = k1 + (seam - base_seam) * 1.5
        elif seam > base_seam and ("晶格" in msg or "質心" in msg):
            k0 = max(k0, 2)
            k1 = k1 + (seam - base_seam) * 3.0
        adjusted.append(((k0, k1), tile, msg))
    adjusted.sort(key=lambda t: t[0])
    best_key, best_tile, best_msg = adjusted[0]

    if best_key[0] == 0:
        return best_tile, best_msg
    if best_key[0] == 1:
        return best_tile, f"{best_msg}（軟通過 cuts=0）"
    return best_tile, f"FAIL 點綴未對齊（{best_msg}）"


def large_stamp_like(
    arr: np.ndarray,
    bg: Sequence[int],
    threshold: float = 40.0,
) -> bool:
    """獨立大圖章（動物／貼紙），相對滿鋪紋理。"""
    # 超大圖先縮再判斷，面積門檻依縮放補償
    h0, w0 = arr.shape[:2]
    if max(h0, w0) > _DISCRETE_FULLRES_TRIGGER:
        work, scale = _downscale_for_discrete_analysis(arr, max_side=1600)
        area_s = scale * scale
    else:
        work, area_s = arr, 1.0
    fg = _fg_mask_merged(work, bg, threshold)
    # 密鋪／滿花：前景覆蓋高，禁止當大圖章（否則半幅偏移被擋死）
    if float(np.mean(fg)) > 0.22:
        return False
    med = _interior_median_motif_area(fg)
    min_area = max(40, int(med * 0.08) if med else 40)
    # 含碰邊塊：密花常被合併成少數超大邊塊，skip_edge 會誤成「稀疏大圖章」
    cents_all = _fg_centroids(fg, min_area=min_area, skip_edge=False)
    if len(cents_all) >= 12:
        return False
    areas_all = [a for *_, a in cents_all]
    if areas_all:
        mx_all = max(areas_all)
        # 碰邊超大連通域＝滿鋪花網，不是單隻貼紙
        if mx_all >= fg.size * 0.18 and len(cents_all) >= 5:
            return False
    cents = _fg_centroids(fg, min_area=min_area, skip_edge=True)
    if len(cents) < 4:
        return False
    # 碎花／拼布格：很多中等塊，勿當大圖章（否則清邊會咬出白齒縫）
    if len(cents) > 36:
        return False
    areas = [a for *_, a in cents]
    large_thr = 4000.0 / area_s
    stamp_thr = 8000.0 / area_s
    large_n = sum(1 for a in areas if a >= large_thr)
    if large_n > 10:
        return False
    # 多個大塊重複陣列／滿鋪：非「單隻動物圖章」
    if large_n >= 3 and len(cents) >= 8:
        return False
    if len(cents) >= 16:
        return False
    mx = max(areas)
    med_a = float(np.median(areas))
    return mx >= stamp_thr and mx >= med_a * 5.0


cross_seam_cut_count = _cross_seam_cut_count
wrap_gap_ratios = _wrap_gap_ratios
discrete_integrity_ok = _discrete_integrity_ok
