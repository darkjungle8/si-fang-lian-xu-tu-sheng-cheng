"""四方連續：清掉碰邊殘花後，用完整圖案環繞補回密度（不模糊）。"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from PIL import Image, ImageFilter
import cv2

from app.color_utils import color_distance, detect_background
from app.discrete_lattice import (
    _cross_seam_cut_count,
    _discrete_integrity_ok,
    _wrap_gap_ratios,
    large_stamp_like,
    looks_like_regular_lattice,
)


def _to_rgb_array(image: Image.Image, bg: Sequence[int]) -> np.ndarray:
    """轉為不透明 RGB；若有透明通道則先合成到背景色。"""
    if image.mode == "RGBA":
        background = Image.new("RGBA", image.size, (*tuple(bg), 255))
        composited = Image.alpha_composite(background, image)
        return np.asarray(composited.convert("RGB"), dtype=np.uint8)
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _foreground_mask(arr: np.ndarray, bg: Sequence[int], threshold: float) -> np.ndarray:
    return color_distance(arr, bg) > threshold


def _edge_band_mask(h: int, w: int, margin_px: int) -> np.ndarray:
    mx = max(0, min(margin_px, w // 2))
    my = max(0, min(margin_px, h // 2))
    edge = np.zeros((h, w), dtype=bool)
    if my > 0:
        edge[:my, :] = True
        edge[h - my :, :] = True
    if mx > 0:
        edge[:, :mx] = True
        edge[:, w - mx :] = True
    return edge


@dataclass
class MotifStamp:
    """一塊完整前景圖案（含外接矩形 patch 與 mask）。"""

    patch: np.ndarray  # (hm, wm, 3) uint8
    mask: np.ndarray  # (hm, wm) bool
    area: int
    # 在 patch 內的質心（相對座標）
    cy: float
    cx: float


def _iter_components(
    fg: np.ndarray,
) -> list[list[tuple[int, int]]]:
    """4-連通前景，回傳各連通域座標列表（OpenCV；與純 Python BFS 等價）。"""
    n, labels = cv2.connectedComponents(fg.astype(np.uint8), connectivity=4)
    components: list[list[tuple[int, int]]] = []
    for i in range(1, n):
        ys, xs = np.where(labels == i)
        components.append(list(zip(ys.tolist(), xs.tolist())))
    return components


def _component_to_stamp_from_mask(
    arr: np.ndarray,
    mask_comp: np.ndarray,
    *,
    dilate_px: int = 4,
) -> MotifStamp:
    """由布林／0-1 mask 建 MotifStamp。"""
    h, w = arr.shape[:2]
    mask_full = mask_comp.astype(np.uint8)
    ys_o, xs_o = np.where(mask_full)
    if dilate_px > 0:
        k = 2 * int(dilate_px) + 1
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask_full = cv2.dilate(mask_full, ker)
    ys, xs = np.where(mask_full)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    hm, wm = y1 - y0 + 1, x1 - x0 + 1
    patch = np.zeros((hm, wm, 3), dtype=np.uint8)
    mask = mask_full[y0 : y1 + 1, x0 : x1 + 1].astype(bool)
    patch[mask] = arr[y0 : y1 + 1, x0 : x1 + 1][mask]
    area = int(ys_o.size)
    cy = float(np.mean(ys_o) - y0)
    cx = float(np.mean(xs_o) - x0)
    return MotifStamp(patch=patch, mask=mask, area=area, cy=cy, cx=cx)


def _component_to_stamp(
    arr: np.ndarray,
    coords: list[tuple[int, int]],
    *,
    dilate_px: int = 4,
) -> MotifStamp:
    h, w = arr.shape[:2]
    mask_full = np.zeros((h, w), dtype=np.uint8)
    if coords:
        ys_o = np.array([c[0] for c in coords], dtype=np.int32)
        xs_o = np.array([c[1] for c in coords], dtype=np.int32)
        mask_full[ys_o, xs_o] = 1
    return _component_to_stamp_from_mask(arr, mask_full, dilate_px=dilate_px)


def _clear_edge_ruins_motifs(
    src: np.ndarray,
    filled: np.ndarray,
    bg: Sequence[int],
    threshold: float,
) -> bool:
    """清邊補花是否把動物／圖章抹成地色或留下淡鬼影。"""
    d0 = color_distance(src, bg)
    d1 = color_distance(filled, bg)
    fg0 = d0 > threshold
    fg1 = d1 > threshold
    erased = float(np.mean(fg0 & ~fg1))
    # 抹掉過多前景（柯基／貼紙清邊常見）
    if erased >= 0.03:
        return True
    changed = float((filled != src).any(axis=2).mean())
    if changed >= 0.10 and erased >= 0.015:
        return True
    # 原前景處變成半透明／近地色殘影
    ghost = fg0 & (d1 > threshold * 0.2) & (d1 <= threshold)
    if float(np.mean(ghost)) >= 0.015 and erased >= 0.01:
        return True
    return False


def extract_interior_motifs(
    arr: np.ndarray,
    bg: Sequence[int],
    threshold: float,
    margin_px: int,
    min_area: int = 40,
) -> list[MotifStamp]:
    """取出未碰邊緣帶的完整圖案，作為補花素材。"""
    fg = _foreground_mask(arr, bg, threshold)
    edge = _edge_band_mask(*arr.shape[:2], margin_px)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        fg.astype(np.uint8), connectivity=4
    )
    motifs: list[MotifStamp] = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        mask_i = labels == i
        if np.any(mask_i & edge):
            continue
        motifs.append(_component_to_stamp_from_mask(arr, mask_i))
    motifs.sort(key=lambda m: m.area, reverse=True)
    return motifs


def remove_edge_touching_components(
    arr: np.ndarray,
    bg: Sequence[int],
    threshold: float,
    margin_px: int,
) -> tuple[np.ndarray, list[tuple[float, float, int]]]:
    """
    刪除碰到邊緣帶的整塊連通前景。
    回傳 (清理後圖, 被刪圖案的質心與面積列表) 供後續補花定位。
    """
    out = arr.copy()
    h, w = out.shape[:2]
    removed: list[tuple[float, float, int]] = []
    if margin_px <= 0:
        return out, removed

    fg = _foreground_mask(out, bg, threshold)
    edge = _edge_band_mask(h, w, margin_px)
    bg_rgb = np.array(bg, dtype=np.uint8)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(
        fg.astype(np.uint8), connectivity=4
    )
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    for i in range(1, n):
        mask_i = labels == i
        if not np.any(mask_i & edge):
            continue
        area = int(stats[i, cv2.CC_STAT_AREA])
        # OpenCV centroid 是 (x, y)
        cx, cy = float(centroids[i][0]), float(centroids[i][1])
        removed.append((cy, cx, area))
        mask_c = cv2.dilate(mask_i.astype(np.uint8), ker)
        out[mask_c.astype(bool)] = bg_rgb
    return out, removed


def stamp_motif_wrapped(
    canvas: np.ndarray,
    motif: MotifStamp,
    center_y: float,
    center_x: float,
) -> np.ndarray:
    """
    以環面座標貼上圖案（超出右邊界的部分出現在左邊，依此類推）。
    硬貼、不模糊。
    """
    h, w = canvas.shape[:2]
    out = canvas
    top = int(round(center_y - motif.cy))
    left = int(round(center_x - motif.cx))
    my, mx = np.where(motif.mask)
    if len(my) == 0:
        return out
    ty = (top + my) % h
    tx = (left + mx) % w
    out[ty, tx] = motif.patch[my, mx]
    return out


def _overlap_ratio(
    canvas: np.ndarray,
    motif: MotifStamp,
    center_y: float,
    center_x: float,
    occupied: np.ndarray,
) -> float:
    h, w = canvas.shape[:2]
    top = int(round(center_y - motif.cy))
    left = int(round(center_x - motif.cx))
    my, mx = np.where(motif.mask)
    if len(my) == 0:
        return 1.0
    ty = (top + my) % h
    tx = (left + mx) % w
    return float(np.mean(occupied[ty, tx]))


def _pick_motif(motifs: list[MotifStamp], target_area: int, rng: np.random.Generator) -> MotifStamp:
    if len(motifs) == 1:
        return motifs[0]
    areas = np.array([m.area for m in motifs], dtype=np.float64)
    # 偏好面積接近被刪圖案者
    dist = np.abs(areas - float(target_area))
    weights = 1.0 / (1.0 + dist / max(float(target_area), 1.0))
    weights = weights / weights.sum()
    idx = int(rng.choice(len(motifs), p=weights))
    return motifs[idx]


def _torus_distance(
    y1: float, x1: float, y2: float, x2: float, h: int, w: int
) -> float:
    dy = abs(y1 - y2)
    dx = abs(x1 - x2)
    dy = min(dy, h - dy)
    dx = min(dx, w - dx)
    return float(np.sqrt(dy * dy + dx * dx))


def _nudge_off_corners(cy: float, cx: float, h: int, w: int) -> tuple[float, float]:
    """把過近四角的放置點往邊中點挪，避免 2×2 中心十字擠成一團。"""
    corner = 0.14 * min(h, w)
    near_left = cx < corner
    near_right = cx > w - corner
    near_top = cy < corner
    near_bottom = cy > h - corner
    if (near_left or near_right) and (near_top or near_bottom):
        # 角上：推到較長邊的中段
        if near_left or near_right:
            cy = h * 0.35 if near_top else h * 0.65
        if near_top or near_bottom:
            cx = w * 0.35 if near_left else w * 0.65
    return cy % h, cx % w


def refill_with_wrapped_motifs(
    arr: np.ndarray,
    bg: Sequence[int],
    threshold: float,
    motifs: list[MotifStamp],
    removed: list[tuple[float, float, int]],
    seed: int = 42,
    max_overlap: float = 0.08,
) -> np.ndarray:
    """
    在被清掉的位置用完整圖案環繞貼回；控制間距，避免接縫擠成一團。
    放不下就不硬塞（寧願略疏，不要重疊堆疊）。
    """
    if not motifs:
        return arr

    out = arr.copy()
    h, w = out.shape[:2]
    rng = np.random.default_rng(seed)
    occupied = _foreground_mask(out, bg, threshold)

    # 只補被刪數量的一部分，並做空間稀疏化
    placements: list[tuple[float, float, int]] = []
    # 大面積先補，較能代表原密度
    ordered = sorted(removed, key=lambda t: t[2], reverse=True)
    large_cut = sum(1 for t in ordered if t[2] >= 4000)
    if large_cut > 0:
        # 動物等大圖章：被切掉的都補回（環繞貼），否則邊緣空洞
        budget = len(ordered)
    else:
        # 最多補 removed 的 70%，且不超過素材數*2
        budget = min(
            len(ordered),
            max(1, int(round(len(ordered) * 0.7))),
            max(2, len(motifs) * 2),
        )
    for cy, cx, area in ordered[:budget]:
        # 大圖章要留在邊緣才能環繞接上；小碎花仍避開四角擠堆
        if area < 4000:
            cy, cx = _nudge_off_corners(cy, cx, h, w)
        placements.append((cy, cx, area))

    # 邊緣若仍明顯偏空，少量補點（嚴格上限）
    edge_wide = _edge_band_mask(h, w, max(8, min(h, w) // 8))
    edge_fg = float(np.mean(occupied[edge_wide])) if np.any(edge_wide) else 0.0
    interior = ~edge_wide
    int_fg = float(np.mean(occupied[interior])) if np.any(interior) else 0.0
    if int_fg > 0.02 and edge_fg < int_fg * 0.35:
        need = min(3, max(0, int(round((int_fg * 0.5 - edge_fg) * np.count_nonzero(edge_wide) / max(motifs[0].area, 1)))))
        band = max(8, min(h, w) // 7)
        for _ in range(need):
            side = int(rng.integers(0, 4))
            if side == 0:
                cy, cx = float(rng.uniform(band * 0.3, band)), float(rng.uniform(w * 0.2, w * 0.8))
            elif side == 1:
                cy, cx = float(rng.uniform(h - band, h - band * 0.3)), float(rng.uniform(w * 0.2, w * 0.8))
            elif side == 2:
                cy, cx = float(rng.uniform(h * 0.2, h * 0.8)), float(rng.uniform(band * 0.3, band))
            else:
                cy, cx = float(rng.uniform(h * 0.2, h * 0.8)), float(rng.uniform(w - band, w - band * 0.3))
            cy, cx = _nudge_off_corners(cy, cx, h, w)
            placements.append((cy, cx, motifs[0].area))

    placed_centers: list[tuple[float, float, float]] = []  # cy, cx, min_radius

    for cy, cx, area in placements:
        motif = _pick_motif(motifs, area, rng)
        radius = max(12.0, 0.9 * float(np.sqrt(motif.area / np.pi)) * 2.2)
        overlap_lim = 0.14 if motif.area >= 4000 else max_overlap

        # 與已放置朵保持環面距離
        too_close = any(
            _torus_distance(cy, cx, py, px, h, w) < max(radius, pr) * 0.85
            for py, px, pr in placed_centers
        )
        if too_close:
            # 嘗試抖動找空位
            found = False
            for _ in range(20):
                ny = float((cy + rng.normal(0, radius * 0.6)) % h)
                nx = float((cx + rng.normal(0, radius * 0.6)) % w)
                if motif.area < 4000:
                    ny, nx = _nudge_off_corners(ny, nx, h, w)
                if any(
                    _torus_distance(ny, nx, py, px, h, w) < max(radius, pr) * 0.85
                    for py, px, pr in placed_centers
                ):
                    continue
                if _overlap_ratio(out, motif, ny, nx, occupied) <= overlap_lim:
                    cy, cx = ny, nx
                    found = True
                    break
            if not found:
                continue  # 放棄，不硬塞
        else:
            # 檢查重疊；不行就抖動，再不放
            ok = _overlap_ratio(out, motif, cy, cx, occupied) <= overlap_lim
            if not ok:
                found = False
                for _ in range(16):
                    ny = float((cy + rng.normal(0, radius * 0.5)) % h)
                    nx = float((cx + rng.normal(0, radius * 0.5)) % w)
                    if motif.area < 4000:
                        ny, nx = _nudge_off_corners(ny, nx, h, w)
                    if any(
                        _torus_distance(ny, nx, py, px, h, w) < max(radius, pr) * 0.85
                        for py, px, pr in placed_centers
                    ):
                        continue
                    if _overlap_ratio(out, motif, ny, nx, occupied) <= overlap_lim:
                        cy, cx = ny, nx
                        found = True
                        break
                if not found:
                    continue

        stamp_motif_wrapped(out, motif, cy, cx)
        top = int(round(cy - motif.cy))
        left = int(round(cx - motif.cx))
        my, mx = np.where(motif.mask)
        occupied[(top + my) % h, (left + mx) % w] = True
        placed_centers.append((cy, cx, radius))

    return out


def _luminance_map(arr: np.ndarray) -> np.ndarray:
    a = arr.astype(np.float64)
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def _box_blur_gray(gray: np.ndarray, radius: int) -> np.ndarray:
    r = max(0, int(radius))
    if r <= 0:
        return gray.astype(np.float64)
    im = Image.fromarray(np.clip(gray, 0, 255).astype(np.uint8), mode="L")
    im = im.filter(ImageFilter.BoxBlur(r))
    im = im.filter(ImageFilter.BoxBlur(max(1, r // 2)))
    return np.asarray(im, dtype=np.float64)


def _stripe_mask(arr: np.ndarray, bg: Sequence[int] | None = None) -> np.ndarray:
    """抽出條紋結構。有背景色時用色距，否則用亮度中位。"""
    if bg is not None:
        soft = _box_blur_gray(
            color_distance(arr, bg).astype(np.float64),
            max(2, min(arr.shape[:2]) // 100),
        )
        thr = float(np.median(soft))
        return (soft > thr).astype(np.float64)
    gray = _luminance_map(arr)
    soft = _box_blur_gray(gray, max(2, min(gray.shape) // 100))
    return (soft > float(np.median(soft))).astype(np.float64)


def _autocorr_best_periods(signal: np.ndarray, min_p: int, max_p: int, top_k: int = 5) -> list[tuple[int, float]]:
    s = signal.astype(np.float64)
    n = len(s)
    max_p = min(max_p, n // 2)
    min_p = max(4, min_p)
    if max_p <= min_p or n < min_p * 2:
        return []
    s = s - s.mean()
    denom = float(np.dot(s, s)) + 1e-12
    fft = np.fft.rfft(s, n=n * 2)
    ac = np.fft.irfft(fft * np.conj(fft), n=n * 2)[:n] / denom
    window = ac[min_p : max_p + 1]
    peaks: list[tuple[float, int]] = []
    for i in range(1, len(window) - 1):
        if window[i] >= window[i - 1] and window[i] >= window[i + 1] and window[i] >= 0.08:
            peaks.append((float(window[i]), min_p + i))
    if not peaks:
        i = int(np.argmax(window))
        if float(window[i]) >= 0.06:
            peaks.append((float(window[i]), min_p + i))
    peaks.sort(reverse=True)
    out: list[tuple[int, float]] = []
    for score, p in peaks:
        if all(abs(p - q) > max(2, p // 35) for q, _ in out):
            out.append((p, score))
        if len(out) >= top_k:
            break
    return out


def _diagonal_projection(gray: np.ndarray, sign: int = 1) -> np.ndarray:
    h, w = gray.shape
    g = gray.astype(np.float64)
    ys = np.arange(h, dtype=np.int32)[:, None]
    xs = np.arange(w, dtype=np.int32)[None, :]
    if sign >= 0:
        idx = (ys + xs).ravel()
    else:
        idx = (ys - xs + (w - 1)).ravel()
    n = h + w - 1
    acc = np.bincount(idx, weights=g.ravel(), minlength=n)
    cnt = np.bincount(idx, minlength=n).astype(np.float64)
    return acc / np.maximum(cnt, 1.0)


def _repeat_error(field: np.ndarray, period: int, axis: int) -> float:
    """相隔 period 的內容重複誤差（真週期驗證）。"""
    if period < 4:
        return 1e9
    band = max(2, min(6, period // 20))
    if axis == 0:
        length = field.shape[1]
        if period + band > length:
            return 1e9
        # 向量化：用捲動差
        a = field[:, : length - period]
        b = field[:, period:]
        return float(np.mean(np.abs(a - b)))
    length = field.shape[0]
    if period + band > length:
        return 1e9
    a = field[: length - period, :]
    b = field[period:, :]
    return float(np.mean(np.abs(a - b)))


def edge_mismatch_score(arr: np.ndarray, band: int = 3) -> float:
    """對邊色差（只算邊緣帶，快）。越低越適合四方連續。"""
    b = max(1, min(int(band), arr.shape[0] // 4, arr.shape[1] // 4))
    left = arr[:, :b].astype(np.float64)
    right = arr[:, -b:].astype(np.float64)
    top = arr[:b].astype(np.float64)
    bottom = arr[-b:].astype(np.float64)
    return float(np.mean(np.abs(left - right))) + float(np.mean(np.abs(top - bottom)))


def structural_edge_score(arr: np.ndarray, band: int = 4) -> float:
    """
    對邊亮度 + 簡易梯度差。只取邊緣帶，避免全圖 float 轉換拖慢搜尋。
    """
    h, w = arr.shape[:2]
    b = max(1, min(int(band), h // 4, w // 4))

    def lum(strip: np.ndarray) -> np.ndarray:
        a = strip.astype(np.float32, copy=False)
        return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]

    left, right = lum(arr[:, :b]), lum(arr[:, -b:])
    top, bottom = lum(arr[:b]), lum(arr[-b:])
    color = float(np.mean(np.abs(left - right))) + float(np.mean(np.abs(top - bottom)))

    # 沿邊緣方向的一維梯度（結構對齊）
    def row_grad(strip: np.ndarray) -> np.ndarray:
        return np.abs(np.diff(strip, axis=0, prepend=strip[:1]))

    def col_grad(strip: np.ndarray) -> np.ndarray:
        return np.abs(np.diff(strip, axis=1, prepend=strip[:, :1]))

    struct = (
        float(np.mean(np.abs(row_grad(left) - row_grad(right))))
        + float(np.mean(np.abs(col_grad(top) - col_grad(bottom))))
    )
    return color + 1.25 * struct


def best_phase_roll(arr: np.ndarray) -> tuple[np.ndarray, tuple[int, int], float]:
    """
    循環平移找最佳原點，使對邊最吻合（滿鋪圖常因裁切相位不對而接不上）。
    """
    h, w = arr.shape[:2]
    scale = max(h, w) / 256.0
    sm = np.asarray(
        Image.fromarray(arr).resize(
            (max(48, int(round(w / scale))), max(48, int(round(h / scale)))),
            Image.Resampling.BILINEAR,
        )
    )
    sh, sw = sm.shape[:2]
    best, bo = 1e9, (0, 0)
    step = max(1, min(sh, sw) // 40)
    for ox in range(0, sw, step):
        for oy in range(0, sh, step):
            sc = structural_edge_score(np.roll(np.roll(sm, -oy, 0), -ox, 1))
            if sc < best:
                best, bo = sc, (ox, oy)
    ox0, oy0 = bo
    for dx in range(-14, 15):
        for dy in range(-14, 15):
            ox, oy = (ox0 + dx) % sw, (oy0 + dy) % sh
            sc = structural_edge_score(np.roll(np.roll(sm, -oy, 0), -ox, 1))
            if sc < best:
                best, bo = sc, (ox, oy)

    ox = int(round(bo[0] * scale)) % w
    oy = int(round(bo[1] * scale)) % h
    best, bxy = 1e9, (ox, oy)
    for dx in range(-8, 9):
        for dy in range(-8, 9):
            x, y = (ox + dx) % w, (oy + dy) % h
            sc = structural_edge_score(np.roll(np.roll(arr, -y, 0), -x, 1))
            if sc < best:
                best, bxy = sc, (x, y)
    x, y = bxy
    return np.roll(np.roll(arr, -y, 0), -x, 1), (x, y), best


def _axis_period_candidates(
    gray: np.ndarray,
    axis: int,
    scale: float,
    min_side: int,
) -> list[int]:
    """axis=0 → 水平週期（沿 x）；axis=1 → 垂直週期（沿 y）。"""
    if axis == 0:
        mean_sig = gray.mean(0)
        grad_sig = np.abs(np.diff(gray, axis=1)).mean(0)
        grad_sig = np.concatenate([grad_sig, grad_sig[-1:]])
    else:
        mean_sig = gray.mean(1)
        grad_sig = np.abs(np.diff(gray, axis=0)).mean(1)
        grad_sig = np.concatenate([grad_sig, grad_sig[-1:]])

    n = len(mean_sig)
    min_p, max_p = max(10, n // 40), n // 2
    votes: dict[int, float] = {}
    for sig, wt in ((mean_sig, 1.0), (grad_sig, 2.2)):
        for p, score in _autocorr_best_periods(sig, min_p, max_p, top_k=6):
            key = p
            for existing in list(votes.keys()):
                if abs(existing - p) <= max(2, p // 35):
                    key = existing
                    break
            votes[key] = votes.get(key, 0.0) + wt * score

    scored: list[tuple[float, int]] = []
    for p, sc in votes.items():
        err = _repeat_error(gray, p, axis)
        scored.append((err / (1.0 + sc), p))
    scored.sort()

    out: list[int] = []
    for _, p in scored[:6]:
        pf = max(8, int(round(p * scale)))
        if pf < min_side * 0.10:
            continue
        for d in (-2, -1, 0, 1, 2):
            if pf + d >= 8:
                out.append(pf + d)
    # 去重保序
    seen: set[int] = set()
    uniq: list[int] = []
    for p in out:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def _best_phase_for_size(
    arr: np.ndarray,
    cw: int,
    ch: int,
    px: int,
    py: int,
    *,
    sm: np.ndarray | None = None,
    scale: float | None = None,
) -> tuple[np.ndarray, tuple[int, int], float]:
    h, w = arr.shape[:2]
    if sm is None or scale is None:
        scale = max(h, w) / 220.0
        sm = np.asarray(
            Image.fromarray(arr).resize(
                (max(40, int(round(w / scale))), max(40, int(round(h / scale)))),
                Image.Resampling.BILINEAR,
            )
        )
    spx = max(1, int(round(px / scale)))
    spy = max(1, int(round(py / scale)))
    sh, sw = sm.shape[:2]
    # 對應到目標比例的小圖裁切尺寸
    scw = min(sw, max(8, int(round(cw / scale))))
    sch = min(sh, max(8, int(round(ch / scale))))
    scw = min(scw, sw)
    sch = min(sch, sh)

    best, bo = 1e9, (0, 0)
    max_ox = max(1, min(spx, sw - scw + 1))
    max_oy = max(1, min(spy, sh - sch + 1))
    stepx, stepy = max(1, spx // 6), max(1, spy // 6)
    for ox in range(0, max_ox, stepx):
        for oy in range(0, max_oy, stepy):
            sc = structural_edge_score(sm[oy : oy + sch, ox : ox + scw])
            if sc < best:
                best, bo = sc, (ox, oy)

    ox = min(max(0, int(round(bo[0] * scale))), w - cw)
    oy = min(max(0, int(round(bo[1] * scale))), h - ch)

    def _key(tile: np.ndarray) -> tuple[float, float, float]:
        sc = structural_edge_score(tile)
        sv, shs = _tile_seam_scores(tile)
        seam = sv + shs
        return (sc + seam * 0.35, seam, sc)

    best_key = _key(arr[oy : oy + ch, ox : ox + cw])
    bxy = (ox, oy)
    # 縮圖相位附近精修。超大裁切縮小精修窗，避免上萬次邊緣掃描。
    huge = max(cw, ch) >= 4000 or (cw * ch) >= 12_000_000
    refine_cap = 4 if huge else 12
    refine = max(3 if huge else 5, min(refine_cap, int(round(3.5 * scale))))
    x0, y0 = bxy
    for dx in range(-refine, refine + 1):
        for dy in range(-refine, refine + 1):
            x = min(max(0, x0 + dx), w - cw)
            y = min(max(0, y0 + dy), h - ch)
            k = _key(arr[y : y + ch, x : x + cw])
            if k < best_key:
                best_key, bxy = k, (x, y)
    if (not huge) and (
        best_key[1] > 40.0 or (max(px, py) >= 100 and best_key[1] > 28.0)
    ):
        # 全週期粗搜（棋盤／大週期常需大相位）；超大裁切改信縮圖相位
        stepx = max(2, px // 8 if px > 80 else px // 10)
        stepy = max(2, py // 8 if py > 80 else py // 10)
        if px <= 120 and py <= 120:
            stepx = max(2, min(stepx, 4))
            stepy = max(2, min(stepy, 4))
        for x in range(0, min(px, w - cw + 1), stepx):
            for y in range(0, min(py, h - ch + 1), stepy):
                sc = structural_edge_score(arr[y : y + ch, x : x + cw])
                if sc < best_key[0]:
                    best_key, bxy = (sc, 0.0, sc), (x, y)
        x0, y0 = bxy
        best_key = _key(arr[y0 : y0 + ch, x0 : x0 + cw])
        for dx in range(-5, 6):
            for dy in range(-5, 6):
                x = min(max(0, x0 + dx), w - cw)
                y = min(max(0, y0 + dy), h - ch)
                k = _key(arr[y : y + ch, x : x + cw])
                if k < best_key:
                    best_key, bxy = k, (x, y)
    # 細週期接縫抽樣（結構分誤導棋盤時）；超大裁切跳過
    if (not huge) and px <= 120 and py <= 120 and best_key[1] > 32.0:
        seam_best_s = float(best_key[1])
        sxy = bxy
        sx = max(2, px // 18)
        sy = max(2, py // 18)
        for x in range(0, min(px, w - cw + 1), sx):
            for y in range(0, min(py, h - ch + 1), sy):
                tile = arr[y : y + ch, x : x + cw]
                sv, shs = _tile_seam_scores(tile)
                seam = sv + shs
                if seam < seam_best_s - 0.5:
                    seam_best_s = seam
                    sxy = (x, y)
        if seam_best_s < best_key[1] - 4.0:
            best_key = _key(arr[sxy[1] : sxy[1] + ch, sxy[0] : sxy[0] + cw])
            bxy = sxy
            x0, y0 = bxy
            for dx in range(-4, 5):
                for dy in range(-4, 5):
                    x = min(max(0, x0 + dx), w - cw)
                    y = min(max(0, y0 + dy), h - ch)
                    k = _key(arr[y : y + ch, x : x + cw])
                    if k[1] < best_key[1] - 0.3 or (
                        k[1] <= best_key[1] + 1.0 and k < best_key
                    ):
                        best_key, bxy = k, (x, y)
    x, y = bxy
    return arr[y : y + ch, x : x + cw].copy(), (x, y), best_key[2]


def _grid_period_bonus(px: int, py: int, w: int, h: int) -> float:
    """負值=更優先。同 n 方格週期優於 x/y 各配不同 n 的半週期混搭。"""
    matched: list[tuple[int, int, int]] = []
    for n in (4, 5, 6, 7, 8):
        gx, gy = w // n, h // n
        if gx < 80 or gy < 80:
            continue
        dx, dy = abs(px - gx), abs(py - gy)
        if dx <= 3 and dy <= 3:
            matched.append((n, dx, dy))
    if not matched:
        return 0.0
    # 同 n 同時命中 x/y → 最佳（例：313×313 = 4 列格）
    for n, dx, dy in matched:
        gx, gy = w // n, h // n
        if abs(px - gx) <= 3 and abs(py - gy) <= 3:
            return -25.0 - float(dx + dy)
    # 只命中單軸或 x/y 來自不同 n → 半週期混搭，略懲罰
    return 8.0


def _axis_join_run_penalty(arr: np.ndarray, axis: int) -> float:
    """
    平鋪接縫處若兩側同色 run 合併成 ≈2× 內部典型寬度，視為半週期／錯相位。
    axis=0 查左右縫；axis=1 查上下縫。回傳懲罰（0=正常，越大越差）。
    """
    # 只取中線薄帶再轉 float，避免大裁切全圖 float64 卡死
    if axis == 0:
        y0 = arr.shape[0] // 2
        strip = arr[max(0, y0 - 4) : min(arr.shape[0], y0 + 5)]
        if arr.ndim == 3:
            band = strip.astype(np.float32).mean(axis=2).mean(axis=0)
        else:
            band = strip.astype(np.float32).mean(axis=0)
    else:
        x0 = arr.shape[1] // 2
        strip = arr[:, max(0, x0 - 4) : min(arr.shape[1], x0 + 5)]
        if arr.ndim == 3:
            band = strip.astype(np.float32).mean(axis=2).mean(axis=1)
        else:
            band = strip.astype(np.float32).mean(axis=1)
    if band.size < 24:
        return 0.0
    thr = 0.5 * (float(band.min()) + float(band.max()))
    # 對比過低：非條紋／色塊，不懲罰
    if float(band.max() - band.min()) < 12.0:
        return 0.0
    dark = band < thr
    runs: list[tuple[bool, int]] = []
    i = 0
    n = int(dark.size)
    while i < n:
        j = i + 1
        while j < n and bool(dark[j]) == bool(dark[i]):
            j += 1
        runs.append((bool(dark[i]), j - i))
        i = j
    if len(runs) < 4:
        return 0.0
    # 兩端同型才會在平鋪時合併
    if runs[0][0] != runs[-1][0]:
        return 0.0
    joined = runs[0][1] + runs[-1][1]
    interior = [w for t, w in runs[1:-1] if t == runs[0][0] and w >= 4]
    if not interior:
        return 0.0
    med = float(np.median(np.asarray(interior, dtype=np.float64)))
    if med < 4.0:
        return 0.0
    ratio = joined / med
    # 正常相位合併應接近 1×；1.65～2.4× 是典型雙倍條紋
    if 1.65 <= ratio <= 2.45:
        return 40.0 + 25.0 * abs(ratio - 2.0)
    if ratio > 2.45:
        return 20.0 + 8.0 * min(ratio, 4.0)
    return 0.0


def _tile_join_run_penalty(arr: np.ndarray) -> float:
    """左右＋上下接縫 run 寬度懲罰。"""
    return _axis_join_run_penalty(arr, 0) + _axis_join_run_penalty(arr, 1)


def _looks_like_icon_checkerboard(arr: np.ndarray) -> bool:
    """不依賴週期字串：邊緣亮度多峰 + 中等能量 → 咖啡格等棋盤圖示。"""
    energy = _edge_motif_energy(arr)
    if energy < 2.0 or energy > 8.0:
        return False
    band = max(8, int(min(arr.shape[0], arr.shape[1]) * 0.05))
    # 只取邊緣帶，避免超大圖整幅轉 float
    parts = (
        arr[:, :band],
        arr[:, -band:],
        arr[:band, :],
        arr[-band:, :],
    )
    edge = np.concatenate(
        [
            (
                p.astype(np.float32).mean(axis=2).ravel()
                if p.ndim == 3
                else p.astype(np.float32).ravel()
            )
            for p in parts
        ]
    )
    hist, _ = np.histogram(edge, bins=8, range=(0, 255))
    peaks = int(np.sum(hist > hist.max() * 0.35))
    return peaks >= 3


def _is_icon_grid_tile(arr: np.ndarray, detail: str, w: int, h: int) -> bool:
    """
    圖示／棋盤格（咖啡格等）：必須同時滿足
    1) 週期落在同 n 的方格 w//n×h//n
    2) 邊緣亮度直方圖多峰（棋盤格特徵）
    """
    if not _looks_like_icon_checkerboard(arr):
        return False
    m = re.search(r"週期\s*(\d+)\s*[×x]\s*(\d+)", detail)
    if not m:
        return True  # 影像像棋盤格，即使尚未寫出週期
    px, py = int(m.group(1)), int(m.group(2))
    return _grid_period_bonus(px, py, w, h) <= -20.0


def try_period_crop(
    arr: np.ndarray,
    bg: Sequence[int] | None = None,
    *,
    log: Callable[[str], None] | None = None,
) -> tuple[np.ndarray | None, str]:
    """
    滿鋪幾何／斜紋／魚鱗：用亮度+梯度找獨立 xy 週期，再裁成整數倍並搜相位。
    以 structural_edge_score + 接縫色差驗證，沒改善則失敗。
    """
    del bg  # 不再依賴背景色二值化（滿鋪時易誤判）
    del log  # 介面相容；搜尋不因逾時提早結束，避免接縫變差

    h, w = arr.shape[:2]
    base = structural_edge_score(arr)
    base_v, base_h = _tile_seam_scores(arr)
    base_seam = base_v + base_h
    scale = max(h, w) / 400.0
    small = np.asarray(
        Image.fromarray(arr).resize(
            (max(64, int(round(w / scale))), max(64, int(round(h / scale)))),
            Image.Resampling.BILINEAR,
        )
    )
    gray = _luminance_map(small)
    # 深色輪廓（魚鱗描邊等）對週期很敏感
    dark = (gray < float(np.percentile(gray, 18))).astype(np.float64)
    # 也投斜向，補強斜條紋
    votes_extra_x: list[int] = []
    votes_extra_y: list[int] = []
    for sign in (1, -1):
        for field in (gray, dark):
            proj = _diagonal_projection(field, sign)
            for p, _ in _autocorr_best_periods(
                proj, max(10, len(proj) // 40), len(proj) // 2, top_k=4
            ):
                pf = max(8, int(round(p * scale)))
                if pf >= min(h, w) * 0.10:
                    votes_extra_x.append(pf)
                    votes_extra_y.append(pf)

    xs = (
        _axis_period_candidates(gray, 0, scale, min(h, w))
        + _axis_period_candidates(dark, 0, scale, min(h, w))
        + votes_extra_x
    )
    ys = (
        _axis_period_candidates(gray, 1, scale, min(h, w))
        + _axis_period_candidates(dark, 1, scale, min(h, w))
        + votes_extra_y
    )
    # 棋盤等：半週期常被偵測，補上 2× 色週期；並在強峰附近微調
    xs = list(xs) + [2 * p for p in xs if 16 <= 2 * p <= min(h, w) // 2]
    ys = list(ys) + [2 * p for p in ys if 16 <= 2 * p <= min(h, w) // 2]
    refined: list[int] = []
    for p in sorted(set(xs))[:6]:
        for d in range(-10, 11, 2):
            if p + d >= 16:
                refined.append(p + d)
    xs = sorted(set(refined))[:16]
    refined = []
    for p in sorted(set(ys))[:6]:
        for d in range(-10, 11, 2):
            if p + d >= 16:
                refined.append(p + d)
    ys = sorted(set(refined))[:16]
    if not xs or not ys:
        return None, "未偵測到週期"

    # 全解析度自相關強峰（縮圖常漏掉接近半幅的真週期，如 250 on 627）
    # 下限勿過高：條紋基本週期常 < 12% 寬（例 88 on 1231）
    # 超大圖改在中等縮圖上找峰，再映射回原圖尺度（避免整圖 luminance）
    full_x: list[int] = []
    full_y: list[int] = []
    if max(h, w) > 2800:
        mid_scale = max(h, w) / 1600.0
        mid = np.asarray(
            Image.fromarray(arr).resize(
                (
                    max(64, int(round(w / mid_scale))),
                    max(64, int(round(h / mid_scale))),
                ),
                Image.Resampling.BILINEAR,
            )
        )
        mid_gray = _luminance_map(mid)
        mh, mw = mid_gray.shape
        for p, score in _autocorr_best_periods(
            mid_gray.mean(1), max(8, mh // 40), mh // 2, top_k=5
        ):
            if score >= 0.25 and mh * 0.05 <= p <= mh * 0.48:
                pf = max(8, int(round(p * mid_scale)))
                full_y.extend([pf, int(round(pf / 2)), int(round(pf / 4))])
        for p, score in _autocorr_best_periods(
            mid_gray.mean(0), max(8, mw // 40), mw // 2, top_k=5
        ):
            if score >= 0.25 and mw * 0.05 <= p <= mw * 0.48:
                pf = max(8, int(round(p * mid_scale)))
                full_x.extend([pf, int(round(pf / 2)), int(round(pf / 4))])
    else:
        full_gray = _luminance_map(arr)
        for p, score in _autocorr_best_periods(
            full_gray.mean(1), max(16, h // 40), h // 2, top_k=5
        ):
            if score >= 0.25 and h * 0.05 <= p <= h * 0.48:
                full_y.extend([int(p), int(round(p / 2)), int(round(p / 4))])
        for p, score in _autocorr_best_periods(
            full_gray.mean(0), max(16, w // 40), w // 2, top_k=5
        ):
            if score >= 0.25 and w * 0.05 <= p <= w * 0.48:
                full_x.extend([int(p), int(round(p / 2)), int(round(p / 4))])
    # 強峰置頂，再接縮圖候選
    xs = list(dict.fromkeys([*full_x, *xs]))
    ys = list(dict.fromkeys([*full_y, *ys]))
    # 圖標／棋盤格：優先試整除格寬（咖啡杯格常是 4–8 列）
    grid_px: list[int] = []
    grid_py: list[int] = []
    for n in (4, 5, 6, 7, 8):
        gx, gy = w // n, h // n
        if 80 <= gx <= w // 2:
            grid_px.append(gx)
        if 80 <= gy <= h // 2:
            grid_py.append(gy)
    xs = list(dict.fromkeys([*grid_px, *xs]))
    ys = list(dict.fromkeys([*grid_py, *ys]))
    # 僅在真像棋盤／高邊緣圖示能量時走圖示格捷徑。
    # 不可用 len(grid_px)>=2：幾乎所有大圖都有 w//4..w//8，會把直條紋誤判成格點。
    icon_grid_likely = _looks_like_icon_checkerboard(arr) or _edge_motif_energy(arr) >= 3.0
    if icon_grid_likely:
        xs = list(dict.fromkeys([*grid_px, *xs[:8]]))[:12]
        ys = list(dict.fromkeys([*grid_py, *ys[:8]]))[:12]
    else:
        xs = [p for p in xs if 16 <= p <= w // 2][:20]
        ys = [p for p in ys if 16 <= p <= h // 2][:20]

    best_tile: np.ndarray | None = None
    best_rank = (base + base_seam * 0.35, base_seam, base)
    best_detail = ""

    # 相位搜尋共用一份縮圖，避免每個週期候選都重 resize
    phase_scale = max(h, w) / 220.0
    phase_sm = np.asarray(
        Image.fromarray(arr).resize(
            (
                max(40, int(round(w / phase_scale))),
                max(40, int(round(h / phase_scale))),
            ),
            Image.Resampling.BILINEAR,
        )
    )

    period_pairs: list[tuple[int, int]] = []
    if icon_grid_likely:
        for n in (4, 5, 6, 7, 8):
            gx, gy = w // n, h // n
            if 80 <= gx <= w // 2 and 80 <= gy <= h // 2 and abs(gx - gy) <= 4:
                period_pairs.append((gx, gy))
        period_pairs = list(dict.fromkeys(period_pairs))
    # 非圖示格收斂週期對數；圖示格保留較寬搜尋
    x_cap = 12 if icon_grid_likely else 8
    y_cap = 12 if icon_grid_likely else 8
    for px in xs[:x_cap]:
        for py in ys[:y_cap]:
            period_pairs.append((px, py))
    period_pairs = list(dict.fromkeys(period_pairs))
    if icon_grid_likely and period_pairs:
        # 圖示格：先只試方格週期，命中後可提早結束
        grid_only = [p for p in period_pairs if _grid_period_bonus(p[0], p[1], w, h) <= -20.0]
        if grid_only:
            period_pairs = grid_only + [p for p in period_pairs if p not in grid_only]

    # 縮圖粗排：只對結構分最好的 Top-N 做全解析度相位搜尋
    sh_sm, sw_sm = phase_sm.shape[:2]

    def _thumb_struct(px: int, py: int, cw: int, ch: int) -> float:
        spx = max(1, int(round(px / phase_scale)))
        spy = max(1, int(round(py / phase_scale)))
        scw = min(sw_sm, max(8, int(round(cw / phase_scale))))
        sch = min(sh_sm, max(8, int(round(ch / phase_scale))))
        max_ox = max(1, min(spx, sw_sm - scw + 1))
        max_oy = max(1, min(spy, sh_sm - sch + 1))
        stepx, stepy = max(1, spx // 5), max(1, spy // 5)
        best = 1e9
        for ox in range(0, max_ox, stepx):
            for oy in range(0, max_oy, stepy):
                sc = structural_edge_score(phase_sm[oy : oy + sch, ox : ox + scw])
                if sc < best:
                    best = sc
        return best

    jobs: list[tuple[float, int, int, int, int]] = []
    for px, py in period_pairs:
        nmax = w // px
        mmax = h // py
        for dn in (0, 1):
            for dm in (0, 1):
                cw = (nmax - dn) * px
                ch = (mmax - dm) * py
                if cw < int(w * 0.72) or ch < int(h * 0.72):
                    continue
                if cw < max(320, w // 2) or ch < max(320, h // 2):
                    continue
                jobs.append((_thumb_struct(px, py, cw, ch), px, py, cw, ch))
    jobs.sort(key=lambda t: t[0])
    top_n = 28 if icon_grid_likely else 18
    if max(h, w) > 4000:
        top_n = 10 if icon_grid_likely else 8
    # 方格／近正方形週期保底進候選（縮圖分不一定最好，但常是正解）
    guaranteed: set[tuple[int, int, int, int]] = set()
    for _, px, py, cw, ch in jobs:
        if abs(px - py) <= max(3, min(px, py) // 10):
            guaranteed.add((px, py, cw, ch))
        if icon_grid_likely and _grid_period_bonus(px, py, w, h) <= -20.0:
            guaranteed.add((px, py, cw, ch))
    # 限制保底數量，避免又掃回上百組
    g_cap = 6 if max(h, w) > 4000 else 12
    if len(guaranteed) > g_cap:
        # 優先較小週期（細密紋）與較大覆蓋
        guaranteed = set(
            sorted(
                guaranteed,
                key=lambda t: (abs(t[0] - t[1]), t[0] + t[1], -(t[2] * t[3])),
            )[:g_cap]
        )
    selected: list[tuple[int, int, int, int]] = []
    seen_job: set[tuple[int, int, int, int]] = set()
    for _, px, py, cw, ch in jobs:
        key = (px, py, cw, ch)
        if key in seen_job:
            continue
        seen_job.add(key)
        selected.append(key)
        if len(selected) >= top_n:
            break
    for key in guaranteed:
        if key not in seen_job:
            selected.append(key)
            seen_job.add(key)

    for px, py, cw, ch in selected:
                tile, off, sc = _best_phase_for_size(
                    arr, cw, ch, px, py, sm=phase_sm, scale=phase_scale
                )
                sv, shs = _tile_seam_scores(tile)
                seam = sv + shs
                join_pen = _tile_join_run_penalty(tile)
                # 雙倍條紋等錯相位：直接淘汰，避免低色差假勝利
                if join_pen >= 40.0:
                    continue
                bonus = _grid_period_bonus(px, py, w, h) if icon_grid_likely else 0.0
                # 近正方形週期略加分（細格紋常被拆成 105×209 半週期混搭）
                square_bonus = 0.0
                if abs(px - py) <= max(3, min(px, py) // 10):
                    square_bonus = -8.0
                phase_pen = 0.0
                if bonus <= -20.0:
                    # 棋盤格：相位應靠近格線，禁止切在格子中間
                    rx, ry = off[0] % max(px, 1), off[1] % max(py, 1)
                    rx = min(rx, px - rx)
                    ry = min(ry, py - ry)
                    phase_pen = float(rx + ry) * 0.55
                # 原圖接縫很高時以色差為主（結構分常偏好錯相位大裁切）
                seam_w = 0.85 if base_seam > 80.0 else 0.35
                rank = (
                    sc * (0.15 if base_seam > 80.0 else 1.0)
                    + seam * seam_w
                    + bonus
                    + square_bonus
                    + phase_pen
                    + join_pen,
                    seam,
                    sc,
                )
                # 候選可暫收下中等色差；最終是否採用由 try_make_dense + 色差均衡決定
                if seam > 95.0 or max(sv, shs) > 70.0:
                    continue
                grid_ok = bonus <= -20.0 and phase_pen <= 12.0
                if rank < best_rank and (
                    seam <= base_seam
                ) and (
                    sc + 0.15 < base
                    or seam < base_seam * 0.85
                    or seam < base_seam - 10
                    or (grid_ok and seam <= base_seam * 0.98)
                ):
                    best_rank = rank
                    best_tile = tile
                    best_detail = (
                        f"週期 {px}×{py}px → 單元 {cw}×{ch}"
                        f"（偏移 {off[0]},{off[1]}，接縫分 {sc:.2f}←{base:.2f}）"
                    )

    if best_tile is None:
        return None, f"無穩定週期（接縫分 {base:.2f}，裁切無改善）"
    return best_tile, best_detail


def _min_cut_path(err: np.ndarray) -> np.ndarray:
    """
    err: (length, band) 非負代價。回傳每列的切割欄位（沿 length 方向前進）。
    """
    length, band = err.shape
    cum = err.astype(np.float64).copy()
    back = np.zeros((length, band), dtype=np.int16)
    for i in range(1, length):
        for j in range(band):
            opts = []
            for dj in (-1, 0, 1):
                pj = j + dj
                if 0 <= pj < band:
                    opts.append((cum[i - 1, pj], pj))
            val, pj = min(opts)
            cum[i, j] = err[i, j] + val
            back[i, j] = pj
    cut = np.zeros(length, dtype=np.int32)
    j = int(np.argmin(cum[-1]))
    for i in range(length - 1, -1, -1):
        cut[i] = j
        if i > 0:
            j = int(back[i, j])
    return cut


def detect_edge_shifts(
    arr: np.ndarray, band: int = 12, lim: int = 220
) -> tuple[int, int, float, float]:
    """
    搜尋左右拼合最佳垂直錯位 dy、上下拼合最佳水平錯位 dx。
    回傳 (dy, dx, score_v, score_h)。
    """
    h, w = arr.shape[:2]
    band = max(4, min(band, h // 8, w // 8))
    lim = min(lim, h // 3, w // 3)
    # 只轉邊緣帶，禁止整圖 float64（12k 圖會吃掉數 GB）
    left = arr[:, :band].astype(np.float32)
    right = arr[:, -band:].astype(np.float32)
    best_v = (1e9, 0)
    for d in range(-lim, lim + 1):
        sc = float(np.mean(np.abs(left - np.roll(right, d, axis=0))))
        if sc < best_v[0]:
            best_v = (sc, d)

    top = arr[:band].astype(np.float32)
    bottom = arr[-band:].astype(np.float32)
    best_h = (1e9, 0)
    for d in range(-lim, lim + 1):
        sc = float(np.mean(np.abs(top - np.roll(bottom, d, axis=1))))
        if sc < best_h[0]:
            best_h = (sc, d)

    return best_v[1], best_h[1], best_v[0], best_h[0]


def _wrapped_shear_v(arr: np.ndarray, dy: int) -> np.ndarray:
    """環繞錯切：y 隨 x 平移，把左右拼合所需的垂直錯位攤平。"""
    if dy == 0:
        return arr
    h, w = arr.shape[:2]
    yy, xx = np.indices((h, w))
    src_y = (yy - dy * xx / max(w - 1, 1)) % h
    y0 = np.floor(src_y).astype(np.int32) % h
    y1 = (y0 + 1) % h
    t = (src_y - np.floor(src_y))[..., None]
    af = arr.astype(np.float64)
    out = (1.0 - t) * af[y0, xx] + t * af[y1, xx]
    return np.clip(out, 0, 255).astype(np.uint8)


def _wrapped_shear_h(arr: np.ndarray, dx: int) -> np.ndarray:
    """環繞錯切：x 隨 y 平移，把上下拼合所需的水平錯位攤平。"""
    if dx == 0:
        return arr
    h, w = arr.shape[:2]
    yy, xx = np.indices((h, w))
    src_x = (xx - dx * yy / max(h - 1, 1)) % w
    x0 = np.floor(src_x).astype(np.int32) % w
    x1 = (x0 + 1) % w
    t = (src_x - np.floor(src_x))[..., None]
    af = arr.astype(np.float64)
    out = (1.0 - t) * af[yy, x0] + t * af[yy, x1]
    return np.clip(out, 0, 255).astype(np.uint8)


def _tile_seam_scores(arr: np.ndarray) -> tuple[float, float]:
    """2×2 中心垂直接縫、水平接縫的平均色差。"""
    h, w = arr.shape[:2]
    # 模擬 tile 接縫：右邊緣 vs 左邊緣、下邊緣 vs 上邊緣
    v = float(np.mean(np.abs(arr[:, -1].astype(np.float64) - arr[:, 0].astype(np.float64))))
    hh = float(np.mean(np.abs(arr[-1].astype(np.float64) - arr[0].astype(np.float64))))
    return v, hh


def flatten_illumination(arr: np.ndarray, radius: int | None = None) -> np.ndarray:
    """去掉大範圍光照／拍攝色溫梯度，保留紋理細節。"""
    h, w = arr.shape[:2]
    if radius is None:
        # 超大圖 GaussianBlur 半徑會卡死；光照只需粗尺度
        radius = max(24, min(96, min(h, w) // 6))
    blur = np.asarray(
        Image.fromarray(arr).filter(ImageFilter.GaussianBlur(radius=radius)),
        dtype=np.float64,
    )
    a = arr.astype(np.float64)
    mean = a.mean(axis=(0, 1), keepdims=True)
    return np.clip(a - blur + mean, 0, 255).astype(np.uint8)


def soft_edge_midpoint(
    arr: np.ndarray,
    *,
    band_frac: float = 0.05,
    strength: float = 0.45,
    structure_safe: bool = False,
) -> np.ndarray:
    """
    對邊邊緣帶向中點靠攏，消除可視色帶／十字縫色差。
    strength 過高會糊邊；生產預設約 0.45。
    structure_safe：僅在對邊都較平坦時混合（避免圖示重影）。
    """
    h, w = arr.shape[:2]
    band = max(8, int(min(h, w) * band_frac))
    out = arr.astype(np.float64).copy()
    if structure_safe:
        g = out.mean(axis=2)
        # 邊緣結構強度：局部梯度
        gx = np.abs(np.diff(g, axis=1, prepend=g[:, :1]))
        gy = np.abs(np.diff(g, axis=0, prepend=g[:1, :]))
        struct = gx + gy
    for i in range(band):
        t = 1.0 - i / float(band)
        alpha = strength * t * t
        mid = 0.5 * (out[:, i] + out[:, w - 1 - i])
        if structure_safe:
            # 兩側都平坦才允許混合
            s = np.maximum(struct[:, i], struct[:, w - 1 - i])
            a = alpha * (s < 18.0).astype(np.float64)[:, None]
            out[:, i] = (1.0 - a) * out[:, i] + a * mid
            out[:, w - 1 - i] = (1.0 - a) * out[:, w - 1 - i] + a * mid
        else:
            out[:, i] = (1.0 - alpha) * out[:, i] + alpha * mid
            out[:, w - 1 - i] = (1.0 - alpha) * out[:, w - 1 - i] + alpha * mid
    for i in range(band):
        t = 1.0 - i / float(band)
        alpha = strength * t * t
        mid = 0.5 * (out[i] + out[h - 1 - i])
        if structure_safe:
            s = np.maximum(struct[i], struct[h - 1 - i])
            a = alpha * (s < 18.0).astype(np.float64)[:, None]
            out[i] = (1.0 - a) * out[i] + a * mid
            out[h - 1 - i] = (1.0 - a) * out[h - 1 - i] + a * mid
        else:
            out[i] = (1.0 - alpha) * out[i] + alpha * mid
            out[h - 1 - i] = (1.0 - alpha) * out[h - 1 - i] + alpha * mid
    return np.clip(out, 0, 255).astype(np.uint8)


def _edge_motif_energy(arr: np.ndarray, band_frac: float = 0.05) -> float:
    """邊緣帶高頻能量：圖示／線條多時 soft 對齊易出重影。"""
    h, w = arr.shape[:2]
    band = max(8, int(min(h, w) * band_frac))

    def _lum_band(strip: np.ndarray) -> np.ndarray:
        a = strip.astype(np.float32, copy=False)
        if a.ndim == 2:
            return a
        return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]

    parts = (
        np.abs(np.diff(_lum_band(arr[:, :band]), axis=0)),
        np.abs(np.diff(_lum_band(arr[:, -band:]), axis=0)),
        np.abs(np.diff(_lum_band(arr[:band, :]), axis=1)),
        np.abs(np.diff(_lum_band(arr[-band:, :]), axis=1)),
    )
    return float(np.percentile(np.concatenate([p.ravel() for p in parts]), 90))


def _soft_blend_ghosts(
    orig: np.ndarray,
    soft: np.ndarray,
    *,
    band_frac: float = 0.05,
) -> bool:
    """
    對邊 soft 混合是否把對側圖示「印」進本側邊緣（咖啡／圖標格常見重影）。
    """
    h, w = orig.shape[:2]
    band = max(8, int(min(h, w) * band_frac))
    o = orig.astype(np.float64)
    s = soft.astype(np.float64)

    def _side_ghost(a0: np.ndarray, a1: np.ndarray, b0: np.ndarray) -> float:
        toward = a1 - a0
        delta = b0 - a0
        mag = np.linalg.norm(toward, axis=2)
        mask = mag > 14.0
        if not np.any(mask):
            return 0.0
        cos = np.sum(delta * toward, axis=2) / (
            np.linalg.norm(delta, axis=2) * mag + 1e-6
        )
        return float(np.mean(np.clip(cos[mask], 0.0, 1.0)) * np.mean(np.abs(delta)[mask]))

    g = max(
        _side_ghost(o[:, :band], o[:, -band:], s[:, :band]),
        _side_ghost(o[:, -band:], o[:, :band], s[:, -band:]),
        _side_ghost(o[:band], o[-band:], s[:band]),
        _side_ghost(o[-band:], o[:band], s[-band:]),
    )
    # 邊緣本就有圖示結構時，門檻更嚴
    energy = _edge_motif_energy(orig, band_frac=band_frac)
    return g > (2.0 if energy >= 5.0 else 6.0)


def equalize_seam_colors(arr: np.ndarray) -> np.ndarray:
    """色差修復：去光照梯度 + 對邊柔和對齊。"""
    return soft_edge_midpoint(flatten_illumination(arr))


def equalize_edge_means(
    arr: np.ndarray,
    *,
    band_frac: float = 0.04,
) -> np.ndarray:
    """
    只對齊對邊邊緣帶的顏色均值／方差（不做像素混合），避免圖示重影。
    """
    h, w = arr.shape[:2]
    band = max(6, int(min(h, w) * band_frac))
    out = arr.astype(np.float64).copy()

    def _match(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ma = a.mean(axis=(0, 1), keepdims=True)
        mb = b.mean(axis=(0, 1), keepdims=True)
        mid = 0.5 * (ma + mb)
        sa = a.std(axis=(0, 1), keepdims=True) + 1e-6
        sb = b.std(axis=(0, 1), keepdims=True) + 1e-6
        smid = 0.5 * (sa + sb)
        a2 = (a - ma) * (smid / sa) + mid
        b2 = (b - mb) * (smid / sb) + mid
        return a2, b2

    # 左右
    for i in range(band):
        t = 1.0 - i / float(band)
        t = t * t
        L = out[:, i : i + 1]
        R = out[:, w - 1 - i : w - i]
        L2, R2 = _match(L, R)
        out[:, i : i + 1] = (1.0 - t) * L + t * L2
        out[:, w - 1 - i : w - i] = (1.0 - t) * R + t * R2
    # 上下
    for i in range(band):
        t = 1.0 - i / float(band)
        t = t * t
        T = out[i : i + 1]
        B = out[h - 1 - i : h - i]
        T2, B2 = _match(T, B)
        out[i : i + 1] = (1.0 - t) * T + t * T2
        out[h - 1 - i : h - i] = (1.0 - t) * B + t * B2
    return np.clip(out, 0, 255).astype(np.uint8)


def maybe_equalize_seam_colors(
    arr: np.ndarray,
    *,
    min_improve: float = 2.0,
    ratio: float = 0.92,
) -> tuple[np.ndarray, str]:
    """
    若色差均衡能實質降低對邊色差則採用，否則原樣返回。
    高接縫時會加大對齊強度並可多輪，直到落到可視可接受範圍或無再改善。
    邊緣有圖示結構時禁止強 soft（避免接縫重影）。
    """
    v0, h0 = _tile_seam_scores(arr)
    s0 = v0 + h0
    if s0 < 8.0:
        return arr, ""

    motif_edge = _edge_motif_energy(arr) >= 3.0
    best_arr = arr
    best_s = s0
    label = "色差對齊"

    # 圖示邊緣：優先均值對齊（無重影），再試極弱 soft
    mean_eq = equalize_edge_means(arr)
    mild = soft_edge_midpoint(arr, strength=0.18, band_frac=0.02)
    first_pass: list[tuple[str, np.ndarray]] = [
        ("邊緣均值對齊", mean_eq),
        ("色差弱對齊", mild),
    ]
    if not motif_edge:
        first_pass.extend(
            (
                ("色差均衡", equalize_seam_colors(arr)),
                ("色差對齊", soft_edge_midpoint(arr, strength=0.5)),
            )
        )

    for name, cand in first_pass:
        if _soft_blend_ghosts(arr, cand):
            continue
        v, h = _tile_seam_scores(cand)
        s = v + h
        if s <= s0 * ratio and s <= s0 - min_improve and s < best_s:
            best_arr, best_s, label = cand, s, name

    # 圖示／棋盤邊緣：窄帶中等 soft 常無重影，可把 40+ 殘縫壓到審計線下
    if best_s > 22.0 and motif_edge:
        cur = best_arr
        for strength, band in (
            (0.28, 0.02),
            (0.35, 0.025),
            (0.42, 0.03),
            (0.48, 0.03),
        ):
            cand = soft_edge_midpoint(cur, band_frac=band, strength=strength)
            # 相對當前結果偵測重影（相對原圖過嚴會誤殺窄帶對齊）
            if _soft_blend_ghosts(cur, cand, band_frac=max(band, 0.03)):
                continue
            v, h = _tile_seam_scores(cand)
            s = v + h
            if s < best_s - 1.0:
                cur, best_s = cand, s
                best_arr = cur
                label = "色差窄帶對齊"
            if best_s <= 18.0 and max(v, h) <= 12.0:
                break
        # 貼審計 cross 門檻（約 28）時再試一檔
        if 22.0 < best_s <= 32.0:
            for strength, band in ((0.50, 0.03), (0.55, 0.035)):
                cand = soft_edge_midpoint(best_arr, band_frac=band, strength=strength)
                if _soft_blend_ghosts(best_arr, cand, band_frac=max(band, 0.03)):
                    continue
                v, h = _tile_seam_scores(cand)
                s = v + h
                if s < best_s - 0.8:
                    best_arr, best_s = cand, s
                    label = "色差窄帶對齊"
                if best_s <= 26.0:
                    break

    # 仍偏高：加強／加寬邊緣帶多輪對齊（專門壓可視色帶）；圖示格跳過
    if best_s > 22.0 and not motif_edge:
        cur = best_arr
        for strength, band in ((0.62, 0.06), (0.78, 0.08), (0.88, 0.10), (0.95, 0.12)):
            cand = soft_edge_midpoint(cur, band_frac=band, strength=strength)
            if _soft_blend_ghosts(arr, cand, band_frac=band):
                break
            v, h = _tile_seam_scores(cand)
            s = v + h
            if s < best_s - 0.8:
                cur, best_s = cand, s
                best_arr = cur
                label = "色差強化對齊"
            if best_s <= 18.0 and max(v, h) <= 14.0:
                break

    if best_s >= s0 - min_improve and best_s > s0 * ratio:
        return arr, ""

    v1, h1 = _tile_seam_scores(best_arr)
    return best_arr, f"{label} {v0:.0f}+{h0:.0f}→{v1:.0f}+{h1:.0f}"


def _edge_profile_corr(arr: np.ndarray) -> tuple[float, float]:
    """左右／上下邊緣亮度剖面相關（1=對齊好）。"""
    lum = arr.astype(np.float64).mean(axis=2)
    def _corr(a: np.ndarray, b: np.ndarray) -> float:
        if a.std() < 1e-6 or b.std() < 1e-6:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    return _corr(lum[:, 0], lum[:, -1]), _corr(lum[0], lum[-1])


def _seam_rank(arr: np.ndarray) -> float:
    """越小越好：色差為主，邊緣相關不足時加罰。"""
    v, h = _tile_seam_scores(arr)
    cv, ch = _edge_profile_corr(arr)
    penalty = (max(0.0, 0.85 - cv) + max(0.0, 0.85 - ch)) * 40.0
    return v + h + penalty


def _offset_mincut_axis(
    arr: np.ndarray,
    axis: int,
    ov: int,
    lim: int,
    *,
    prefer_flat: bool = False,
) -> tuple[np.ndarray, int]:
    """
    沿 axis 半幅偏移後，在中央重疊帶做 min-cut 硬切。
    不滾動半幅本體，以保留外緣=原圖中心相鄰像素（四方連續保證）。
    prefer_flat：代價加上局部梯度，迫使切割走平坦底色（避免切斷咖啡格圖示）。
    """
    del lim  # 外緣連續優先，不再用半幅錯位
    h, w = arr.shape[:2]

    def _err_map(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """a/b: (..., band, 3) → (length, band) 代價。"""
        af = a.astype(np.float64)
        bf = b.astype(np.float64)
        err = np.mean(np.abs(af - bf), axis=2)
        if prefer_flat:
            # 兩側平均圖的梯度高 → 代價高（寧可切在平坦格底）
            mid = 0.5 * (af + bf)
            lum = mid.mean(axis=2)
            if lum.ndim == 2 and lum.shape[1] >= 2:
                g = np.abs(np.diff(lum, axis=1, prepend=lum[:, :1]))
                g = g + np.abs(np.diff(lum, axis=0, prepend=lum[:1, :]))
            else:
                g = np.zeros_like(err)
            err = err + 1.35 * g
        return err

    if axis == 1:
        mid = w // 2
        out = np.roll(arr, mid, axis=1)
        left_ov = out[:, mid - ov : mid]
        right_ov = out[:, mid : mid + ov]
        err = _err_map(left_ov, right_ov)
        cut = _min_cut_path(err)
        for y in range(h):
            j = int(cut[y])
            for x in range(j, ov):
                out[y, mid - ov + x] = right_ov[y, x]
        return out, 0

    mid = h // 2
    out = np.roll(arr, mid, axis=0)
    top_ov = out[mid - ov : mid]
    bot_ov = out[mid : mid + ov]
    err = _err_map(top_ov, bot_ov).T
    cut = _min_cut_path(err)
    for x in range(w):
        j = int(cut[x])
        for y in range(j, ov):
            out[mid - ov + y, x] = bot_ov[y, x]
    return out, 0


def offset_quilt_seamless(arr: np.ndarray) -> tuple[np.ndarray, str]:
    """
    多圖拼接對齊（半幅偏移 + 硬切）：
    外緣取自原圖中心連續區，2×2 紅線處斜紋會接上；
    單元圖中央用 min-cut 消化原本的破邊。
    """
    h, w = arr.shape[:2]
    base_v, base_h = _tile_seam_scores(arr)
    base_cv, base_ch = _edge_profile_corr(arr)
    base_rank = _seam_rank(arr)
    iconish = _looks_like_icon_checkerboard(arr)
    ov = max(80, min(h, w) // 10)
    if iconish:
        # 圖示格：加寬重疊帶，讓切割有機會繞開杯子／豆子
        ov = max(ov, min(h, w) // 6)

    # 純半幅偏移：外緣=原圖中心相鄰像素，四方連續有數學保證
    pure = np.roll(np.roll(arr, w // 2, axis=1), h // 2, axis=0)

    # 超大圖：跳過 min-cut／大半徑去光照，只保留純滾（否則會卡數分鐘）
    if max(h, w) >= 4000:
        v, hs = _tile_seam_scores(pure)
        cv, ch = _edge_profile_corr(pure)
        if _seam_rank(pure) <= base_rank * 0.95 or (cv + ch) >= (base_cv + base_ch) + 0.2:
            return (
                pure,
                f"半幅偏移拼接（純滾）（接縫 {base_v:.0f}+{base_h:.0f}→{v:.0f}+{hs:.0f}，"
                f"邊緣相關 {base_cv:.2f}/{base_ch:.2f}→{cv:.2f}/{ch:.2f}）",
            )
        return (
            arr.copy(),
            f"拼接無改善（原接縫 {base_v:.0f}+{base_h:.0f}）",
        )

    # 圖示格：min-cut 即使「平切」仍常切斷格子；純偏移外緣已連續。
    # 先對原圖邊緣做均值對齊，再滾動，把殘餘色差留在單元中央且較淡。
    if iconish:
        pre = equalize_edge_means(flatten_illumination(arr), band_frac=0.06)
        pure_pre = np.roll(np.roll(pre, w // 2, axis=1), h // 2, axis=0)
        # 外緣改回「未 soft 的純滾」中心相鄰列，保住四方連續
        edge = max(2, min(8, ov // 20))
        pure_pre[:, :edge] = pure[:, :edge]
        pure_pre[:, -edge:] = pure[:, -edge:]
        pure_pre[:edge, :] = pure[:edge, :]
        pure_pre[-edge:, :] = pure[-edge:, :]
        v, hs = _tile_seam_scores(pure_pre)
        cv, ch = _edge_profile_corr(pure_pre)
        if _seam_rank(pure_pre) <= base_rank * 0.95 or (cv + ch) >= (base_cv + base_ch) + 0.2:
            return (
                pure_pre,
                f"半幅偏移拼接（純滾）（接縫 {base_v:.0f}+{base_h:.0f}→{v:.0f}+{hs:.0f}，"
                f"邊緣相關 {base_cv:.2f}/{base_ch:.2f}→{cv:.2f}/{ch:.2f}）",
            )
        v, hs = _tile_seam_scores(pure)
        cv, ch = _edge_profile_corr(pure)
        if _seam_rank(pure) <= base_rank * 0.95 or (cv + ch) >= (base_cv + base_ch) + 0.2:
            return (
                pure,
                f"半幅偏移拼接（純滾）（接縫 {base_v:.0f}+{base_h:.0f}→{v:.0f}+{hs:.0f}，"
                f"邊緣相關 {base_cv:.2f}/{base_ch:.2f}→{cv:.2f}/{ch:.2f}）",
            )
        return (
            arr.copy(),
            f"拼接無改善（原接縫 {base_v:.0f}+{base_h:.0f}）",
        )

    cur, _ = _offset_mincut_axis(arr, axis=1, ov=ov, lim=0, prefer_flat=False)
    cur, _ = _offset_mincut_axis(cur, axis=0, ov=ov, lim=0, prefer_flat=False)
    # 縫合可能動到外緣，強制恢復純偏移的外緣帶，保住 2×2 拼縫
    edge = max(2, min(8, ov // 20))
    cur[:, :edge] = pure[:, :edge]
    cur[:, -edge:] = pure[:, -edge:]
    cur[:edge, :] = pure[:edge, :]
    cur[-edge:, :] = pure[-edge:, :]

    v, hs = _tile_seam_scores(cur)
    cv, ch = _edge_profile_corr(cur)
    # 若縫合整體更差，直接用純偏移
    if _seam_rank(cur) > _seam_rank(pure):
        cur = pure
        v, hs = _tile_seam_scores(cur)
        cv, ch = _edge_profile_corr(cur)

    if _seam_rank(cur) > base_rank * 0.95 and (cv + ch) < (base_cv + base_ch) + 0.2:
        return (
            arr.copy(),
            f"拼接無改善（原接縫 {base_v:.0f}+{base_h:.0f}）",
        )
    tag = "半幅偏移拼接（平切）" if iconish else "半幅偏移拼接"
    return (
        cur,
        f"{tag}（接縫 {base_v:.0f}+{base_h:.0f}→{v:.0f}+{hs:.0f}，"
        f"邊緣相關 {base_cv:.2f}/{base_ch:.2f}→{cv:.2f}/{ch:.2f}）",
    )


def shear_align_seamless(arr: np.ndarray) -> tuple[np.ndarray, str]:
    """錯位測完後用環繞錯切攤平（備選；斜紋有時不如偏移拼接）。"""
    base_v, base_h = _tile_seam_scores(arr)
    base_rank = _seam_rank(arr)
    cur = arr.copy()
    total_dy = 0
    total_dx = 0
    best = (base_rank, cur.copy(), 0, 0)
    for _ in range(3):
        dy, dx, _, _ = detect_edge_shifts(cur)
        if abs(dy) <= 1 and abs(dx) <= 1:
            break
        if abs(dy) > 1:
            cur = _wrapped_shear_v(cur, dy)
            total_dy += dy
        if abs(dx) > 1:
            cur = _wrapped_shear_h(cur, dx)
            total_dx += dx
        r = _seam_rank(cur)
        if r < best[0]:
            best = (r, cur.copy(), total_dy, total_dx)
    cur = best[1]
    v, hs = _tile_seam_scores(cur)
    if _seam_rank(cur) > base_rank * 0.92:
        return arr.copy(), "錯切對齊無改善"
    return (
        cur,
        f"錯切對齊 dy≈{best[2]} dx≈{best[3]} "
        f"（接縫 {base_v:.0f}+{base_h:.0f}→{v:.0f}+{hs:.0f}）",
    )


def stitch_align_seamless(arr: np.ndarray) -> tuple[np.ndarray, str]:
    """滿鋪對齊入口：優先多圖偏移拼接，其次錯切。"""
    # 超大圖禁止環繞錯切（全圖 mesh + float 會卡死）
    if max(arr.shape[0], arr.shape[1]) >= 4000:
        return offset_quilt_seamless(arr)
    candidates: list[tuple[float, np.ndarray, str]] = []
    for fn in (offset_quilt_seamless, shear_align_seamless):
        out, msg = fn(arr)
        candidates.append((_seam_rank(out), out, msg))
    candidates.sort(key=lambda t: t[0])
    return candidates[0][1], candidates[0][2]


def try_make_dense_seamless(
    arr: np.ndarray,
    log: Callable[[str], None] | None = None,
) -> tuple[np.ndarray, str]:
    """
    滿鋪紋：單元圖優先保留原圖（或穩定週期裁切）。
    斜紋對齊改在 2×2 用多圖錯位拼接完成，不在單張上半幅偏移。
    最後以色差均衡壓低對邊色帶。
    """
    def _lg(msg: str) -> None:
        if log is not None:
            log(msg)

    base = structural_edge_score(arr)
    base_v, base_h = _tile_seam_scores(arr)
    base_seam = base_v + base_h

    best = arr.copy()
    best_detail = (
        f"保留原圖（接縫 {base_v:.0f}+{base_h:.0f}；2×2 用多圖錯位拼接對齊）"
    )
    best_seam = base_seam

    h, w = arr.shape[:2]
    _lg("  → 滿鋪：開始週期裁切搜尋（保證接縫，可能較慢）…")
    t_crop = time.perf_counter()
    cropped, detail = try_period_crop(arr, log=log)
    _lg(f"  → 滿鋪：週期裁切結束（{time.perf_counter() - t_crop:.1f}s）")
    if cropped is not None:
        csc = structural_edge_score(cropped)
        cv, ch = _tile_seam_scores(cropped)
        crop_seam = cv + ch
        icon_crop = _is_icon_grid_tile(cropped, detail, w, h)
        # 非圖示格：裁切不得比原圖更差（錯誤週期常把指標與可視一起拉壞）
        if not icon_crop and crop_seam >= base_seam * 0.95:
            cropped = None
    if cropped is not None:
        csc = structural_edge_score(cropped)
        cv, ch = _tile_seam_scores(cropped)
        crop_seam = cv + ch
        icon_crop = _is_icon_grid_tile(cropped, detail, w, h)
        abs_ok = crop_seam <= 38.0 and max(cv, ch) <= 30.0
        # 相對原圖大幅改善時放寬絕對門檻（格紋／織紋週期裁切常落在 40–75）
        strong_rel = (
            crop_seam < base_seam * 0.60
            and crop_seam < base_seam - 20.0
            and cv <= base_v * 0.90 + 1.0
            and ch <= base_h * 0.90 + 1.0
            and crop_seam <= 80.0
            and max(cv, ch) <= 45.0
        )
        # 單軸爆縫（常見格紋 H 或 V 單獨高）：總縫下降且最差軸明顯壓低即可
        axis_rescue = (
            crop_seam < base_seam - 5.0
            and max(cv, ch) < max(base_v, base_h) * 0.78
            and crop_seam <= 55.0
            and max(cv, ch) <= 32.0
        )
        accept_abs = abs_ok or strong_rel or axis_rescue
        better_seam = (
            accept_abs
            and crop_seam < base_seam * 0.80
            and crop_seam < base_seam - 8.0
        )
        # axis_rescue 只要總縫變好即可收下
        if axis_rescue and crop_seam < best_seam:
            better_seam = True
        better_struct = accept_abs and csc < base * 0.85 and crop_seam <= base_seam
        if better_struct or better_seam:
            best = cropped.copy()
            best_detail = (
                f"週期裁切 {detail}（接縫 {base_v:.0f}+{base_h:.0f}→{cv:.0f}+{ch:.0f}；"
                "2×2 再多圖錯位拼接）"
            )
            best_seam = crop_seam
        elif crop_seam < best_seam and crop_seam <= base_seam:
            # 圖示格：純週期裁切，不做 soft 色差（會把對邊杯子印成重影）
            # 裁切後接縫仍極高 → 不採用（寧可原圖+對齊）
            if (
                icon_crop
                and crop_seam < base_seam * 0.85
                and crop_seam <= 85.0
                and max(cv, ch) <= 50.0
            ):
                # 圖示格：週期裁切 vs 原圖，選結構安全拋光後接縫更好者
                mean_crop = equalize_edge_means(cropped)
                safe_crop = soft_edge_midpoint(
                    mean_crop, strength=0.55, band_frac=0.06, structure_safe=True
                )
                mean_keep = equalize_edge_means(arr)
                safe_keep = soft_edge_midpoint(
                    mean_keep, strength=0.55, band_frac=0.06, structure_safe=True
                )
                candidates_ig = [
                    ("週期裁切", cropped, crop_seam),
                    ("週期+安全對齊", safe_crop, sum(_tile_seam_scores(safe_crop))),
                    ("原圖+安全對齊", safe_keep, sum(_tile_seam_scores(safe_keep))),
                ]
                # 禁止重影
                candidates_ig = [
                    (n, a, s)
                    for n, a, s in candidates_ig
                    if not _soft_blend_ghosts(arr if "原圖" in n else cropped, a)
                ]
                if candidates_ig:
                    name, tile, seam = min(candidates_ig, key=lambda t: t[2])
                    if seam < best_seam:
                        best = tile
                        best_seam = seam
                        tv, th = _tile_seam_scores(tile)
                        best_detail = (
                            f"{name} {detail}（接縫 {base_v:.0f}+{base_h:.0f}"
                            f"→{tv:.0f}+{th:.0f}；圖示格免色差）"
                        )
            elif not icon_crop:
                # 結構改善但色差仍偏高：先收下，後面色差均衡再壓
                eq, eq_msg = maybe_equalize_seam_colors(cropped)
                ev, eh = _tile_seam_scores(eq)
                eq_seam = ev + eh
                if (
                    (eq_seam <= 38.0 and max(ev, eh) <= 30.0)
                    or (
                        eq_seam < best_seam * 0.60
                        and eq_seam < best_seam - 20.0
                        and eq_seam <= 80.0
                        and max(ev, eh) <= 45.0
                    )
                ) and eq_seam < best_seam:
                    best = eq
                    best_detail = (
                        f"週期裁切+色差 {detail}（接縫 {base_v:.0f}+{base_h:.0f}"
                        f"→{cv:.0f}+{ch:.0f}→{ev:.0f}+{eh:.0f}"
                        f"{('；' + eq_msg) if eq_msg else ''}）"
                    )
                    best_seam = eq_seam

    # 接縫仍高：去光照 → 弱對齊 → 半幅偏移（不限「保留原圖」，週期裁切殘高縫也可救）
    # 棋盤圖示格：允許「平切」半幅偏移（切割偏好地色），禁止舊式硬切撕格
    if best_seam > 22.0 or max(_tile_seam_scores(best)) > 18.0:
        icon_grid = _looks_like_icon_checkerboard(best) or _is_icon_grid_tile(
            best, best_detail, w, h
        )
        sample = best[:: max(1, min(h, w) // 400), :: max(1, min(h, w) // 400)]
        bg_est = tuple(int(x) for x in np.median(sample.reshape(-1, 3), axis=0))
        from app.color_utils import color_distance as _cd

        fg_frac = float(np.mean(_cd(best, bg_est) > 40.0))
        # 密花連枝且無穩定週期：禁止去光照／soft（鬼影）；半幅仍可試，但門檻更嚴
        dense_nonperiodic = (
            fg_frac >= 0.35
            and "週期裁切" not in best_detail
            and not icon_grid
        )
        if (
            "保留原圖" in best_detail
            and max(h, w) < 4000
            and not dense_nonperiodic
        ):
            flat = flatten_illumination(best)
            fv, fh = _tile_seam_scores(flat)
            if fv + fh < best_seam - 1.5:
                eq, eq_msg = maybe_equalize_seam_colors(flat)
                ev, eh = _tile_seam_scores(eq)
                if (
                    ev + eh < best_seam - 2.0
                    and not _soft_blend_ghosts(flat, eq)
                    and not _soft_blend_ghosts(best, eq)
                ):
                    best = eq
                    best_seam = ev + eh
                    best_detail = (
                        f"去光照 {best_detail}"
                        f"{('；' + eq_msg) if eq_msg else ''}"
                    )
        # 圖示格與一般滿鋪都可試半幅偏移；圖示格走 prefer_flat 平切
        # 動物／貼紙大圖章禁止半幅偏移（會從中央切斷）
        stampish = large_stamp_like(best, bg_est, 40.0)
        if not stampish:
            # 優先在當前 best 上半幅；若仍殘高縫再試原圖半幅（週期裁切偏相位時）
            shift_srcs: list[tuple[np.ndarray, str]] = [(best, "")]
            if "週期裁切" in best_detail and max(_tile_seam_scores(best)) > 16.0:
                shift_srcs.append((arr, "原圖"))
            for src_arr, src_tag in shift_srcs:
                aligned, ad = stitch_align_seamless(src_arr)
                av, ah = _tile_seam_scores(aligned)
                # 圖示格：必須大幅改善，且邊緣相關要夠高（否則寧可不撕）
                if icon_grid:
                    cv2, ch2 = _edge_profile_corr(aligned)
                    ok_icon = (
                        av + ah < best_seam * 0.45
                        and av + ah <= 22.0
                        and max(av, ah) <= 14.0
                        and cv2 >= 0.85
                        and ch2 >= 0.85
                    )
                    if ok_icon:
                        best = aligned
                        best_seam = av + ah
                        tag = f"{ad}" if not src_tag else f"{ad}←{src_tag}"
                        best_detail = f"邊緣對齊 {tag}；{best_detail}"
                        break
                elif av + ah < best_seam - 2.0 or max(av, ah) + 4.0 < max(
                    *_tile_seam_scores(best)
                ):
                    # 禁止單軸明顯變差（棋盤／格紋半幅偏移常把一軸拉壞）
                    bv, bh = _tile_seam_scores(best)
                    cv2, ch2 = _edge_profile_corr(aligned)
                    both_ok = av <= bv + 2.0 and ah <= bh + 2.0
                    # 兩軸邊緣相關極高且總縫大幅下降：允許單軸微升
                    strong_corr = (
                        av + ah < best_seam * 0.55
                        and av + ah <= 28.0
                        and max(av, ah) <= 20.0
                        and cv2 >= 0.90
                        and ch2 >= 0.90
                    )
                    # 週期裁切後單軸仍爆：半幅若雙軸均衡且相關夠好則改用
                    balanced = (
                        max(bv, bh) > 30.0
                        and max(av, ah) + 6.0 < max(bv, bh)
                        and max(av, ah) <= 32.0
                        and av + ah <= best_seam + 10.0
                        and cv2 >= 0.78
                        and ch2 >= 0.78
                        and min(av, ah) <= 30.0
                    )
                    # 密花無週期：相關高且接縫明顯下降即可；源縫很高時放寬
                    # （最終仍有強制半幅兜底；此處優先較好的半幅候選）
                    dense_half_ok = False
                    if dense_nonperiodic:
                        src_high = best_seam >= 28.0 or max(bv, bh) >= 22.0
                        dense_half_ok = (
                            av + ah < best_seam * (0.55 if src_high else 0.40)
                            and av + ah <= (32.0 if src_high else 22.0)
                            and max(av, ah) <= (22.0 if src_high else 14.0)
                            and cv2 >= (0.88 if src_high else 0.95)
                            and ch2 >= (0.88 if src_high else 0.95)
                        )
                        if not dense_half_ok:
                            continue
                    if both_ok or strong_corr or balanced or dense_half_ok:
                        # 連枝花／水彩：半幅若跨縫切開暴增則拒絕（分數好看但畫面碎）
                        # 密花／源縫很高時允許切開指標波動（否則無法成四方連續）
                        try:
                            cuts_before = _cross_seam_cut_count(
                                best, bg_est, 40.0
                            )
                            cuts_after = _cross_seam_cut_count(
                                aligned, bg_est, 40.0
                            )
                        except Exception:
                            cuts_before, cuts_after = 0, 0
                        src_high = best_seam >= 28.0 or max(bv, bh) >= 22.0
                        if (
                            cuts_after >= max(3, cuts_before + 2)
                            and not (
                                dense_nonperiodic
                                and (
                                    (cv2 >= 0.95 and ch2 >= 0.95)
                                    or src_high
                                )
                            )
                        ):
                            continue
                        best = aligned
                        best_seam = av + ah
                        tag = f"{ad}" if not src_tag else f"{ad}←{src_tag}"
                        best_detail = f"邊緣對齊 {tag}；{best_detail}"
                        # 半幅偏移後禁止 soft 混合，只做均值對齊
                        # 超大圖接縫已夠好則跳過（全圖 float 對齊太慢）
                        if max(h, w) < 4000 or best_seam > 18.0:
                            mean2 = equalize_edge_means(best, band_frac=0.08)
                            ev2, eh2 = _tile_seam_scores(mean2)
                            if ev2 + eh2 < best_seam - 1.0:
                                best = mean2
                                best_seam = ev2 + eh2
                                best_detail = (
                                    f"{best_detail}；邊緣均值對齊 "
                                    f"{av:.0f}+{ah:.0f}→{ev2:.0f}+{eh2:.0f}"
                                )
                        break

    return best, best_detail


def _pure_half_offset(arr: np.ndarray) -> np.ndarray:
    """純半幅滾動：外緣取自原圖中心相鄰列，數學上四方連續。"""
    h, w = arr.shape[:2]
    return np.roll(np.roll(arr, w // 2, axis=1), h // 2, axis=0)


def _force_half_seamless(
    arr: np.ndarray,
    detail: str,
    *,
    seam_sum_lim: float = 14.0,
    seam_max_lim: float = 10.0,
) -> tuple[np.ndarray, str]:
    """
    源圖非無縫時：接縫仍高則強制半幅純滾做成四方連續。
    不連續會移到單元中央；外緣保證可平鋪（這是工業常用做法）。
    """
    v0, h0 = _tile_seam_scores(arr)
    if v0 + h0 <= seam_sum_lim and max(v0, h0) <= seam_max_lim:
        return arr, detail

    already_half = ("半幅" in detail) or ("強制半幅" in detail)
    # 先前半幅／min-cut 仍高：對當前圖再純滾會近原圖，改做均值對齊壓色帶
    if already_half:
        if v0 + h0 <= 22.0 and max(v0, h0) <= 16.0:
            return arr, detail
        mean_eq = equalize_edge_means(arr, band_frac=0.08)
        ev, eh = _tile_seam_scores(mean_eq)
        if ev + eh < v0 + h0 - 0.8:
            return (
                mean_eq,
                f"{detail}；邊緣均值對齊 {v0:.0f}+{h0:.0f}→{ev:.0f}+{eh:.0f}",
            )
        return arr, detail

    pure = _pure_half_offset(arr)
    pv, ph = _tile_seam_scores(pure)
    # 純滾外緣數學連續，一律採用（高頻紋理可能讓 RGB 分數偏高，不代表切斷）
    detail = (
        f"強制半幅四方連續（接縫 {v0:.0f}+{h0:.0f}→{pv:.0f}+{ph:.0f}）；{detail}"
    )
    arr = pure
    v0, h0 = pv, ph
    if v0 + h0 > 8.0 or max(v0, h0) > 6.0:
        mean_eq = equalize_edge_means(arr, band_frac=0.08)
        ev, eh = _tile_seam_scores(mean_eq)
        if ev + eh < v0 + h0 - 0.4:
            arr = mean_eq
            detail = (
                f"{detail}；邊緣均值對齊 "
                f"{v0:.0f}+{h0:.0f}→{ev:.0f}+{eh:.0f}"
            )
    return arr, detail


def _finalize_unit(arr: np.ndarray, detail: str) -> tuple[Image.Image, str]:
    """最終色差拋光（滿鋪／點綴共用出口）。"""
    # 非無縫源圖：接縫仍高則強制做成四方連續（優先半幅純滾）
    arr, detail = _force_half_seamless(arr, detail)

    # 清邊補花／卡通圖章：禁止 soft 對邊混合（會把對側動物印成殘影）
    # 仍允許邊緣均值對齊壓白齒／色帶（不做像素混合）
    if "清邊" in detail or "補花" in detail:
        v0, h0 = _tile_seam_scores(arr)
        mean_eq = equalize_edge_means(arr, band_frac=0.04)
        v1, h1 = _tile_seam_scores(mean_eq)
        if v1 + h1 < v0 + h0 - 0.4 and not _soft_blend_ghosts(arr, mean_eq):
            detail = f"{detail}；邊緣均值對齊 {v0:.0f}+{h0:.0f}→{v1:.0f}+{h1:.0f}"
            return Image.fromarray(mean_eq, mode="RGB"), detail
        return Image.fromarray(arr, mode="RGB"), detail
    # 點綴保留原圖：禁止 soft／色差混合（柯基等圖章會出鬼影／暈邊）
    # 若已強制半幅，直接返回（外緣已連續）
    if "點綴保留原圖" in detail or "晶格未對齊" in detail:
        return Image.fromarray(arr, mode="RGB"), detail
    iconish = ("圖示格免色差" in detail) or _looks_like_icon_checkerboard(arr)
    # 半幅偏移後外緣已連續：禁止 soft 混合（會把不相鄰內容印壞，造成「切斷」假象）
    # 僅允許均值／方差對齊壓殘餘色帶
    if "半幅偏移" in detail or "強制半幅" in detail:
        v0, h0 = _tile_seam_scores(arr)
        if v0 + h0 <= 22.0 and max(v0, h0) <= 16.0:
            return Image.fromarray(arr, mode="RGB"), detail
        # 均值／方差對齊不做對側像素混合，不套用 soft 鬼影門檻
        # （1(84) 等半幅後色帶靠此壓到個位數）
        mean_eq = equalize_edge_means(arr, band_frac=0.08)
        v1, h1 = _tile_seam_scores(mean_eq)
        if v1 + h1 < v0 + h0 - 1.0:
            detail = f"{detail}；邊緣均值對齊 {v0:.0f}+{h0:.0f}→{v1:.0f}+{h1:.0f}"
            return Image.fromarray(mean_eq, mode="RGB"), detail
        return Image.fromarray(arr, mode="RGB"), detail
    if iconish:
        # 圖示格：去光照 + 均值對齊 + 結構安全 soft（只糊平坦底色）
        flat = flatten_illumination(arr, radius=max(32, min(arr.shape[0], arr.shape[1]) // 4))
        mean_eq = equalize_edge_means(flat)
        safe = soft_edge_midpoint(mean_eq, strength=0.72, band_frac=0.08, structure_safe=True)
        v0, h0 = _tile_seam_scores(arr)
        best, best_s, label = arr, v0 + h0, ""
        for name, cand in (
            ("去光照均值", mean_eq),
            ("結構安全對齊", safe),
        ):
            # 結構安全路徑：ghost 門檻放寬（平坦底色混合不應算重影）
            if name != "結構安全對齊" and _soft_blend_ghosts(arr, cand):
                continue
            # 結構安全仍可能把圖示印淡：若偵測到重影則跳過
            if name == "結構安全對齊" and _soft_blend_ghosts(arr, cand):
                continue
            v, h = _tile_seam_scores(cand)
            if v + h < best_s - 1.0:
                best, best_s, label = cand, v + h, name
        if label:
            v1, h1 = _tile_seam_scores(best)
            detail = f"{detail}；{label} {v0:.0f}+{h0:.0f}→{v1:.0f}+{h1:.0f}"
            return Image.fromarray(best, mode="RGB"), detail
        return Image.fromarray(arr, mode="RGB"), detail
    eq, msg = maybe_equalize_seam_colors(arr)
    if msg and _soft_blend_ghosts(arr, eq):
        return Image.fromarray(arr, mode="RGB"), detail
    # 密花連枝且無週期裁切：禁止 soft／窄帶混合（罌粟花會出鬼影）
    dense_keep = (
        "保留原圖" in detail
        and "週期裁切" not in detail
        and "半幅" not in detail
    )
    if dense_keep:
        sample = arr[::8, ::8].reshape(-1, 3)
        bg_est = tuple(int(x) for x in np.median(sample, axis=0))
        if float(np.mean(color_distance(arr, bg_est) > 40.0)) >= 0.35:
            return Image.fromarray(arr, mode="RGB"), detail
    if msg:
        detail = f"{detail}；{msg}"
        arr = eq
    # 清邊／點綴後仍偏高：再試一輪結構安全 soft（只糊平坦區）
    v0, h0 = _tile_seam_scores(arr)
    if v0 + h0 > 14.0 or max(v0, h0) > 10.0:
        safe = soft_edge_midpoint(arr, strength=0.55, band_frac=0.06, structure_safe=True)
        if not _soft_blend_ghosts(arr, safe):
            v1, h1 = _tile_seam_scores(safe)
            if v1 + h1 < v0 + h0 - 1.0:
                arr = safe
                detail = f"{detail}；結構安全對齊 {v0:.0f}+{h0:.0f}→{v1:.0f}+{h1:.0f}"
                v0, h0 = v1, h1
    # 週期裁切後殘縫仍貼審計門檻：再加一輪均值＋弱 soft
    if ("週期裁切" in detail) and (v0 + h0 > 28.0 or max(v0, h0) > 18.0):
        mean2 = equalize_edge_means(arr, band_frac=0.06)
        mild2 = soft_edge_midpoint(mean2, strength=0.35, band_frac=0.04)
        for cand, name in ((mean2, "邊緣均值對齊"), (mild2, "色差弱對齊")):
            if _soft_blend_ghosts(arr, cand):
                continue
            v2, h2 = _tile_seam_scores(cand)
            if v2 + h2 < v0 + h0 - 0.8 and max(v2, h2) <= max(v0, h0) + 0.5:
                detail = f"{detail}；{name} {v0:.0f}+{h0:.0f}→{v2:.0f}+{h2:.0f}"
                arr = cand
                v0, h0 = v2, h2
                break
    return Image.fromarray(arr, mode="RGB"), detail


def tile_2x2_multi(
    unit: Image.Image,
    *,
    prefer_plain: bool = False,
) -> tuple[Image.Image, str, tuple[int, int]]:
    """
    2x2 多圖錯位拼接（按你說的補白方式）：
    1) 先錯位貼圖（會空出一塊）
    2) 空白處再貼同一張圖：下面緊挨上面、左邊緊挨右邊

    prefer_plain：規則點綴單元已對齊時直接拼接，避免錯位補白捏出雙列。
    """
    arr = np.asarray(unit.convert("RGB"), dtype=np.uint8)
    h, w = arr.shape[:2]

    plain_v = float(
        np.mean(np.abs(arr[:, -1].astype(np.float64) - arr[:, 0].astype(np.float64)))
    )
    plain_h = float(
        np.mean(np.abs(arr[-1].astype(np.float64) - arr[0].astype(np.float64)))
    )

    def _plain() -> tuple[Image.Image, str, tuple[int, int]]:
        out = np.empty((2 * h, 2 * w, 3), dtype=np.uint8)
        out[:h, :w] = arr
        out[:h, w:] = arr
        out[h:, :w] = arr
        out[h:, w:] = arr
        return (
            Image.fromarray(out, mode="RGB"),
            f"2x2 直接拼接（中縫 {plain_v:.0f}+{plain_h:.0f}）",
            (w, h),
        )

    # 邊緣剖面相關高且對邊色差低 → 直接拼接；僅相關高但色差仍大時走錯位
    cv, ch = _edge_profile_corr(arr)
    plain_sum = plain_v + plain_h
    if prefer_plain or (cv >= 0.85 and ch >= 0.85 and plain_sum <= 18.0):
        return _plain()
    # 環繞圖章：對邊是同一隻動物的左右半，剖面相關低但像素差小
    if plain_sum <= 16.0 and max(plain_v, plain_h) <= 12.0:
        return _plain()
    # 半幅偏移單元：外緣來自原圖中心相鄰列，高頻紋理會讓 plain_sum 偏高，
    # 但可視上連續；相關尚可時優先直接拼接，避免錯位補白造假縫
    if cv >= 0.50 and ch >= 0.50 and plain_sum <= 80.0:
        return _plain()

    # 疏點綴（花圈／碎花）：錯位緊挨補白會打亂晶格節奏，看起來像接縫空一截
    # 滿鋪斜紋前景通常更密，仍走後面的錯位邏輯
    bg_est = np.median(arr.reshape(-1, 3), axis=0)
    fg_r = float(
        np.mean(
            np.linalg.norm(arr.astype(np.float64) - bg_est, axis=2) > 40.0
        )
    )
    if fg_r < 0.20 and (plain_v + plain_h) < 90.0:
        return _plain()

    blurred = np.asarray(
        Image.fromarray(arr).filter(ImageFilter.GaussianBlur(radius=12)),
        dtype=np.uint8,
    )
    lim = min(400, h // 3, w // 3)
    dy_b, dx_b, sv_b, sh_b = detect_edge_shifts(blurred, band=24, lim=lim)
    dy_r, dx_r, sv_r, sh_r = detect_edge_shifts(arr, band=12, lim=lim)
    dy, dx = (dy_b, dx_b) if (sv_b + sh_b) <= (sv_r + sh_r) * 0.92 else (dy_r, dx_r)

    def _paste(canvas: np.ndarray, yy: int, xx: int, empty_only: bool = False) -> None:
        ch, cw = canvas.shape[:2]
        y0, x0 = max(0, yy), max(0, xx)
        y1, x1 = min(ch, yy + h), min(cw, xx + w)
        if y1 <= y0 or x1 <= x0:
            return
        patch = arr[
            y0 - yy : y0 - yy + (y1 - y0),
            x0 - xx : x0 - xx + (x1 - x0),
        ]
        if not empty_only:
            canvas[y0:y1, x0:x1] = patch
            return
        region = canvas[y0:y1, x0:x1]
        empty = np.all(region == 0, axis=2)
        if np.any(empty):
            filled = region.copy()
            filled[empty] = patch[empty]
            canvas[y0:y1, x0:x1] = filled

    def _build_v(sy: int) -> tuple[np.ndarray, float, float]:
        """右列相對左列上下錯 sy；空白用同一張圖上下緊挨補滿。"""
        out = np.zeros((2 * h, 2 * w, 3), dtype=np.uint8)
        # 左列：兩張上下緊挨
        _paste(out, 0, 0)
        _paste(out, h, 0)
        # 右列：錯位後的第一張
        _paste(out, -sy, w)
        # 空白處：緊挨上下再貼（只填空）
        _paste(out, -sy - h, w, empty_only=True)
        _paste(out, -sy + h, w, empty_only=True)
        _paste(out, -sy + 2 * h, w, empty_only=True)
        _paste(out, -sy + 3 * h, w, empty_only=True)
        mid_v = float(
            np.mean(
                np.abs(out[:h, w - 1].astype(np.float64) - out[:h, w].astype(np.float64))
            )
        )
        mid_h = float(
            np.mean(
                np.abs(out[h - 1, :w].astype(np.float64) - out[h, :w].astype(np.float64))
            )
        )
        return out, mid_v, mid_h

    def _build_h(sx: int) -> tuple[np.ndarray, float, float]:
        """下行相對上行左右錯 sx；空白用同一張圖左右緊挨補滿。"""
        out = np.zeros((2 * h, 2 * w, 3), dtype=np.uint8)
        # 上行：兩張左右緊挨
        _paste(out, 0, 0)
        _paste(out, 0, w)
        # 下行：錯位後的第一張
        _paste(out, h, -sx)
        _paste(out, h, -sx - w, empty_only=True)
        _paste(out, h, -sx + w, empty_only=True)
        _paste(out, h, -sx + 2 * w, empty_only=True)
        _paste(out, h, -sx + 3 * w, empty_only=True)
        mid_v = float(
            np.mean(
                np.abs(out[:h, w - 1].astype(np.float64) - out[:h, w].astype(np.float64))
            )
        )
        mid_h = float(
            np.mean(
                np.abs(out[h - 1, :w].astype(np.float64) - out[h, :w].astype(np.float64))
            )
        )
        return out, mid_v, mid_h

    candidates_v: list[tuple[float, np.ndarray, int, float, float]] = []
    candidates_h: list[tuple[float, np.ndarray, int, float, float]] = []
    for sy in (dy, -dy):
        canvas, mv, mh = _build_v(sy)
        empty = float(np.mean(np.all(canvas == 0, axis=2)))
        candidates_v.append((mv + empty * 100.0, canvas, sy, mv, mh))
    for sx in (dx, -dx):
        canvas, mv, mh = _build_h(sx)
        empty = float(np.mean(np.all(canvas == 0, axis=2)))
        candidates_h.append((mh + empty * 100.0, canvas, sx, mv, mh))
    candidates_v.sort(key=lambda t: t[0])
    candidates_h.sort(key=lambda t: t[0])

    # 誰能把「對應方向」的接縫明顯修好，就用誰；斜紋優先豎向錯位
    pick = None
    if candidates_v and candidates_v[0][3] < plain_v * 0.75:
        _, canvas, shift, mv, mh = candidates_v[0]
        pick = ("V", shift, canvas, mv, mh)
    elif candidates_h and candidates_h[0][4] < plain_h * 0.75:
        _, canvas, shift, mv, mh = candidates_h[0]
        pick = ("H", shift, canvas, mv, mh)
    elif candidates_v and candidates_v[0][3] <= candidates_h[0][4]:
        _, canvas, shift, mv, mh = candidates_v[0]
        pick = ("V", shift, canvas, mv, mh)
    elif candidates_h:
        _, canvas, shift, mv, mh = candidates_h[0]
        pick = ("H", shift, canvas, mv, mh)

    if pick is None or (pick[3] + pick[4]) >= plain_sum * 0.90:
        # 錯位無明顯收益：僅在直接拼接本身可接受時才用 plain
        if plain_sum <= 22.0 and max(plain_v, plain_h) <= 16.0:
            return _plain()
        # 否則仍用錯位結果（寧可保留較高中縫，避免 plain 爆出十字縫）
        if pick is not None:
            axis, shift, canvas, mv, mh = pick
            empty = np.all(canvas == 0, axis=2)
            if np.any(empty):
                canvas = canvas.copy()
                canvas[empty] = np.median(arr.reshape(-1, 3), axis=0).astype(np.uint8)
            return (
                Image.fromarray(canvas, mode="RGB"),
                f"2x2 錯位後緊挨補白 {axis}={shift} "
                f"（中縫 {plain_v:.0f}+{plain_h:.0f}→{mv:.0f}+{mh:.0f}）",
                (w, h),
            )
        return _plain()

    # 拒單軸「假修好」：一軸暴跌、另一軸幾乎沒動 → 棋盤等會拼出雙列同色
    mv, mh = pick[3], pick[4]
    if (
        min(mv, mh) < min(plain_v, plain_h) * 0.25
        and max(mv, mh) > max(plain_v, plain_h) * 0.85
        and max(mv, mh) > 40.0
    ):
        return _plain()

    axis, shift, canvas, mv, mh = pick
    empty = np.all(canvas == 0, axis=2)
    if np.any(empty):
        canvas[empty] = np.median(arr.reshape(-1, 3), axis=0).astype(np.uint8)

    return (
        Image.fromarray(canvas, mode="RGB"),
        f"2x2 錯位後緊挨補白 {axis}={shift} "
        f"（中縫 {plain_v:.0f}+{plain_h:.0f}→{mv:.0f}+{mh:.0f}）",
        (w, h),
    )



def tile_2x2(unit: Image.Image) -> Image.Image:
    """將單元圖拼成 2×2；滿鋪斜紋會自動多圖錯位對齊。"""
    preview, _, _ = tile_2x2_multi(unit)
    return preview


def foreground_ratio(
    arr: np.ndarray,
    bg: Sequence[int],
    threshold: float,
) -> float:
    return float(np.mean(_foreground_mask(arr, bg, threshold)))


def resolve_margin_px(
    image: Image.Image,
    margin: float,
    margin_is_percent: bool,
) -> int:
    w, h = image.size
    if margin_is_percent:
        return int(round(min(w, h) * float(margin)))
    return int(round(float(margin)))


def _looks_like_discrete_motifs(
    arr: np.ndarray,
    bg: Sequence[int],
    threshold: float,
) -> bool:
    """粗估是否為規則點綴（多個中等連通塊 + 近似晶格），而非整片相連或不規則散點。"""
    h, w = arr.shape[:2]
    # 縮小後估連通域，避免全圖 BFS 太慢
    scale = max(h, w) / 192.0
    small_rgb = np.asarray(
        Image.fromarray(arr).resize(
            (max(32, int(round(w / scale))), max(32, int(round(h / scale)))),
            Image.Resampling.BILINEAR,
        )
    )
    fg = _foreground_mask(small_rgb, bg, threshold)
    # 密花／滿鋪覆蓋高：走滿鋪週期，勿誤判點綴晶格（12k 密花會卡死）
    if float(np.mean(fg)) > 0.42:
        return False
    # 垂直／水平條紋：某一軸投影幾乎恆定 → 走滿鋪
    row_p = fg.mean(axis=1)
    col_p = fg.mean(axis=0)
    if float(row_p.std()) < 0.02 and float(col_p.std()) > 0.08:
        return False
    if float(col_p.std()) < 0.02 and float(row_p.std()) > 0.08:
        return False
    comps_n, _, stats, _ = cv2.connectedComponentsWithStats(
        fg.astype(np.uint8), connectivity=4
    )
    if comps_n - 1 < 8:
        return False
    areas = sorted(
        (int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, comps_n)),
        reverse=True,
    )
    total = int(np.count_nonzero(fg)) or 1
    # 最大塊不能佔掉大半前景（否則是滿鋪連成一片）
    if areas[0] > total * 0.35:
        return False
    mid = [a for a in areas if 20 <= a <= total * 0.2]
    if len(mid) < 6:
        return False
    # 不規則散點（手繪四方連續動物等）不要走晶格硬門禁
    return looks_like_regular_lattice(arr, bg, threshold)


def make_seamless_hard_cut(
    image: Image.Image,
    bg: Sequence[int] | None = None,
    margin: float = 0.03,
    threshold: float = 40.0,
    margin_is_percent: bool = True,
    log: Callable[[str], None] | None = None,
) -> tuple[Image.Image, str]:
    """
    產生四方連續單元圖，回傳 (圖, 模式說明)。

    - 規則點綴（星／花／愛心等）：質心晶格構造裁切，跨縫完整性硬門禁
    - 滿鋪：週期裁切（與邊緣帶無關）
    - 稀疏不規則點綴：清碰邊殘花 + 環繞補花（需邊緣帶 > 0，且非規則陣列）

    log：進度回呼（批次時印到命令列；不改變選圖／接縫判定）。
    """
    from app.discrete_lattice import try_make_discrete_seamless

    def _lg(msg: str) -> None:
        if log is not None:
            log(msg)

    if bg is None:
        bg = detect_background(image)

    arr = _to_rgb_array(image, bg)
    h, w = arr.shape[:2]

    if margin_is_percent:
        margin_px = int(round(min(h, w) * float(margin)))
    else:
        margin_px = int(round(float(margin)))
    margin_px = max(0, margin_px)

    ratio = foreground_ratio(arr, bg, threshold)
    discrete = _looks_like_discrete_motifs(arr, bg, threshold)
    _lg(
        f"  → 分類：{'點綴/晶格' if discrete else '滿鋪或稀疏'}，"
        f"前景 {ratio:.0%}，尺寸 {w}×{h}"
    )

    # 規則點綴：構造裁切；禁止清邊打散陣列、禁止邊緣相關備選勝出
    if discrete:
        from app.discrete_lattice import (
            _fg_mask_merged as _d_fg,
            _interior_median_motif_area,
            _rank_tile,
            wrap_gap_ratios,
        )

        _lg("  → 演算法：點綴晶格…")
        t_d = time.perf_counter()
        unit_d, detail_d = try_make_discrete_seamless(
            arr, bg, threshold, log=log
        )
        _lg(f"  → 點綴晶格完成（{time.perf_counter() - t_d:.1f}s）：{detail_d}")
        # 僅硬通過且接縫夠好時直接採用。軟通過／過短週期／接縫仍高
        # 若提前 return，會蓋掉明顯更好的週期裁切／半幅偏移。
        # 源圖色差接縫已極低時勿提早採用晶格大裁切（交錯爪印會被裁歪）。
        sv_src0, sh_src0 = _tile_seam_scores(arr)
        src_seam_excellent = (sv_src0 + sh_src0) <= 5.0 and max(
            sv_src0, sh_src0
        ) <= 3.0
        if (
            (not src_seam_excellent)
            and "FAIL" not in detail_d
            and "軟通過" not in detail_d
        ):
            sv_d0, sh_d0 = _tile_seam_scores(unit_d)
            small_period = False
            m_per = re.search(r"晶格週期\s*(\d+)\s*[×x]\s*(\d+)", detail_d)
            if m_per and (
                int(m_per.group(1)) < 80 or int(m_per.group(2)) < 80
            ):
                small_period = True
            if (
                not small_period
                and (sv_d0 + sh_d0) <= 28.0
                and max(sv_d0, sh_d0) <= 20.0
            ):
                return _finalize_unit(
                    unit_d,
                    f"點綴（前景 {ratio:.0%}）：{detail_d}",
                )
        elif src_seam_excellent and "原圖（接縫已低）" in detail_d:
            return _finalize_unit(
                unit_d,
                f"點綴保留原圖（前景 {ratio:.0%}，接縫已低）",
            )
        # 軟通過／FAIL：用「跨縫完整性／間距」選備選，禁止只靠接縫色差
        # （錯誤週期裁切常把色差修好，卻把花距拉成 2 倍，看起來很空）
        from app.discrete_lattice import _downscale_for_discrete_analysis
        from app.discrete_lattice import _large_motif_median as _d_large_med
        from app.discrete_lattice import _rank_tile_light as _d_rank_light

        huge_tile = max(h, w) >= 4000
        if huge_tile:
            # 超大圖前景／中位面積改在縮圖估，避免全圖閉運算卡死
            work_m, scale_m = _downscale_for_discrete_analysis(arr, max_side=1600)
            med_s = _interior_median_motif_area(_d_fg(work_m, bg, threshold))
            med = med_s * scale_m * scale_m
            cut_med = (
                _d_large_med(work_m, bg, threshold, med_s) * scale_m * scale_m
                if med_s >= 40
                else None
            )
        else:
            med = _interior_median_motif_area(_d_fg(arr, bg, threshold))
            cut_med = _d_large_med(arr, bg, threshold, med) if med >= 40 else med

        def _integrity_key(tile: np.ndarray) -> tuple[int, float]:
            # 超大單元禁止完整連通域評分（會卡數分鐘）
            if huge_tile or max(tile.shape[0], tile.shape[1]) >= 4000:
                tier, sc = _d_rank_light(
                    tile, bg, threshold, med, cut_med=cut_med
                )
                sv, shs = _tile_seam_scores(tile)
                return (tier, sc + (sv + shs) * 0.25)
            tier, sc, gx, gy = _rank_tile(
                tile, bg, threshold, med, cut_med=cut_med
            )
            gap_pen = (abs(gx - 1.0) + abs(gy - 1.0)) * 60.0
            return (tier, sc + gap_pen)

        candidates: list[tuple[tuple[int, float], np.ndarray, str]] = []
        key_d = _integrity_key(unit_d)
        sv_d0, sh_d0 = _tile_seam_scores(unit_d)
        # FAIL 結果不得靠「分數碰巧好看」壓過滿鋪／原圖
        if "FAIL" in detail_d:
            key_d = (max(key_d[0], 2) + 1, key_d[1] + 120.0)
        elif "軟通過" in detail_d:
            # 軟通過但色差／色帶仍差：略降級，讓硬通過級滿鋪週期有機會勝出
            if (sv_d0 + sh_d0) > 12.0 or max(sv_d0, sh_d0) > 10.0:
                key_d = (key_d[0] + 1, key_d[1] + (sv_d0 + sh_d0))
        else:
            # 硬通過但未提早返回（接縫高／過短週期）：降級好讓滿鋪勝出
            if (sv_d0 + sh_d0) > 28.0 or max(sv_d0, sh_d0) > 20.0:
                key_d = (key_d[0] + 1, key_d[1] + (sv_d0 + sh_d0))
            m_per2 = re.search(r"晶格週期\s*(\d+)\s*[×x]\s*(\d+)", detail_d)
            if m_per2 and (
                int(m_per2.group(1)) < 80 or int(m_per2.group(2)) < 80
            ):
                key_d = (key_d[0] + 1, key_d[1] + 80.0)
        soft_or_fail_msg = (
            f"點綴（前景 {ratio:.0%}）：{detail_d}"
            if "軟通過" in detail_d
            else f"點綴最佳嘗試（前景 {ratio:.0%}）：{detail_d}"
        )
        candidates.append(
            (
                key_d,
                unit_d,
                soft_or_fail_msg,
            )
        )
        keep_arr = arr.copy()
        keep_msg = f"點綴保留原圖（前景 {ratio:.0%}，晶格未對齊）"
        kv0, kh0 = _tile_seam_scores(keep_arr)
        stampish = large_stamp_like(keep_arr, bg, threshold)
        cuts_keep0 = (
            0 if huge_tile else _cross_seam_cut_count(arr, bg, threshold)
        )
        # 點綴保留：禁止去光照／soft 色差（水彩碎花會大面積改色且切開不降）
        # 僅允許半幅／邊緣對齊，且切開數不得變差
        if kv0 + kh0 > 26.0 and (not stampish) and (not huge_tile):
            _lg("  → 切換：邊緣對齊…")
            t_al = time.perf_counter()
            icon_grid = _is_icon_grid_tile(keep_arr, keep_msg, w, h)
            aligned, ad = stitch_align_seamless(keep_arr)
            _lg(f"  → 邊緣對齊完成（{time.perf_counter() - t_al:.1f}s）")
            kav, kah = _tile_seam_scores(aligned)
            cur_sum = kv0 + kh0
            cuts_al = _cross_seam_cut_count(aligned, bg, threshold)
            cuts_ok = cuts_al <= cuts_keep0 + 0
            if icon_grid or _looks_like_icon_checkerboard(keep_arr):
                cv2c, ch2 = _edge_profile_corr(aligned)
                if (
                    cuts_ok
                    and kav + kah < cur_sum * 0.45
                    and kav + kah <= 22.0
                    and max(kav, kah) <= 14.0
                    and cv2c >= 0.85
                    and ch2 >= 0.85
                ):
                    keep_arr = aligned
                    keep_msg = f"邊緣對齊 {ad}；{keep_msg}"
            elif cuts_ok and kav + kah < cur_sum - 2.0:
                if kav <= kv0 + 2.0 and kah <= kh0 + 2.0:
                    keep_arr = aligned
                    keep_msg = f"邊緣對齊 {ad}；{keep_msg}"
        candidates.append(
            (
                _integrity_key(keep_arr),
                keep_arr,
                keep_msg,
            )
        )
        why = "FAIL" if "FAIL" in detail_d else "軟通過或接縫偏高"
        _lg(
            f"  → 難圖說明：點綴{why}（接縫 {sv_d0:.0f}+{sh_d0:.0f}），"
            "切換滿鋪週期搜尋以保證接縫（可能較慢，請稍候）…"
        )
        t_f = time.perf_counter()
        unit_f, detail_f = try_make_dense_seamless(arr, log=log)
        _lg(f"  → 滿鋪回退完成（{time.perf_counter() - t_f:.1f}s）：{detail_f}")
        if huge_tile:
            # 超大圖跳過全圖 wrap_gap（連通域），改用輕量接縫懲罰
            gap0 = gapf = 0.0
            gxf = gyf = 1.0
            dense_key = _integrity_key(unit_f)
            sv0, sh0 = _tile_seam_scores(arr)
            svf, shf = _tile_seam_scores(unit_f)
            if (svf + shf) > (sv0 + sh0) * 0.98:
                dense_key = (dense_key[0] + 1, dense_key[1] + 40.0)
        else:
            gx0, gy0 = wrap_gap_ratios(arr, bg, threshold, med=med)
            gxf, gyf = wrap_gap_ratios(unit_f, bg, threshold, med=med)
            gap0 = abs(gx0 - 1.0) + abs(gy0 - 1.0)
            gapf = abs(gxf - 1.0) + abs(gyf - 1.0)
            dense_key = _integrity_key(unit_f)
            # 間距明顯變差（例如≈2 倍空檔）→ 降級。
            # 花距拉稀時禁止「接縫碰巧變好」只吃輕罰（否則會勝過保留原圖）。
            if gapf > gap0 + 0.25 or max(gxf, gyf) >= 1.6:
                sv0, sh0 = _tile_seam_scores(arr)
                svf, shf = _tile_seam_scores(unit_f)
                if max(gxf, gyf) >= 1.8 or gapf > gap0 + 0.55:
                    dense_key = (dense_key[0] + 1, dense_key[1] + 220.0)
                elif (svf + shf) < (sv0 + sh0) * 0.75:
                    dense_key = (dense_key[0], dense_key[1] + 40.0)
                else:
                    dense_key = (dense_key[0] + 1, dense_key[1] + 200.0)
        if stampish and "半幅偏移" in detail_f:
            dense_key = (dense_key[0] + 1, dense_key[1] + 250.0)
        candidates.append(
            (
                dense_key,
                unit_f,
                f"滿鋪回退（點綴未對齊，前景 {ratio:.0%}）：{detail_f}",
            )
        )
        # 接縫偏高時放寬清邊門檻（恐龍／切斷圖示必須清邊）
        clear_ratio_lim = 0.55 if (kv0 + kh0) > 30.0 else 0.45
        # 超大密花跳過清邊（全圖連通域／補花會卡死）
        if huge_tile and ratio >= 0.20:
            clear_ratio_lim = -1.0
        # 原圖接縫已不太差的非大圖章：清邊易咬動物／留下鬼影，寧可不清
        if (not stampish) and (kv0 + kh0) <= 28.0 and max(kv0, kh0) <= 18.0:
            clear_ratio_lim = -1.0
        cuts0 = (
            0
            if huge_tile
            else _cross_seam_cut_count(arr, bg, threshold)
        )
        # GUI 邊緣帶=0 時，碰邊殘花仍自動清（大圖章被切斷必須補）
        clear_px = margin_px
        if clear_px <= 0 and (
            "FAIL" in detail_d or cuts0 >= 1 or stampish
        ):
            clear_px = 3
        if clear_px > 0 and ratio < clear_ratio_lim:
            _lg("  → 切換：清邊補花…")
            motifs = extract_interior_motifs(arr, bg, threshold, clear_px)
            cleaned, removed = remove_edge_touching_components(
                arr, bg, threshold, clear_px
            )
            if len(motifs) >= 2 and len(removed) > 0:
                filled = refill_with_wrapped_motifs(
                    cleaned, bg, threshold, motifs, removed, seed=42
                )
                worse = (
                    structural_edge_score(filled)
                    > structural_edge_score(arr) * 1.02
                )
                sv0, sh0 = _tile_seam_scores(arr)
                svf, shf = _tile_seam_scores(filled)
                seam_sum0 = sv0 + sh0
                seam_sumf = svf + shf
                ratio_f = foreground_ratio(filled, bg, threshold)
                gxf2, gyf2 = wrap_gap_ratios(
                    filled, bg, threshold, med=med
                )
                gapf2 = abs(gxf2 - 1.0) + abs(gyf2 - 1.0)
                # 連枝花布清邊會把大半前景清掉，接縫分數變好但畫面變空
                density_collapse = ratio_f < max(0.04, ratio * 0.55)
                gap_blow = (
                    max(gxf2, gyf2) >= 1.45
                    and gapf2 > gap0 + 0.25
                )
                density_ok = ratio_f >= ratio * 0.75
                gap_ok_fill = (
                    max(gxf2, gyf2) <= 1.35 and min(gxf2, gyf2) >= 0.65
                )
                ruined = _clear_edge_ruins_motifs(
                    arr, filled, bg, threshold
                )
                # 非大圖章：接縫變差／未改善一律不進候選（花布清邊易咬出白齒縫）
                # 大圖章切斷才允許小幅接縫代價換完整性
                if not stampish and (
                    seam_sumf > seam_sum0 + 0.5
                    or (seam_sumf >= seam_sum0 * 0.98 and seam_sumf > 8.0)
                ):
                    pass
                elif stampish and seam_sumf > seam_sum0 + 0.8:
                    pass
                elif worse and not stampish:
                    pass
                elif density_collapse and not stampish:
                    pass
                elif gap_blow and not stampish:
                    pass
                elif ruined:
                    _lg("  → 清邊補花：偵測到抹花／鬼影，捨棄")
                    pass
                else:
                    filled_key = _integrity_key(filled)
                    if stampish and (cuts0 >= 1) and seam_sumf <= seam_sum0 + 0.5:
                        filled_key = (0, filled_key[1] * 0.5)
                    elif (
                        cuts0 >= 1
                        and seam_sumf <= seam_sum0 + 0.5
                        and density_ok
                        and gap_ok_fill
                        and not ruined
                    ):
                        # 僅密度／間距仍正常時，才因跨縫切斷升到硬通過級
                        filled_key = (0, filled_key[1] * 0.75)
                    elif seam_sumf > seam_sum0 + 0.5:
                        filled_key = (filled_key[0] + 1, filled_key[1] + 80.0)
                    candidates.append(
                        (
                            filled_key,
                            filled,
                            f"點綴未對齊改清邊補花（前景 {ratio:.0%}）",
                        )
                    )
        candidates.sort(key=lambda t: t[0])
        _score, best_arr, best_msg = candidates[0]
        return _finalize_unit(best_arr, best_msg)

    # 稀疏不規則才清邊補花（與滿鋪週期裁切比接縫，避免清邊打壞已近無縫圖）
    use_motif = margin_px > 0 and ratio < 0.16 and not discrete

    if use_motif:
        candidates: list[tuple[float, np.ndarray, str]] = []
        _lg("  → 稀疏點綴：先試滿鋪回退…")
        unit_f, detail_f = try_make_dense_seamless(arr, log=log)
        fv, fh = _tile_seam_scores(unit_f)
        candidates.append(
            (
                fv + fh,
                unit_f,
                f"點綴回退滿鋪（前景 {ratio:.0%}）：{detail_f}",
            )
        )
        motifs = extract_interior_motifs(arr, bg, threshold, margin_px)
        cleaned, removed = remove_edge_touching_components(
            arr, bg, threshold, margin_px
        )
        filled = refill_with_wrapped_motifs(
            cleaned, bg, threshold, motifs, removed, seed=42
        )
        edge = _edge_band_mask(h, w, max(margin_px, 8))
        interior = ~edge
        edge_fg = float(np.mean(_foreground_mask(filled, bg, threshold)[edge]))
        int_fg = (
            float(np.mean(_foreground_mask(filled, bg, threshold)[interior]))
            if np.any(interior)
            else 0.0
        )
        too_empty = int_fg > 0.05 and edge_fg < int_fg * 0.25
        worse = structural_edge_score(filled) > structural_edge_score(arr) * 1.02
        if not (too_empty or worse or len(motifs) < 2):
            sv, shs = _tile_seam_scores(filled)
            bv0, bh0 = _tile_seam_scores(arr)
            # 接縫未實質改善則不進候選（避免清邊咬出白齒／假無縫）
            if (sv + shs) > (bv0 + bh0) + 0.5:
                pass
            elif (sv + shs) >= (bv0 + bh0) * 0.98 and (sv + shs) > 8.0:
                pass
            else:
                changed = float((filled != arr).any(axis=2).mean())
                # 改動越大越罰：避免清邊「縫分數好看」卻打散花回
                candidates.append(
                    (
                        sv + shs + changed * 80.0,
                        filled,
                        f"點綴模式（前景 {ratio:.0%}）：清邊後補花",
                    )
                )
        # 原圖也進候選：已接近無縫時勿亂動
        bv, bh = _tile_seam_scores(arr)
        candidates.append(
            (
                bv + bh + 2.0,
                arr.copy(),
                f"點綴保留原圖（前景 {ratio:.0%}）",
            )
        )
        candidates.sort(key=lambda t: t[0])
        _score, best_arr, best_msg = candidates[0]
        return _finalize_unit(best_arr, best_msg)

    _lg("  → 演算法：滿鋪對齊…")
    unit, detail = try_make_dense_seamless(arr, log=log)
    if margin_px == 0 and ratio < 0.18:
        return _finalize_unit(
            unit,
            f"滿鋪對齊（前景 {ratio:.0%}，邊緣帶 0 未清邊）：{detail}",
        )
    return _finalize_unit(
        unit,
        f"滿鋪模式（前景 {ratio:.0%}）：{detail}",
    )
