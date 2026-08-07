"""四方連續：清掉碰邊殘花後，用完整圖案環繞補回密度（不模糊）。"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from PIL import Image, ImageFilter

from app.color_utils import color_distance, detect_background
from app.discrete_lattice import (
    _cross_seam_cut_count,
    _discrete_integrity_ok,
    _wrap_gap_ratios,
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
    """4-連通前景，回傳各連通域座標列表。"""
    h, w = fg.shape
    visited = np.zeros((h, w), dtype=bool)
    neighbors = ((-1, 0), (1, 0), (0, -1), (0, 1))
    components: list[list[tuple[int, int]]] = []

    ys, xs = np.where(fg)
    for y0, x0 in zip(ys.tolist(), xs.tolist()):
        if visited[y0, x0]:
            continue
        comp: list[tuple[int, int]] = []
        q: deque[tuple[int, int]] = deque([(y0, x0)])
        visited[y0, x0] = True
        while q:
            cy, cx = q.popleft()
            comp.append((cy, cx))
            for dy, dx in neighbors:
                ny, nx = cy + dy, cx + dx
                if (
                    0 <= ny < h
                    and 0 <= nx < w
                    and fg[ny, nx]
                    and not visited[ny, nx]
                ):
                    visited[ny, nx] = True
                    q.append((ny, nx))
        components.append(comp)
    return components


def _component_to_stamp(
    arr: np.ndarray,
    coords: list[tuple[int, int]],
) -> MotifStamp:
    ys = [c[0] for c in coords]
    xs = [c[1] for c in coords]
    y0, y1 = min(ys), max(ys)
    x0, x1 = min(xs), max(xs)
    hm, wm = y1 - y0 + 1, x1 - x0 + 1
    patch = np.zeros((hm, wm, 3), dtype=np.uint8)
    mask = np.zeros((hm, wm), dtype=bool)
    for y, x in coords:
        mask[y - y0, x - x0] = True
        patch[y - y0, x - x0] = arr[y, x]
    area = len(coords)
    cy = float(np.mean(ys) - y0)
    cx = float(np.mean(xs) - x0)
    return MotifStamp(patch=patch, mask=mask, area=area, cy=cy, cx=cx)


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
    motifs: list[MotifStamp] = []
    for coords in _iter_components(fg):
        if len(coords) < min_area:
            continue
        touches = any(edge[y, x] for y, x in coords)
        if touches:
            continue
        motifs.append(_component_to_stamp(arr, coords))
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

    for coords in _iter_components(fg):
        if not any(edge[y, x] for y, x in coords):
            continue
        ys = [c[0] for c in coords]
        xs = [c[1] for c in coords]
        removed.append((float(np.mean(ys)), float(np.mean(xs)), len(coords)))
        for y, x in coords:
            out[y, x] = bg_rgb
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
    # 最多補 removed 的 70%，且不超過素材數*2
    budget = min(len(ordered), max(1, int(round(len(ordered) * 0.7))), max(2, len(motifs) * 2))
    for cy, cx, area in ordered[:budget]:
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
                ny, nx = _nudge_off_corners(ny, nx, h, w)
                if any(
                    _torus_distance(ny, nx, py, px, h, w) < max(radius, pr) * 0.85
                    for py, px, pr in placed_centers
                ):
                    continue
                if _overlap_ratio(out, motif, ny, nx, occupied) <= max_overlap:
                    cy, cx = ny, nx
                    found = True
                    break
            if not found:
                continue  # 放棄，不硬塞
        else:
            # 檢查重疊；不行就抖動，再不放
            ok = _overlap_ratio(out, motif, cy, cx, occupied) <= max_overlap
            if not ok:
                found = False
                for _ in range(16):
                    ny = float((cy + rng.normal(0, radius * 0.5)) % h)
                    nx = float((cx + rng.normal(0, radius * 0.5)) % w)
                    ny, nx = _nudge_off_corners(ny, nx, h, w)
                    if any(
                        _torus_distance(ny, nx, py, px, h, w) < max(radius, pr) * 0.85
                        for py, px, pr in placed_centers
                    ):
                        continue
                    if _overlap_ratio(out, motif, ny, nx, occupied) <= max_overlap:
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
        a = strip.astype(np.float64)
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
) -> tuple[np.ndarray, tuple[int, int], float]:
    h, w = arr.shape[:2]
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
    # 全週期粗搜（棋盤／網格常需大相位），再局部精修
    stepx = max(1, px // 10)
    stepy = max(1, py // 10)
    for x in range(0, min(px, w - cw + 1), stepx):
        for y in range(0, min(py, h - ch + 1), stepy):
            k = _key(arr[y : y + ch, x : x + cw])
            if k < best_key:
                best_key, bxy = k, (x, y)
    x0, y0 = bxy
    for dx in range(-8, 9):
        for dy in range(-8, 9):
            x = min(max(0, x0 + dx), w - cw)
            y = min(max(0, y0 + dy), h - ch)
            k = _key(arr[y : y + ch, x : x + cw])
            if k < best_key:
                best_key, bxy = k, (x, y)
    x, y = bxy
    return arr[y : y + ch, x : x + cw].copy(), (x, y), best_key[2]


def try_period_crop(
    arr: np.ndarray,
    bg: Sequence[int] | None = None,
) -> tuple[np.ndarray | None, str]:
    """
    滿鋪幾何／斜紋／魚鱗：用亮度+梯度找獨立 xy 週期，再裁成整數倍並搜相位。
    以 structural_edge_score + 接縫色差驗證，沒改善則失敗。
    """
    del bg  # 不再依賴背景色二值化（滿鋪時易誤判）
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
    full_gray = _luminance_map(arr)
    full_x: list[int] = []
    full_y: list[int] = []
    for p, score in _autocorr_best_periods(
        full_gray.mean(1), max(16, h // 40), h // 2, top_k=5
    ):
        if score >= 0.25 and h * 0.12 <= p <= h * 0.48:
            full_y.extend([int(p), int(round(p / 2))])
    for p, score in _autocorr_best_periods(
        full_gray.mean(0), max(16, w // 40), w // 2, top_k=5
    ):
        if score >= 0.25 and w * 0.12 <= p <= w * 0.48:
            full_x.extend([int(p), int(round(p / 2))])
    # 強峰置頂，再接縮圖候選
    xs = list(dict.fromkeys([*full_x, *xs]))
    ys = list(dict.fromkeys([*full_y, *ys]))
    xs = [p for p in xs if 16 <= p <= w // 2][:20]
    ys = [p for p in ys if 16 <= p <= h // 2][:20]

    best_tile: np.ndarray | None = None
    best_rank = (base + base_seam * 0.35, base_seam, base)
    best_detail = ""

    for px in xs[:12]:
        for py in ys[:12]:
            # 試最大與次大整數倍：剛好整除時 max 倍無法做相位偏移
            nmax = w // px
            mmax = h // py
            size_opts: list[tuple[int, int]] = []
            for dn in (0, 1):
                for dm in (0, 1):
                    cw = (nmax - dn) * px
                    ch = (mmax - dm) * py
                    if cw < int(w * 0.72) or ch < int(h * 0.72):
                        continue
                    if cw < max(320, w // 2) or ch < max(320, h // 2):
                        continue
                    size_opts.append((cw, ch))
            seen_sz: set[tuple[int, int]] = set()
            for cw, ch in size_opts:
                if (cw, ch) in seen_sz:
                    continue
                seen_sz.add((cw, ch))
                tile, off, sc = _best_phase_for_size(arr, cw, ch, px, py)
                sv, shs = _tile_seam_scores(tile)
                seam = sv + shs
                rank = (sc + seam * 0.35, seam, sc)
                if rank < best_rank and (
                    sc + 0.15 < base
                    or seam < base_seam * 0.85
                    or seam < base_seam - 10
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
    a = arr.astype(np.float64)
    h, w = a.shape[:2]
    band = max(4, min(band, h // 8, w // 8))
    lim = min(lim, h // 3, w // 3)

    left, right = a[:, :band], a[:, -band:]
    best_v = (1e9, 0)
    for d in range(-lim, lim + 1):
        sc = float(np.mean(np.abs(left - np.roll(right, d, axis=0))))
        if sc < best_v[0]:
            best_v = (sc, d)

    top, bottom = a[:band], a[-band:]
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
) -> tuple[np.ndarray, int]:
    """
    沿 axis 半幅偏移後，在中央重疊帶做 min-cut 硬切。
    不滾動半幅本體，以保留外緣=原圖中心相鄰像素（四方連續保證）。
    """
    del lim  # 外緣連續優先，不再用半幅錯位
    h, w = arr.shape[:2]
    if axis == 1:
        mid = w // 2
        out = np.roll(arr, mid, axis=1)
        left_ov = out[:, mid - ov : mid]
        right_ov = out[:, mid : mid + ov]
        err = np.mean(
            np.abs(left_ov.astype(np.float64) - right_ov.astype(np.float64)),
            axis=2,
        )
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
    err = np.mean(
        np.abs(top_ov.astype(np.float64) - bot_ov.astype(np.float64)),
        axis=2,
    ).T
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
    ov = max(80, min(h, w) // 10)

    # 純半幅偏移：外緣=原圖中心相鄰像素，四方連續有數學保證
    pure = np.roll(np.roll(arr, w // 2, axis=1), h // 2, axis=0)

    cur, _ = _offset_mincut_axis(arr, axis=1, ov=ov, lim=0)
    cur, _ = _offset_mincut_axis(cur, axis=0, ov=ov, lim=0)
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
    return (
        cur,
        f"半幅偏移拼接（接縫 {base_v:.0f}+{base_h:.0f}→{v:.0f}+{hs:.0f}，"
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
    candidates: list[tuple[float, np.ndarray, str]] = []
    for fn in (offset_quilt_seamless, shear_align_seamless):
        out, msg = fn(arr)
        candidates.append((_seam_rank(out), out, msg))
    candidates.sort(key=lambda t: t[0])
    return candidates[0][1], candidates[0][2]


def try_make_dense_seamless(arr: np.ndarray) -> tuple[np.ndarray, str]:
    """
    滿鋪紋：單元圖優先保留原圖（或穩定週期裁切）。
    斜紋對齊改在 2×2 用多圖錯位拼接完成，不在單張上半幅偏移。
    """
    base = structural_edge_score(arr)
    base_v, base_h = _tile_seam_scores(arr)
    base_seam = base_v + base_h

    cropped, detail = try_period_crop(arr)
    if cropped is not None:
        csc = structural_edge_score(cropped)
        cv, ch = _tile_seam_scores(cropped)
        crop_seam = cv + ch
        # 舊門檻 0.45 過嚴：結構降到 0.6× 或接縫總分明顯下降都應採用
        better_struct = csc < base * 0.85
        better_seam = crop_seam < base_seam * 0.80 and crop_seam < base_seam - 8.0
        if better_struct or better_seam:
            return (
                cropped.copy(),
                f"週期裁切 {detail}（接縫 {base_v:.0f}+{base_h:.0f}→{cv:.0f}+{ch:.0f}；"
                "2×2 再多圖錯位拼接）",
            )

    return (
        arr.copy(),
        f"保留原圖（接縫 {base_v:.0f}+{base_h:.0f}；2×2 用多圖錯位拼接對齊）",
    )


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

    # 邊緣剖面相關高 → 單元已對齊，規則點綴勿再錯位
    cv, ch = _edge_profile_corr(arr)
    if prefer_plain or (cv >= 0.85 and ch >= 0.85):
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

    if pick is None or (pick[3] + pick[4]) >= (plain_v + plain_h) * 0.90:
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
    # 垂直／水平條紋：某一軸投影幾乎恆定 → 走滿鋪
    row_p = fg.mean(axis=1)
    col_p = fg.mean(axis=0)
    if float(row_p.std()) < 0.02 and float(col_p.std()) > 0.08:
        return False
    if float(col_p.std()) < 0.02 and float(row_p.std()) > 0.08:
        return False
    comps = _iter_components(fg)
    if len(comps) < 8:
        return False
    areas = sorted((len(c) for c in comps), reverse=True)
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
) -> tuple[Image.Image, str]:
    """
    產生四方連續單元圖，回傳 (圖, 模式說明)。

    - 規則點綴（星／花／愛心等）：質心晶格構造裁切，跨縫完整性硬門禁
    - 滿鋪：週期裁切（與邊緣帶無關）
    - 稀疏不規則點綴：清碰邊殘花 + 環繞補花（需邊緣帶 > 0，且非規則陣列）
    """
    from app.discrete_lattice import try_make_discrete_seamless

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

    # 規則點綴：構造裁切；禁止清邊打散陣列、禁止邊緣相關備選勝出
    if discrete:
        from app.discrete_lattice import (
            _fg_mask_merged as _d_fg,
            _interior_median_motif_area,
            _rank_tile,
            wrap_gap_ratios,
        )

        unit_d, detail_d = try_make_discrete_seamless(arr, bg, threshold)
        if "FAIL" not in detail_d:
            return (
                Image.fromarray(unit_d, mode="RGB"),
                f"點綴（前景 {ratio:.0%}）：{detail_d}",
            )
        # 晶格未硬通過：用「跨縫完整性／間距」選備選，禁止只靠接縫色差
        # （錯誤週期裁切常把色差修好，卻把花距拉成 2 倍，看起來很空）
        med = _interior_median_motif_area(_d_fg(arr, bg, threshold))

        def _integrity_key(tile: np.ndarray) -> tuple[int, float]:
            tier, sc = _rank_tile(tile, bg, threshold, med)
            gx, gy = wrap_gap_ratios(tile, bg, threshold, med=med)
            gap_pen = (abs(gx - 1.0) + abs(gy - 1.0)) * 60.0
            return (tier, sc + gap_pen)

        candidates: list[tuple[tuple[int, float], np.ndarray, str]] = []
        key_d = _integrity_key(unit_d)
        # FAIL 結果不得靠「分數碰巧好看」壓過滿鋪／原圖
        if "FAIL" in detail_d:
            key_d = (max(key_d[0], 2) + 1, key_d[1] + 120.0)
        candidates.append(
            (
                key_d,
                unit_d,
                f"點綴最佳嘗試（前景 {ratio:.0%}）：{detail_d}",
            )
        )
        candidates.append(
            (
                _integrity_key(arr),
                arr.copy(),
                f"點綴保留原圖（前景 {ratio:.0%}，晶格未對齊）",
            )
        )
        unit_f, detail_f = try_make_dense_seamless(arr)
        gx0, gy0 = wrap_gap_ratios(arr, bg, threshold, med=med)
        gxf, gyf = wrap_gap_ratios(unit_f, bg, threshold, med=med)
        gap0 = abs(gx0 - 1.0) + abs(gy0 - 1.0)
        gapf = abs(gxf - 1.0) + abs(gyf - 1.0)
        dense_key = _integrity_key(unit_f)
        # 間距明顯變差（例如≈2 倍空檔）→ 降級；但若接縫色差明顯更好仍保留輕罰
        if gapf > gap0 + 0.25 or max(gxf, gyf) >= 1.6:
            sv0, sh0 = _tile_seam_scores(arr)
            svf, shf = _tile_seam_scores(unit_f)
            if (svf + shf) < (sv0 + sh0) * 0.75:
                dense_key = (dense_key[0], dense_key[1] + 40.0)
            else:
                dense_key = (dense_key[0] + 1, dense_key[1] + 200.0)
        candidates.append(
            (
                dense_key,
                unit_f,
                f"滿鋪回退（點綴未對齊，前景 {ratio:.0%}）：{detail_f}",
            )
        )
        if margin_px > 0 and ratio < 0.45:
            motifs = extract_interior_motifs(arr, bg, threshold, margin_px)
            cleaned, removed = remove_edge_touching_components(
                arr, bg, threshold, margin_px
            )
            if len(motifs) >= 2 and len(removed) > 0:
                filled = refill_with_wrapped_motifs(
                    cleaned, bg, threshold, motifs, removed, seed=42
                )
                worse = (
                    structural_edge_score(filled)
                    > structural_edge_score(arr) * 1.08
                )
                if not worse:
                    candidates.append(
                        (
                            _integrity_key(filled),
                            filled,
                            f"點綴未對齊改清邊補花（前景 {ratio:.0%}）",
                        )
                    )
        candidates.sort(key=lambda t: t[0])
        _score, best_arr, best_msg = candidates[0]
        return Image.fromarray(best_arr, mode="RGB"), best_msg

    # 稀疏不規則才清邊補花（與滿鋪週期裁切比接縫，避免清邊打壞已近無縫圖）
    use_motif = margin_px > 0 and ratio < 0.16 and not discrete

    if use_motif:
        candidates: list[tuple[float, np.ndarray, str]] = []
        unit_f, detail_f = try_make_dense_seamless(arr)
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
        return Image.fromarray(best_arr, mode="RGB"), best_msg

    unit, detail = try_make_dense_seamless(arr)
    if margin_px == 0 and ratio < 0.18:
        return (
            Image.fromarray(unit, mode="RGB"),
            f"滿鋪對齊（前景 {ratio:.0%}，邊緣帶 0 未清邊）：{detail}",
        )
    return (
        Image.fromarray(unit, mode="RGB"),
        f"滿鋪模式（前景 {ratio:.0%}）：{detail}",
    )
