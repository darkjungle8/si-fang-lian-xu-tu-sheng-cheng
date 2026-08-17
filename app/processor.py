"""四方連續單元圖：候選產生器與 2×2 預覽。

取捨邏輯不在這裡，在 `app.select`；保證無縫的算子在 `app.seamless_core`。
本檔提供的是保真度最高的候選來源——週期裁切、清邊補花——以及前景／
圖種分類這些判斷素材。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from PIL import Image
import cv2

from app.color_io import ColorContext, context_of, restore
from app.color_utils import color_distance, detect_background, to_srgb
from app.discrete_lattice import looks_like_regular_lattice


def _to_rgb_array(image: Image.Image, bg: Sequence[int]) -> np.ndarray:
    """轉為不透明 sRGB；若有透明通道則先合成到背景色。判斷一律看這個。"""
    if image.mode == "RGBA":
        background = Image.new("RGBA", image.size, (*tuple(bg), 255))
        composited = Image.alpha_composite(background, image)
        return np.asarray(composited.convert("RGB"), dtype=np.uint8)
    return np.asarray(to_srgb(image), dtype=np.uint8)


def _native_array(image: Image.Image, bg: Sequence[int]) -> np.ndarray:
    """
    原生色彩通道的陣列。實際的像素搬移一律作用在這上面。

    CMYK 印刷稿若走 CMYK→sRGB→CMYK 來回轉換，實測會產生平均 4–8 階、
    最大 37 階的視覺色偏（sRGB 色域裝不下印刷色域，出界的顏色回不來），
    比接縫修復本身大一個數量級。所以判斷歸判斷，像素要留在原生空間。
    """
    if image.mode == "RGBA":
        background = Image.new("RGBA", image.size, (*tuple(bg), 255))
        return np.asarray(
            Image.alpha_composite(background, image).convert("RGB"),
            dtype=np.uint8,
        )
    if image.mode in ("P", "1", "LA"):
        return np.asarray(image.convert("RGB"), dtype=np.uint8)
    return np.asarray(image, dtype=np.uint8)


def _unit_image(arr: np.ndarray, ctx: ColorContext) -> Image.Image:
    """把原生通道陣列包回 PIL 圖，並掛上原始 profile 與 DPI。"""
    if arr.ndim == 2:
        mode = "L"
    elif arr.shape[2] == 4:
        mode = "CMYK"
    else:
        mode = "RGB"
    img = Image.fromarray(np.ascontiguousarray(arr), mode=mode)
    if ctx.icc is not None and mode == ctx.mode:
        img.info["icc_profile"] = ctx.icc
    if ctx.dpi is not None:
        img.info["dpi"] = ctx.dpi
    return img


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


def _component_to_stamp_from_roi(
    arr: np.ndarray,
    roi_mask: np.ndarray,
    top: int,
    left: int,
    *,
    dilate_px: int = 4,
) -> MotifStamp:
    """
    由「外接矩形內的 mask」建 MotifStamp。

    只在外接矩形上工作。舊版收的是全圖大小的 mask，於是每個圖案都要對整
    張圖做一次 `np.where` 與膨脹——一張 4348² 的圖上有上萬個連通域，光這
    裡就要花掉一分半。
    """
    h, w = arr.shape[:2]
    pad = int(dilate_px) + 1
    y0 = max(0, top - pad)
    x0 = max(0, left - pad)
    y1 = min(h, top + roi_mask.shape[0] + pad)
    x1 = min(w, left + roi_mask.shape[1] + pad)

    local = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    local[
        top - y0 : top - y0 + roi_mask.shape[0],
        left - x0 : left - x0 + roi_mask.shape[1],
    ] = roi_mask
    ys_o, xs_o = np.where(local)
    if dilate_px > 0:
        k = 2 * int(dilate_px) + 1
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        local = cv2.dilate(local, ker)

    ys, xs = np.where(local)
    ry0, ry1 = int(ys.min()), int(ys.max())
    rx0, rx1 = int(xs.min()), int(xs.max())
    mask = local[ry0 : ry1 + 1, rx0 : rx1 + 1].astype(bool)
    src = arr[y0 + ry0 : y0 + ry1 + 1, x0 + rx0 : x0 + rx1 + 1]
    patch = np.zeros((mask.shape[0], mask.shape[1], arr.shape[2]), dtype=np.uint8)
    patch[mask] = src[mask]
    return MotifStamp(
        patch=patch,
        mask=mask,
        area=int(ys_o.size),
        cy=float(np.mean(ys_o) - ry0),
        cx=float(np.mean(xs_o) - rx0),
    )


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
    remain = float(np.mean(fg1))
    orig = float(np.mean(fg0))
    ghost = fg0 & (d1 > threshold * 0.2) & (d1 <= threshold)
    # 只清邊緣殘片、內部圖章仍在：允許較高 erased（恐龍等大圖章碰邊可超過 3%）
    if orig >= 0.08 and remain >= orig * 0.68:
        if erased >= 0.14:
            return True
        if float(np.mean(ghost)) >= 0.04:
            return True
        return False
    if erased >= 0.03:
        return True
    changed = float((filled != src).any(axis=2).mean())
    if changed >= 0.10 and erased >= 0.015:
        return True
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
    """
    取出未碰邊緣帶的完整圖案，作為補花素材。

    碰邊與否用 `labels[edge]` 一次查完，取 mask 只在各自的外接矩形內做。
    逐個圖案跑 `labels == i` 是全圖掃描，圖案上萬個時就是上千億次運算。
    """
    fg = _foreground_mask(arr, bg, threshold)
    edge = _edge_band_mask(*arr.shape[:2], margin_px)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        fg.astype(np.uint8), connectivity=4
    )
    interior = np.ones(n, dtype=bool)
    interior[0] = False
    touching = np.unique(labels[edge])
    interior[touching[touching < n]] = False
    interior &= stats[:, cv2.CC_STAT_AREA] >= min_area

    motifs: list[MotifStamp] = []
    for i in np.flatnonzero(interior):
        x0 = int(stats[i, cv2.CC_STAT_LEFT])
        y0 = int(stats[i, cv2.CC_STAT_TOP])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        roi = (labels[y0 : y0 + bh, x0 : x0 + bw] == i).astype(np.uint8)
        motifs.append(_component_to_stamp_from_roi(arr, roi, y0, x0))
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
    touching = np.unique(labels[edge])
    touching = touching[(touching > 0) & (touching < n)]
    if touching.size == 0:
        return out, removed

    for i in touching:
        # OpenCV centroid 是 (x, y)
        removed.append(
            (
                float(centroids[i][1]),
                float(centroids[i][0]),
                int(stats[i, cv2.CC_STAT_AREA]),
            )
        )
    # 一次做完：膨脹對聯集與逐塊分別做的結果相同，但省掉上萬次全圖掃描
    lut = np.zeros(n, dtype=bool)
    lut[touching] = True
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    kill = cv2.dilate(lut[labels].astype(np.uint8), ker).astype(bool)
    out[kill] = bg_rgb
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
                # 密花假週期：某一軸色差仍高且對邊結構不相關
                # 不可用 icon_grid_likely：花瓣描邊能量高會被誤當成棋盤而跳過
                if not _looks_like_icon_checkerboard(arr):
                    cv_e, ch_e = _edge_profile_corr(tile)
                    if (sv > 36.0 and cv_e < 0.50) or (shs > 36.0 and ch_e < 0.50):
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


def _tile_seam_scores(arr: np.ndarray) -> tuple[float, float]:
    """2×2 中心垂直接縫、水平接縫的平均色差。"""
    h, w = arr.shape[:2]
    # 模擬 tile 接縫：右邊緣 vs 左邊緣、下邊緣 vs 上邊緣
    v = float(np.mean(np.abs(arr[:, -1].astype(np.float64) - arr[:, 0].astype(np.float64))))
    hh = float(np.mean(np.abs(arr[-1].astype(np.float64) - arr[0].astype(np.float64))))
    return v, hh


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


def _edge_profile_corr(arr: np.ndarray) -> tuple[float, float]:
    """左右／上下邊緣亮度剖面相關（1=對齊好）。"""
    lum = arr.astype(np.float64).mean(axis=2)
    def _corr(a: np.ndarray, b: np.ndarray) -> float:
        if a.std() < 1e-6 or b.std() < 1e-6:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    return _corr(lum[:, 0], lum[:, -1]), _corr(lum[0], lum[-1])


def tile_2x2_multi(
    unit: Image.Image,
    *,
    prefer_plain: bool = True,
) -> tuple[Image.Image, str, tuple[int, int]]:
    """
    2×2 純網格預覽，並回報實際量到的接縫。

    這裡以前會做「多圖錯位補白」：偵測對邊錯位後把四格挪開再補上空隙，
    好讓預覽看起來連續。但實際擴圖走的是
    `kuotu.image_pipeline.build_tiled_canvas`，那是逐格 `paste` 的純網格；
    預覽等於在騙人——單元根本沒接上，畫面卻是好的，使用者要到成品出來
    才會發現。單元現在由 `app.select` 保證無縫，預覽就該照實呈現。

    `prefer_plain` 只為相容既有呼叫端保留。
    """
    del prefer_plain
    from app.quality import seam_report

    arr = np.asarray(to_srgb(unit).convert("RGB"), dtype=np.uint8)
    h, w = arr.shape[:2]
    out = np.empty((2 * h, 2 * w, 3), dtype=np.uint8)
    out[:h, :w] = arr
    out[:h, w:] = arr
    out[h:, :w] = arr
    out[h:, w:] = arr
    return (
        Image.fromarray(out, mode="RGB"),
        f"2×2 純拼接（{seam_report(arr).describe()}）",
        (w, h),
    )


def tile_2x2(unit: Image.Image) -> Image.Image:
    """將單元圖拼成 2×2 預覽。"""
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


def _clear_and_refill(
    arr: np.ndarray,
    bg: Sequence[int],
    threshold: float,
    margin_px: int,
) -> np.ndarray | None:
    """
    清掉碰邊殘花，再用完整圖案環繞補回密度。

    只在稀疏點綴上有意義，而且成敗要看補完後畫面有沒有變空——清邊很容易
    把邊上的花整朵刪掉卻補不回來。這裡只做「值不值得當候選」的粗篩，
    真正的取捨交給 `app.select` 的閘門與成本。
    """
    motifs = extract_interior_motifs(arr, bg, threshold, margin_px)
    if len(motifs) < 2:
        return None
    cleaned, removed = remove_edge_touching_components(
        arr, bg, threshold, margin_px
    )
    if not removed:
        return None
    filled = refill_with_wrapped_motifs(
        cleaned, bg, threshold, motifs, removed, seed=42
    )
    if _clear_edge_ruins_motifs(arr, filled, bg, threshold):
        return None
    before = foreground_ratio(arr, bg, threshold)
    after = foreground_ratio(filled, bg, threshold)
    if after < max(0.04, before * 0.55):
        return None
    return filled


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

    這裡只負責產生選項，不做取捨。基底候選都是保真度最高的無損來源
    （原圖、週期裁切、點綴晶格、清邊補花）；把每個基底加工成真正無縫的
    變體、過閘門、比成本，全部交給 `app.select` 以客觀量測決定。

    log：進度回呼（批次時印到命令列；不改變選圖／接縫判定）。
    """
    from app.discrete_lattice import try_make_discrete_seamless
    from app.seamless_core import recover_torus_crop
    from app.select import (
        SEAM_PERFECT,
        Base,
        apply_recipe,
        choose,
        source_facts,
        timed,
    )

    def _lg(msg: str) -> None:
        if log is not None:
            log(msg)

    if bg is None:
        bg = detect_background(image)

    ctx = context_of(image)
    arr = _to_rgb_array(image, bg)
    native = _native_array(image, bg)
    h, w = arr.shape[:2]

    if margin_is_percent:
        margin_px = int(round(min(h, w) * float(margin)))
    else:
        margin_px = int(round(float(margin)))
    margin_px = max(0, margin_px)

    ratio = foreground_ratio(arr, bg, threshold)
    discrete = _looks_like_discrete_motifs(arr, bg, threshold)
    src = source_facts(arr, needs_native=ctx.mode == "CMYK")
    _lg(
        f"  → 分類：{'點綴/晶格' if discrete else '滿鋪或稀疏'}，"
        f"前景 {ratio:.0%}，尺寸 {w}×{h}，{ctx.describe()}，{src.rep.describe()}"
    )

    # 原稿已經夠好時直接採用：它的成本天生最低，不可能被贏過。門檻刻意
    # 訂得比「可見」還嚴，灰帶要進候選競賽讓成本函數權衡，別讓 4.0 那種
    # 平坦大色塊上看得見的小台階從這裡溜走。
    if src.rep.wrap_excess <= SEAM_PERFECT:
        _lg("  → 原稿已無縫，直接採用")
        return (
            _unit_image(native, ctx),
            f"前景 {ratio:.0%}｜原稿已無縫（接縫超出 {src.rep.wrap_excess:.1f}）",
        )

    def _crop_base(cropped: np.ndarray, label: str) -> Base:
        """裁切類候選：找回它在原稿環面上的座標，才能重放到原生通道。"""
        origin = recover_torus_crop(arr, cropped)
        recipe = None
        if origin is not None:
            recipe = [
                ("crop", (*origin, cropped.shape[0], cropped.shape[1])),
            ]
        else:
            _lg(f"  → {label} 無法定位裁切座標，只能走 ICC 轉換")
        return Base(cropped, label, True, recipe)

    bases: list[Base] = [Base(arr, "原圖", True, [])]

    crop = timed("週期裁切搜尋", lambda: try_period_crop(arr, log=log), log)
    if crop is not None and crop[0] is not None:
        bases.append(_crop_base(crop[0], f"週期裁切（{crop[1]}）"))

    if discrete:
        got = timed(
            "點綴晶格",
            lambda: try_make_discrete_seamless(arr, bg, threshold, log=log),
            log,
        )
        if got is not None and got[0] is not None:
            unit_d, detail_d = got
            same = unit_d.shape == arr.shape and np.array_equal(unit_d, arr)
            if not same:
                bases.append(_crop_base(unit_d, f"點綴晶格（{detail_d}）"))

    # 稀疏點綴才清邊補花：滿版花布清邊會把整張花網一起清掉
    if margin_px > 0 and ratio < 0.16 and not discrete:
        filled = timed(
            "清邊補花",
            lambda: _clear_and_refill(arr, bg, threshold, margin_px),
            log,
        )
        if filled is not None:
            bases.append(Base(filled, "清邊補花", False, None))

    best = choose(src, bases, log=log)
    if best.recipe is not None:
        unit = _unit_image(apply_recipe(native, best.recipe), ctx)
    else:
        unit = restore(Image.fromarray(best.arr, mode="RGB"), ctx)
    return unit, f"前景 {ratio:.0%}｜{best.label}"
