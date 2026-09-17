"""四方連續單元圖：候選產生器與 2×2 預覽。

取捨邏輯不在這裡，在 `app.select`；保證無縫的算子在 `app.seamless_core`。
本檔提供的是保真度最高的候選來源——週期裁切、清邊補花——以及前景／
圖種分類這些判斷素材。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Callable, Sequence

import numpy as np
from PIL import Image
import cv2

from app.color_io import ColorContext, context_of, restore
from app.color_utils import color_distance, detect_background, to_srgb
from app.discrete_lattice import looks_like_regular_lattice
from app.motif_layout import (
    attach_group_colors,
    background_plate,
    fill_kill_with_plate,
    group_components_into_motifs,
    group_stamp_from_roi,
    layout_stats_from_groups,
    relayout_ring,
    _place_along_wrap,
)


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


def _join_foreground(arr: np.ndarray, bg: Sequence[int], threshold: float) -> np.ndarray:
    """
    連通域用的前景。細線雪花／冰裂若走 4 連通，一條 1px 斜線會碎成上百塊，
    碰邊只刪掉貼框的殘片、內部的臂還留在邊緣——清邊補花疊上去 wrap 熱點
    照樣爆。先 3×3 閉合再 8 連通，把髮絲缺口接回同一朵。
    """
    fg = _foreground_mask(arr, bg, threshold).astype(np.uint8)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(fg, cv2.MORPH_CLOSE, ker)


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


def _distance_p90(fg: np.ndarray) -> float:
    dist = cv2.distanceTransform(fg.astype(np.uint8), cv2.DIST_L2, 5)
    on = dist[fg.astype(bool)]
    if on.size == 0:
        return 0.0
    return float(np.percentile(on, 90))


def _overlay_stamp_mask(
    arr: np.ndarray,
    bg: Sequence[int],
    threshold: float,
) -> np.ndarray | None:
    """
    格紋／滿版底紋上的點綴（蜜蜂疊在灰格上）用色度拆出來。

    一般前景對地色在這種圖會把整張格紋當墨，清邊補花刪掉 90% 畫面。
    灰底色度低、點綴（黃蜜蜂）色度高，兩者差一個數量級才採信。
    """
    fg = _foreground_mask(arr, bg, threshold)
    if float(np.mean(fg)) < 0.50:
        return _sparkle_overlay_mask(arr, bg, threshold)
    rgb = arr.astype(np.float32)
    chroma = rgb.max(axis=2) - rgb.min(axis=2)
    seed = chroma > 40.0
    frac = float(np.mean(seed))
    if frac < 0.004 or frac > 0.25:
        return _sparkle_overlay_mask(arr, bg, threshold)
    if float(chroma[~seed].mean()) > 12.0:
        return _sparkle_overlay_mask(arr, bg, threshold)
    seed_u8 = seed.astype(np.uint8)
    # 只閉合點綴本體。9×9 膨脹會把格紋格子算進 overlay，結構閘門就變成切格。
    ker3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    overlay = cv2.morphologyEx(seed_u8, cv2.MORPH_CLOSE, ker3)
    halo = cv2.dilate(overlay, ker3)
    overlay = (halo.astype(bool) & (chroma > 28.0)).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(overlay, connectivity=8)
    mid = 0
    cap = overlay.size * 0.05
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if 80 <= area <= cap:
            mid += 1
    if mid < 8:
        return _sparkle_overlay_mask(arr, bg, threshold)
    return overlay


def _sparkle_overlay_mask(
    arr: np.ndarray,
    bg: Sequence[int],
    threshold: float,
) -> np.ndarray | None:
    """水彩／雲霧底上的星、雪花：局部比周圍亮的小圖章。

    色度拆點綴會把整片水彩當 overlay（chroma>40 可到 80%+）。星星是
    高通殘差，不是高飽和。格紋／稀疏圖章不要走這條：它們已有週期裁切
    或一般 stamp mask。
    """
    fg = _foreground_mask(arr, bg, threshold)
    fg_frac = float(np.mean(fg))
    if fg_frac < 0.35:
        return None
    gray = _luminance_map(arr)
    h, w = gray.shape
    sigma = max(4.0, min(h, w) / 80.0)
    blur = cv2.GaussianBlur(gray, (0, 0), sigma)
    seed = (gray > blur + 18.0).astype(np.uint8)
    frac = float(np.mean(seed))
    if frac < 0.004 or frac > 0.16:
        return None
    if frac > fg_frac * 0.55:
        return None
    ker3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    overlay = cv2.morphologyEx(seed, cv2.MORPH_CLOSE, ker3)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(overlay, connectivity=8)
    keep = np.zeros(n, dtype=bool)
    cap = overlay.size * 0.05
    mid = 0
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if 80 <= area <= cap:
            keep[i] = True
            mid += 1
    if mid < 12:
        return None
    return keep[labels].astype(np.uint8)


def stamp_structure_view(
    arr: np.ndarray,
    bg: Sequence[int] | None = None,
    threshold: float = 40.0,
) -> np.ndarray:
    """
    格紋點綴：結構閘門只看點綴圖章，不看底紋相位。

    蜜蜂疊在灰格上時，wrap 熱點／切圖都被格紋差幾像素的週期誤差主導，
    即使蜜蜂已經跨縫貼完整。把點綴單獨放在平坦底上再量。
    """
    if bg is None:
        # 全圖中位數會被 30%+ 碎花拉成花色，切圖／熱點就變成假陽性。
        bg = detect_background(Image.fromarray(arr))
    ov = _overlay_stamp_mask(arr, bg, threshold)
    if ov is None:
        fg = _stamp_foreground(arr, bg, threshold).astype(bool)
        out = np.empty_like(arr)
        out[...] = np.array(bg, dtype=np.uint8)
        if fg.any():
            out[fg] = arr[fg]
        return out
    out = np.empty_like(arr)
    out[...] = np.array(bg, dtype=np.uint8)
    m = ov.astype(bool)
    if m.any():
        out[m] = arr[m]
    return out


def _stamp_flat_view(
    arr: np.ndarray, bg: Sequence[int], threshold: float
) -> np.ndarray:
    return stamp_structure_view(arr, bg, threshold)


def _count_big_components(mask: np.ndarray, min_area: int = 200) -> int:
    n, _, st, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if n <= 1:
        return 0
    return int(np.sum(st[1:, cv2.CC_STAT_AREA] >= min_area))


def _split_touching_stamps(fg: np.ndarray) -> np.ndarray:
    """
    葉脈／細莖把多朵花黏成一塊時，用開運算拆開。

    只在距離場中等（約 7–50px）且拆完中等連通域明顯變多時才用。
    實心動物（刺蝟 p90≈90）與細線雪花（p90<6）都跳過。
    """
    p90 = _distance_p90(fg)
    if not (7.0 <= p90 <= 50.0):
        return fg
    src = fg.astype(np.uint8)
    big0 = _count_big_components(src)
    if big0 < 4:
        return src
    best = src
    best_big = big0
    for k in (3, 5, 7):
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        op = cv2.morphologyEx(src, cv2.MORPH_OPEN, ker)
        big = _count_big_components(op)
        if big > best_big:
            best, best_big = op, big
    if best_big >= big0 + max(4, int(round(big0 * 0.12))):
        return best
    return src


def _stamp_foreground(
    arr: np.ndarray,
    bg: Sequence[int],
    threshold: float,
) -> np.ndarray:
    """
    清邊補花用的圖章 mask。

    細線雪花要 3×3 閉合才不會碎臂；實心動物閉合會把鼻尖黏上下一隻。
    距離場厚的當實心，改用未閉合前景。格紋點綴走色度殘差。
    散花葉脈黏連時用開運算拆成單朵，才能只清碰框的那一截。
    """
    overlay = _overlay_stamp_mask(arr, bg, threshold)
    if overlay is not None:
        return overlay
    joined = _join_foreground(arr, bg, threshold)
    if _distance_p90(joined) >= 8.0:
        fg = _foreground_mask(arr, bg, threshold).astype(np.uint8)
    else:
        fg = joined
    return _split_touching_stamps(fg)


def _touch_margin_px(fg: np.ndarray) -> int:
    """
    實心圖章只清真正碰到畫框的殘片（2px）。

    舊的 1.2% 帶會把腳剛擦到邊的完整刺蝟整隻刪掉，質心又在畫面裡，
    補回去不跨縫，2×2 正中央仍是半截。細線幾何仍用較寬的帶抓住差
    6～12px 的臂。
    """
    h, w = fg.shape[:2]
    if _distance_p90(fg) < 6.0:
        return max(4, min(16, min(h, w) // 80))
    return 2


def _crowd_edge_sep(
    stats: np.ndarray,
    centroids: np.ndarray,
    n: int,
    h: int,
    w: int,
    min_area: int,
    skip_ids: set[int] | None = None,
) -> float:
    """
    離框不到約半個內部間距的圖章，2×2 會跟對邊那一列擠成雙行。
    回傳應視為邊緣件的質心距離。
    """
    skip = skip_ids or set()
    pts: list[tuple[float, float]] = []
    for i in range(1, n):
        if i in skip:
            continue
        if int(stats[i, cv2.CC_STAT_AREA]) < min_area:
            continue
        cy = float(centroids[i][1])
        cx = float(centroids[i][0])
        if min(cx, w - cx, cy, h - cy) <= 2.0:
            continue
        pts.append((cy, cx))
    if len(pts) < 2:
        return 0.0
    dists: list[float] = []
    for i, (y, x) in enumerate(pts):
        best = 1e18
        for j, (y2, x2) in enumerate(pts):
            if i == j:
                continue
            best = min(best, _torus_distance(y, x, y2, x2, h, w))
        dists.append(best)
    nn = float(np.median(np.asarray(dists, dtype=np.float64)))
    if nn < 16.0:
        return 0.0
    return float(min(0.38 * nn, 0.08 * min(h, w)))


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


def _hero_typical_area(motifs: list[MotifStamp]) -> float:
    """主圖章面積。碎點很多時改看較大那一檔，避免中位數落在雪點上。"""
    areas = np.array([m.area for m in motifs], dtype=np.float64)
    if areas.size == 0:
        return 40.0
    med = float(np.median(areas))
    p90 = float(np.percentile(areas, 90))
    if p90 > med * 6.0:
        upper = areas[areas >= p90 * 0.45]
        return float(np.median(upper if upper.size else areas))
    cut = float(np.percentile(areas, 70))
    big = areas[areas >= cut]
    return float(np.median(big if big.size else areas))


def _complete_templates(motifs: list[MotifStamp]) -> list[MotifStamp]:
    """丟掉黏成一坨的超大塊、以及相對同伴明顯殘缺的模板。"""
    if len(motifs) < 2:
        return motifs
    from app.quality import _component_solidity

    hero = _hero_typical_area(motifs)
    main = [m for m in motifs if m.area >= max(200, hero * 0.35)]
    if len(main) >= 2:
        motifs = main
        hero = _hero_typical_area(motifs)
    recs: list[tuple[MotifStamp, float | None, int]] = []
    for m in motifs:
        if m.area > hero * 5.0:
            continue
        recs.append((m, _component_solidity(m.mask), m.area))
    if len(recs) < 2:
        return [m for m in motifs if m.area <= hero * 5.0] or motifs
    areas = np.array([a for _, _, a in recs], dtype=np.float64)
    sols = [s for _, s, _ in recs]
    kept: list[MotifStamp] = []
    for m, s, area in recs:
        if s is None:
            kept.append(m)
            continue
        peer = (areas >= area / 1.55) & (areas <= area * 1.55)
        peer_s = [sols[j] for j in range(len(recs)) if peer[j] and sols[j] is not None]
        if len(peer_s) < 4:
            kept.append(m)
            continue
        med_s = float(np.median(peer_s))
        if s < med_s - 0.10 and s < 0.78:
            continue
        kept.append(m)
    if len(kept) >= 2:
        return kept
    return [m for m, _, _ in recs]


def _axes_perpendicular(a: str, b: str) -> bool:
    lr, tb = {"l", "r"}, {"t", "b"}
    return (a in lr and b in tb) or (a in tb and b in lr)


def _apply_wrap_axis(cy: float, cx: float, axis: str, h: int, w: int) -> tuple[float, float]:
    if axis == "l":
        return cy, 0.0
    if axis == "r":
        return cy, float(w)
    if axis == "t":
        return 0.0, cx
    return float(h), cx


def _snap_wrap_placement(
    cy: float,
    cx: float,
    area: int,
    motif: MotifStamp,
    h: int,
    w: int,
    *,
    forced_edge: bool = False,
) -> tuple[float, float, bool]:
    """
    半截殘片把完整圖章的中心吸到 wrap 線上，貼上去才會跨縫出現在對邊。
    內部殘缺維持原質心。回傳 (cy, cx, 是否邊緣件)。

    碰框刪除的件（forced_edge）質心常落在畫面裡（半隻蜜蜂／雪人往內
    縮），帶寬不夠就不會跨縫。這種一律吸到最近的 wrap。
    只吸最近的那一條邊：雪花這種大圖章帶寬可到短邊 40%，左右殘片若再
    被獨立吸到頂底，會全部堆到四角，2×2 看起來像假接。
    """
    mh, mw = motif.mask.shape[:2]
    is_edge = False
    if forced_edge or area < motif.area * 0.92:
        span = max(mw, mh)
        band = min(
            span * (0.90 if forced_edge else 0.55),
            (0.40 if forced_edge else 0.18) * min(h, w),
        )
        opts = sorted(
            (
                (float(cx), "l"),
                (float(w) - float(cx), "r"),
                (float(cy), "t"),
                (float(h) - float(cy), "b"),
            )
        )
        if forced_edge or opts[0][0] <= band:
            is_edge = True
            cy, cx = _apply_wrap_axis(cy, cx, opts[0][1], h, w)
            corner_lim = max(12.0, 0.12 * span)
            if (
                opts[1][0] <= corner_lim
                and _axes_perpendicular(opts[0][1], opts[1][1])
            ):
                cy, cx = _apply_wrap_axis(cy, cx, opts[1][1], h, w)
    else:
        edge_px = max(8.0, 0.04 * min(h, w))
        is_edge = (
            cx <= edge_px
            or cx >= w - edge_px
            or cy <= edge_px
            or cy >= h - edge_px
        )
        if not is_edge:
            cy, cx = _nudge_off_corners(cy, cx, h, w)
    return cy % h, cx % w, is_edge


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
    # 背景版填洞後，紙紋相對偵測到的平色地色常落在「淡墨水」區間。
    # 那不是淡鬼影（鬼影是原圖章幾乎沒被換掉）。要 dist(filled, src) 仍小
    # 才算留下淡印。
    still_src = np.sqrt(
        np.sum(
            (src.astype(np.float32) - filled.astype(np.float32)) ** 2,
            axis=-1,
        )
    ) < (threshold * 0.55)
    ghost = fg0 & (d1 > threshold * 0.2) & (d1 <= threshold) & still_src
    # 只清邊緣殘片、內部圖章仍在：允許較高 erased。
    # 貓頭鷹這種半幅大圖章，光刪掉碰框的半隻就超過畫面 14%，0.14 會把
    # 已經跨縫補好的結果當成毀圖。
    if orig >= 0.08 and remain >= orig * 0.48:
        if erased >= 0.32:
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


def _overlay_lattice_period(arr: np.ndarray, overlay: np.ndarray) -> tuple[int, int]:
    """從非點綴像素估格紋週期。分數太低就當作沒有穩週期。"""
    h, w = arr.shape[:2]
    lum = arr.astype(np.float64).mean(axis=2)
    masked = np.where(overlay.astype(bool), np.nan, lum)
    sigx = np.nanmean(masked, axis=0)
    sigy = np.nanmean(masked, axis=1)
    mx = float(np.nanmean(sigx))
    my = float(np.nanmean(sigy))
    sigx = np.where(np.isnan(sigx), mx, sigx)
    sigy = np.where(np.isnan(sigy), my, sigy)
    xs = _autocorr_best_periods(sigx, 8, max(16, w // 3), top_k=1)
    ys = _autocorr_best_periods(sigy, 8, max(16, h // 3), top_k=1)
    if not xs or not ys or xs[0][1] < 0.20 or ys[0][1] < 0.20:
        return 0, 0
    px, py = int(xs[0][0]), int(ys[0][0])
    if px < 8 or py < 8:
        return 0, 0
    return px, py


def _fill_overlay_with_lattice(arr: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    """
    把點綴全部換成格紋像素（優先從整數週期位移拷貝原稿）。

    清邊補花只該搬蜜蜂，底紋要留下可拼接的格。median 塗洞會留下一坨
    不在週期上的灰斑；從 ±k 個格子拷貝才是原稿像素。
    """
    out = arr.copy()
    ov = overlay.astype(bool)
    if not ov.any():
        return out
    px, py = _overlay_lattice_period(arr, overlay)
    h, w = arr.shape[:2]
    ys, xs = np.where(ov)
    still = np.ones(ys.shape[0], dtype=bool)
    if px >= 8 and py >= 8:
        for k in (1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6):
            if not still.any():
                break
            yy = (ys + k * py) % h
            xx = (xs + k * px) % w
            ok = still & ~ov[yy, xx]
            if not ok.any():
                continue
            out[ys[ok], xs[ok]] = arr[yy[ok], xx[ok]]
            still[ok] = False
    if still.any():
        med = cv2.medianBlur(arr, 21)
        out[ys[still], xs[still]] = med[ys[still], xs[still]]
    return out


def _erase_touching_overlay(
    arr: np.ndarray,
    overlay: np.ndarray,
    margin_px: int,
    min_area: int,
) -> tuple[np.ndarray, np.ndarray, list[tuple[float, float, int, bool]]]:
    """只清碰到畫框的點綴，水彩／內部星留下。"""
    h, w = overlay.shape[:2]
    edge = _edge_band_mask(h, w, max(2, int(margin_px)))
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(
        overlay.astype(np.uint8), connectivity=8
    )
    kill = np.zeros(n, dtype=bool)
    jobs: list[tuple[float, float, int, bool]] = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x0 = int(stats[i, cv2.CC_STAT_LEFT])
        y0 = int(stats[i, cv2.CC_STAT_TOP])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        roi = labels[y0 : y0 + bh, x0 : x0 + bw] == i
        touching = bool(edge[y0 : y0 + bh, x0 : x0 + bw][roi].any())
        jobs.append(
            (float(centroids[i][1]), float(centroids[i][0]), area, touching)
        )
        if touching:
            kill[i] = True
    mask = kill[labels]
    out = arr.copy()
    if mask.any():
        med = cv2.medianBlur(arr, 21)
        out[mask] = med[mask]
    occ = overlay.astype(bool) & ~mask
    return out, occ, jobs


def _stamp_component_jobs(
    fg: np.ndarray,
    margin_px: int,
    min_area: int,
) -> list[tuple[float, float, int, bool]]:
    """每個夠大的圖章群組：(質心 y, x, 面積, 是否碰框／近框)。"""
    h, w = fg.shape[:2]
    edge = _edge_band_mask(h, w, margin_px)
    _gmap, groups, _labels, stats, centroids = group_components_into_motifs(
        fg, min_area, edge=edge
    )
    if not groups:
        return []
    n = int(stats.shape[0])
    touch_ids = {i for g in groups if g.touching for i in g.ids}
    sep = _crowd_edge_sep(stats, centroids, n, h, w, min_area, touch_ids)
    jobs: list[tuple[float, float, int, bool]] = []
    for g in groups:
        near = sep >= 8.0 and min(g.cx, w - g.cx, g.cy, h - g.cy) <= sep
        jobs.append((g.cy, g.cx, g.area, g.touching or near))
    return jobs


def extract_interior_motifs(
    arr: np.ndarray,
    bg: Sequence[int],
    threshold: float,
    margin_px: int,
    min_area: int = 40,
) -> list[MotifStamp]:
    """
    取出未碰邊緣帶的完整圖案，作為補花素材。

    鄰近連通塊（掌墊＋趾）合成一枚圖章，避免補回去只有半隻腳印。
    """
    fg = _stamp_foreground(arr, bg, threshold)
    edge = _edge_band_mask(*arr.shape[:2], margin_px)
    gmap, groups, _labels, _stats, _cents = group_components_into_motifs(
        fg, min_area, edge=edge
    )
    motifs: list[MotifStamp] = []
    for gid, g in enumerate(groups, start=1):
        if g.touching:
            continue
        stamp = group_stamp_from_roi(arr, gmap, gid, g)
        if stamp is not None:
            motifs.append(stamp)
    motifs.sort(key=lambda m: m.area, reverse=True)
    return motifs


def remove_edge_touching_components(
    arr: np.ndarray,
    bg: Sequence[int],
    threshold: float,
    margin_px: int,
    min_area: int = 40,
    *,
    ring_px: int | None = None,
    plate: np.ndarray | None = None,
) -> tuple[np.ndarray, list[tuple[float, float, int]]]:
    """
    刪除碰到邊緣帶（或以群組計的近框件）的整枚圖章。
    回傳 (清理後圖, 被刪圖案的質心與面積列表) 供後續補花定位。
    空洞用背景版填，避免平色鬼影。
    """
    out = arr.copy()
    h, w = out.shape[:2]
    removed: list[tuple[float, float, int]] = []
    if margin_px <= 0 and not ring_px:
        return out, removed

    fg = _stamp_foreground(out, bg, threshold)
    edge = _edge_band_mask(h, w, max(1, int(margin_px)))
    gmap, groups, labels, stats, centroids = group_components_into_motifs(
        fg, min_area, edge=edge
    )
    n = int(stats.shape[0])
    min_area = max(40, int(min_area))
    typical_est = max(float(min_area) / 0.03, float(min_area) * 8.0, 200.0)
    field_area = typical_est * 8.0
    attach_group_colors(arr, labels, stats, groups)
    touch_ids = {i for g in groups if g.touching for i in g.ids}
    sep = _crowd_edge_sep(stats, centroids, n, h, w, min_area, touch_ids)
    kill_g = np.zeros(int(gmap.max()) + 1, dtype=bool)
    for gid, g in enumerate(groups, start=1):
        if g.area >= field_area:
            continue
        near = sep >= 8.0 and min(g.cx, w - g.cx, g.cy, h - g.cy) <= sep
        in_ring = (
            ring_px is not None
            and min(g.cx, w - g.cx, g.cy, h - g.cy) <= float(ring_px)
        )
        wrap_band = max(2, min(8, min(h, w) // 80))
        hits_wrap = (
            g.left <= wrap_band
            or g.left + g.width >= w - wrap_band
            or g.top <= wrap_band
            or g.top + g.height >= h - wrap_band
        )
        hit = g.touching or hits_wrap
        if not hit and not near and not in_ring:
            continue
        if hit and not near and not in_ring and not hits_wrap:
            frame_hit = int(np.count_nonzero((gmap == gid) & edge))
            if frame_hit <= 5 and g.area >= min_area * 3:
                continue
        kill_g[gid] = True
        removed.append((g.cy, g.cx, g.area, True, g.rgb))
    if not removed:
        return out, removed
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kill = cv2.dilate(kill_g[gmap].astype(np.uint8), ker).astype(bool)
    overlay = _overlay_stamp_mask(arr, bg, threshold)
    if overlay is None:
        raw = _foreground_mask(arr, bg, threshold)
        if int(np.count_nonzero(raw)) > int(np.count_nonzero(fg)) * 1.08:
            wide_px = max(24, min(h, w) // 30)
            wide = _edge_band_mask(h, w, wide_px)
            kill = kill | (raw.astype(bool) & wide)
            have = {(round(t[0], 1), round(t[1], 1), int(t[2])) for t in removed}
            for gid, g in enumerate(groups, start=1):
                if kill_g[gid]:
                    continue
                if not ((gmap == gid) & wide).any():
                    continue
                job = (g.cy, g.cx, g.area, g.touching, g.rgb)
                key = (round(job[0], 1), round(job[1], 1), int(job[2]))
                if key in have:
                    continue
                have.add(key)
                kill_g[gid] = True
                removed.append(job)
            kill = cv2.dilate(kill_g[gmap].astype(np.uint8), ker).astype(bool)
            kill = kill | (raw.astype(bool) & wide)
        else:
            bridges = raw & ~fg.astype(bool)
            if bridges.any():
                band2 = _edge_band_mask(h, w, max(16, min(h, w) // 40))
                kill = kill | (bridges & band2)
    if overlay is not None:
        out[kill] = cv2.medianBlur(arr, 21)[kill]
    else:
        if plate is None:
            plate = background_plate(arr, fg.astype(bool))
        out = fill_kill_with_plate(out, kill, plate)
    return out, removed


def _component_mean_rgb(
    arr: np.ndarray, labels: np.ndarray, stats: np.ndarray, i: int
) -> tuple[float, float, float]:
    x0 = int(stats[i, cv2.CC_STAT_LEFT])
    y0 = int(stats[i, cv2.CC_STAT_TOP])
    bw = int(stats[i, cv2.CC_STAT_WIDTH])
    bh = int(stats[i, cv2.CC_STAT_HEIGHT])
    roi = labels[y0 : y0 + bh, x0 : x0 + bw] == i
    pix = arr[y0 : y0 + bh, x0 : x0 + bw][roi]
    if pix.size == 0:
        return (0.0, 0.0, 0.0)
    m = pix.mean(axis=0)
    return (float(m[0]), float(m[1]), float(m[2]))


def _stamp_mean_rgb(motif: MotifStamp) -> np.ndarray:
    pix = motif.patch[motif.mask]
    if pix.size == 0:
        return np.zeros(3, dtype=np.float64)
    return pix.reshape(-1, pix.shape[-1]).mean(axis=0).astype(np.float64)


def _opposite_edge_ink_rgb(
    canvas: np.ndarray, cy: float, cx: float, band: int = 3
) -> tuple[float, float, float] | None:
    """跨縫貼花時，對邊已有墨的平均色（避免左右各貼一色造陰陽臉）。"""
    from app.quality import ink_mask

    h, w = canvas.shape[:2]
    ink = ink_mask(canvas)
    b = max(1, min(int(band), min(h, w) // 40))
    on_v = cx <= 1.5 or cx >= w - 1.5
    on_h = cy <= 1.5 or cy >= h - 1.5
    ys: list[np.ndarray] = []
    if on_v:
        y = int(round(cy)) % h
        y0 = max(0, y - b * 4)
        y1 = min(h, y + b * 4 + 1)
        # 貼在左緣 → 看右緣；貼在右緣 → 看左緣。
        xs = w - b if cx <= 1.5 else 0
        xe = w if cx <= 1.5 else b
        roi_ink = ink[y0:y1, xs:xe]
        if roi_ink.any():
            ys.append(canvas[y0:y1, xs:xe][roi_ink])
    if on_h:
        x = int(round(cx)) % w
        x0 = max(0, x - b * 4)
        x1 = min(w, x + b * 4 + 1)
        ys_ = h - b if cy <= 1.5 else 0
        ye = h if cy <= 1.5 else b
        roi_ink = ink[ys_:ye, x0:x1]
        if roi_ink.any():
            ys.append(canvas[ys_:ye, x0:x1][roi_ink])
    if not ys:
        return None
    pix = np.concatenate([p.reshape(-1, 3) for p in ys], axis=0)
    m = pix.mean(axis=0)
    return (float(m[0]), float(m[1]), float(m[2]))



def _same_edge_ink_rgb(
    canvas: np.ndarray, cy: float, cx: float, band: int = 3
) -> tuple[float, float, float] | None:
    """同側邊帶已有墨的平均色（邊被 wipe 後對邊空，先跟本側已貼的走）。"""
    from app.quality import ink_mask

    h, w = canvas.shape[:2]
    ink = ink_mask(canvas)
    b = max(1, min(int(band), min(h, w) // 40))
    on_v = cx <= 1.5 or cx >= w - 1.5
    on_h = cy <= 1.5 or cy >= h - 1.5
    ys: list[np.ndarray] = []
    if on_v:
        y = int(round(cy)) % h
        y0 = max(0, y - b * 6)
        y1 = min(h, y + b * 6 + 1)
        xs = 0 if cx <= 1.5 else w - b
        xe = b if cx <= 1.5 else w
        roi_ink = ink[y0:y1, xs:xe]
        if roi_ink.any():
            ys.append(canvas[y0:y1, xs:xe][roi_ink])
    if on_h:
        x = int(round(cx)) % w
        x0 = max(0, x - b * 6)
        x1 = min(w, x + b * 6 + 1)
        ys_ = 0 if cy <= 1.5 else h - b
        ye = b if cy <= 1.5 else h
        roi_ink = ink[ys_:ye, x0:x1]
        if roi_ink.any():
            ys.append(canvas[ys_:ye, x0:x1][roi_ink])
    if not ys:
        return None
    pix = np.concatenate([p.reshape(-1, 3) for p in ys], axis=0)
    m = pix.mean(axis=0)
    return (float(m[0]), float(m[1]), float(m[2]))


def _templates_anchor_rgb(
    templates: list,
) -> tuple[float, float, float] | None:
    """圖章池中位色：邊帶清空後當穩定錨，避免左右亂抽造陰陽臉。"""
    if not templates:
        return None
    rgbs = [_stamp_mean_rgb(m) for m in templates]
    med = np.median(np.stack(rgbs, axis=0), axis=0)
    return (float(med[0]), float(med[1]), float(med[2]))


def _collapse_opposite_edge_jobs(
    jobs: list,
    h: int,
    w: int,
) -> list:
    """
    對邊異色各清一截時，erase 會交出兩筆不同色 job；都貼回就再造陰陽臉。
    沿同一條縫、質心接近的只留面積較大那筆。
    """
    if len(jobs) <= 1:
        return jobs
    kept: list = []
    for job in sorted(jobs, key=lambda t: int(t[2]), reverse=True):
        cy, cx = float(job[0]), float(job[1])
        skip = False
        for k in kept:
            ky, kx = float(k[0]), float(k[1])
            same_v = (min(cx, w - cx) <= 2.0 and min(kx, w - kx) <= 2.0) or (
                abs((cx % w) - (kx % w)) < 2.0
                and (cx <= 2 or cx >= w - 2)
                and (kx <= 2 or kx >= w - 2)
            )
            same_h = (min(cy, h - cy) <= 2.0 and min(ky, h - ky) <= 2.0) or (
                abs((cy % h) - (ky % h)) < 2.0
                and (cy <= 2 or cy >= h - 2)
                and (ky <= 2 or ky >= h - 2)
            )
            # 對邊：一個靠左一個靠右、y 接近；或上下對稱。
            opp_v = (cx <= 2 and kx >= w - 2) or (kx <= 2 and cx >= w - 2)
            opp_h = (cy <= 2 and ky >= h - 2) or (ky <= 2 and cy >= h - 2)
            close_y = abs((cy % h) - (ky % h)) <= max(24.0, 0.08 * h) or abs(
                (cy % h) - (ky % h)
            ) >= h - max(24.0, 0.08 * h)
            close_x = abs((cx % w) - (kx % w)) <= max(24.0, 0.08 * w) or abs(
                (cx % w) - (kx % w)
            ) >= w - max(24.0, 0.08 * w)
            if (opp_v and close_y) or (opp_h and close_x) or (same_v and close_y) or (
                same_h and close_x
            ):
                skip = True
                break
        if not skip:
            kept.append(job)
    return kept


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


def _pick_motif(
    motifs: list[MotifStamp],
    target_area: int,
    rng: np.random.Generator,
    target_rgb: tuple[float, float, float] | None = None,
) -> MotifStamp:
    if len(motifs) == 1:
        return motifs[0]
    lo = max(40.0, float(target_area) * 0.45)
    hi = float(target_area) * 2.2
    pool = [m for m in motifs if lo <= m.area <= hi]
    if not pool:
        pool = sorted(motifs, key=lambda m: abs(m.area - target_area))[:8]
    areas = np.array([m.area for m in pool], dtype=np.float64)
    fill = np.array([float(m.mask.mean()) for m in pool], dtype=np.float64)
    med_fill = float(np.median(fill))
    keep = fill >= med_fill - 0.04
    if int(keep.sum()) >= 1:
        pool = [m for m, ok in zip(pool, keep) if ok]
        areas = areas[keep]
        fill = fill[keep]
    # 指定色時硬濾掉明顯異色（粉／灰陰陽臉），避免只靠軟權重仍抽錯。
    if target_rgb is not None and len(pool) > 1:
        tgt = np.asarray(target_rgb, dtype=np.float64)
        cdist0 = np.array(
            [float(np.linalg.norm(_stamp_mean_rgb(m) - tgt)) for m in pool],
            dtype=np.float64,
        )
        hard = cdist0 <= 22.0
        if int(hard.sum()) >= 1:
            pool = [m for m, ok in zip(pool, hard) if ok]
            areas = areas[hard]
            fill = fill[hard]
        else:
            # 全不合格：至少留最近的 2～3 個，別整池亂抽。
            order = np.argsort(cdist0)
            keep_n = min(3, len(pool))
            sel = order[:keep_n]
            pool = [pool[int(i)] for i in sel]
            areas = areas[sel]
            fill = fill[sel]
    dist = np.abs(areas - float(target_area))
    weights = (1.0 / (1.0 + dist / max(float(target_area), 1.0))) * (fill + 0.05)
    if target_rgb is not None and len(pool) > 1:
        tgt = np.asarray(target_rgb, dtype=np.float64)
        cdist = np.array(
            [float(np.linalg.norm(_stamp_mean_rgb(m) - tgt)) for m in pool],
            dtype=np.float64,
        )
        weights = weights * (1.0 / (1.0 + cdist / 6.0))
    weights = weights / weights.sum()
    idx = int(rng.choice(len(pool), p=weights))
    return pool[idx]


def _torus_interval_overlap(
    a0: float, alen: float, b0: float, blen: float, n: int
) -> float:
    """環面上兩段區間重疊長度。"""
    if n <= 0 or alen <= 0 or blen <= 0:
        return 0.0
    a0 = float(a0) % n
    b0 = float(b0) % n
    best = 0.0
    for shift in (-n, 0, n):
        lo = max(a0 + shift, b0)
        hi = min(a0 + shift + alen, b0 + blen)
        best = max(best, hi - lo)
    return float(max(0.0, best))


def _dedupe_wrap_jobs(
    jobs: list[tuple[float, float, MotifStamp]], h: int, w: int
) -> list[tuple[float, float, MotifStamp]]:
    """左右殘片都會吸到同一條 wrap，距離很近或沿縫身體重疊的只留一隻。"""
    if len(jobs) <= 1:
        return jobs
    kept: list[tuple[float, float, MotifStamp]] = []
    for cy, cx, motif in jobs:
        mh, mw = int(motif.mask.shape[0]), int(motif.mask.shape[1])
        span = max(mh, mw, 8)
        skip = False
        on_v = cx <= 1.0 or cx >= w - 1.0
        on_h = cy <= 1.0 or cy >= h - 1.0
        for ky, kx, km in kept:
            kmh, kmw = int(km.mask.shape[0]), int(km.mask.shape[1])
            kspan = max(kmh, kmw, 8)
            # 用較小那隻的尺寸：大雪人不能把旁邊的冬青當成重複清掉。
            if _torus_distance(cy, cx, ky, kx, h, w) < 0.50 * min(span, kspan):
                skip = True
                break
            if on_v and (kx <= 1.0 or kx >= w - 1.0):
                ov = _torus_interval_overlap(
                    cy - motif.cy, mh, ky - km.cy, kmh, h
                )
                smaller, larger = min(mh, kmh), max(mh, kmh)
                similar = smaller >= 0.55 * larger
                if similar and ov > 0.55 * smaller:
                    skip = True
                    break
                if ov > 0.72 * larger:
                    skip = True
                    break
            if on_h and (ky <= 1.0 or ky >= h - 1.0):
                ov = _torus_interval_overlap(
                    cx - motif.cx, mw, kx - km.cx, kmw, w
                )
                smaller, larger = min(mw, kmw), max(mw, kmw)
                similar = smaller >= 0.55 * larger
                if similar and ov > 0.55 * smaller:
                    skip = True
                    break
                if ov > 0.72 * larger:
                    skip = True
                    break
        if skip:
            continue
        kept.append((cy, cx, motif))
    return kept


def _space_wrap_jobs(
    jobs: list[tuple[float, float, MotifStamp]],
    h: int,
    w: int,
    nn_sep: float = 0.0,
) -> list[tuple[float, float, MotifStamp]]:
    """
    左右殘片都會吸到同一條垂直 wrap，上下同理。
    同尺寸沿縫要比照內部間距；小圖章仍可待在大圖章旁邊。
    """
    if len(jobs) <= 1:
        return jobs
    kept: list[tuple[float, float, MotifStamp]] = []
    for cy, cx, motif in jobs:
        on_v = cx <= 1.0 or cx >= w - 1.0
        on_h = cy <= 1.0 or cy >= h - 1.0
        if not (on_v or on_h):
            kept.append((cy, cx, motif))
            continue
        span = max(int(motif.mask.shape[0]), int(motif.mask.shape[1]), 8)
        conflict = False
        for ky, kx, km in kept:
            kspan = max(int(km.mask.shape[0]), int(km.mask.shape[1]), 8)
            similar = min(span, kspan) >= 0.55 * max(span, kspan)
            if similar:
                need = 0.70 * 0.5 * (span + kspan)
                if nn_sep > 0:
                    need = max(need, nn_sep * 0.82)
            else:
                need = 0.50 * min(span, kspan)
            if on_v and (kx <= 1.0 or kx >= w - 1.0):
                dy = min(abs(cy - ky), h - abs(cy - ky))
                if dy < need:
                    conflict = True
                    break
            if on_h and (ky <= 1.0 or ky >= h - 1.0):
                dx = min(abs(cx - kx), w - abs(cx - kx))
                if dx < need:
                    conflict = True
                    break
        if not conflict:
            kept.append((cy, cx, motif))
    return kept


def _interior_nn_separation(occupied: np.ndarray, min_area: int = 80) -> float:
    """畫面內部圖章的中位最近鄰。同尺寸接縫貼花不能比這個更密。"""
    h, w = occupied.shape[:2]
    margin = max(8, min(h, w) // 20)
    inner = occupied.astype(np.uint8).copy()
    inner[_edge_band_mask(h, w, margin)] = 0
    n, _labels, stats, centroids = cv2.connectedComponentsWithStats(inner, 8)
    pts: list[tuple[float, float]] = []
    for i in range(1, n):
        if int(stats[i, cv2.CC_STAT_AREA]) >= min_area:
            pts.append((float(centroids[i][1]), float(centroids[i][0])))
    if len(pts) < 2:
        return 0.0
    dists: list[float] = []
    for i, (y, x) in enumerate(pts):
        best = 1e18
        for j, (y2, x2) in enumerate(pts):
            if i == j:
                continue
            best = min(best, _torus_distance(y, x, y2, x2, h, w))
        dists.append(best)
    return float(np.median(np.asarray(dists, dtype=np.float64)))


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
    occupied: np.ndarray | None = None,
) -> np.ndarray:
    """
    在被清掉的位置用完整圖案環繞貼回。

    半截殘片的質心靠近邊緣，完整圖章中心吸到 wrap 線上才會跨縫出現在對邊。
    內部殘缺維持原位替換。邊緣件優先貼，但沿縫間距要比照內部，不能把
    左右兩邊的花都堆到 2×2 正中央。
    """
    if not motifs:
        return arr

    out = arr.copy()
    h, w = out.shape[:2]
    rng = np.random.default_rng(seed)
    if occupied is None:
        occupied = _stamp_foreground(out, bg, threshold).astype(bool)
    else:
        occupied = occupied.astype(bool)

    def _apply(motif: MotifStamp, cy: float, cx: float) -> None:
        nonlocal occupied
        stamp_motif_wrapped(out, motif, cy, cx)
        my, mx = np.where(motif.mask)
        if len(my) == 0:
            return
        top = int(round(cy - motif.cy))
        left = int(round(cx - motif.cx))
        occupied[(top + my) % h, (left + mx) % w] = True

    edge_jobs: list[tuple[float, float, MotifStamp]] = []
    inner_jobs: list[tuple[float, float, MotifStamp]] = []
    for item in sorted(removed, key=lambda t: t[2], reverse=True):
        cy, cx, area = float(item[0]), float(item[1]), int(item[2])
        forced_edge = len(item) > 3 and bool(item[3])
        color = item[4] if len(item) > 4 else None
        motif = _pick_motif(motifs, area, rng, target_rgb=color)
        cy, cx, is_edge = _snap_wrap_placement(
            cy, cx, area, motif, h, w, forced_edge=forced_edge
        )
        (edge_jobs if (is_edge or forced_edge) else inner_jobs).append(
            (cy, cx, motif)
        )

    pending_edge = _dedupe_wrap_jobs(edge_jobs, h, w)
    # 內部 nn 常被大圖章拉到 150+，沿縫只留得下幾顆 → 2×2 白十字。
    # 步距以內部為上限，但至少保一個主圖章寬，避免假接。
    nn_sep = _interior_nn_separation(occupied)
    if pending_edge:
        spans = [
            max(int(m.mask.shape[0]), int(m.mask.shape[1]), 8)
            for _, _, m in pending_edge
        ]
        span_cap = float(np.median(np.asarray(spans, dtype=np.float64))) * 1.15
        if nn_sep > span_cap > 0:
            nn_sep = span_cap
    pending_edge = _space_wrap_jobs(pending_edge, h, w, nn_sep)
    for cap in (0.28, 0.45):
        still: list[tuple[float, float, MotifStamp]] = []
        for cy, cx, motif in pending_edge:
            if _overlap_ratio(out, motif, cy, cx, occupied) > cap:
                still.append((cy, cx, motif))
                continue
            _apply(motif, cy, cx)
        pending_edge = still
        if not pending_edge:
            break

    caps = (max_overlap, 0.12, 0.18, 0.35, 1.01)
    pending = inner_jobs
    for cap in caps:
        still = []
        for cy, cx, motif in pending:
            if cap < 1.0 and _overlap_ratio(out, motif, cy, cx, occupied) > cap:
                still.append((cy, cx, motif))
                continue
            _apply(motif, cy, cx)
        pending = still
        if not pending:
            break
    for cy, cx, motif in pending:
        _apply(motif, cy, cx)
    return out


def strong_period_score(arr: np.ndarray) -> float:
    """全圖亮度自相關的最強週期分數（0–1 量級）。"""
    gray = _luminance_map(arr)
    h, w = gray.shape[:2]
    xs = _autocorr_best_periods(
        gray.mean(0), max(16, w // 40), w // 2, top_k=1
    )
    ys = _autocorr_best_periods(
        gray.mean(1), max(16, h // 40), h // 2, top_k=1
    )
    sx = float(xs[0][1]) if xs else 0.0
    sy = float(ys[0][1]) if ys else 0.0
    return max(sx, sy)


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


_STRONG_AC = 0.35
_REPEAT_ERR_SLACK = 1.4


def _len_period_rem(length: int, period: int) -> float:
    """邊長對週期的最短餘數比例。"""
    p = int(period)
    if p < 8:
        return 0.0
    r = int(length) % p
    r = min(r, p - r)
    return float(r) / float(p)


def _near_grid_pitch(period: int, pitch: int) -> bool:
    """候選週期是否為細格基本週期的整數倍（容許 1～2px 取整）。"""
    p = int(period)
    g = int(pitch)
    if p < 8 or g < 8:
        return False
    k = max(1, int(round(p / g)))
    return abs(p - k * g) <= max(2, int(round(g * 0.06)))


def _refine_axis_pitch(gray: np.ndarray, axis: int, approx: int) -> int:
    """
    縮圖掃到的週期映射回原圖後，用全解析度重複誤差收斂到基本週期。

    細格在 512 縮圖上 3 格諧波常比 1 格分數高（1254 上 46 → 137），
    用 137 裁 411 會對 46px 格子留下 3px 錯位。
    """
    length = gray.shape[1] if axis == 0 else gray.shape[0]
    approx = max(8, int(approx))
    # 3x 諧波映射回原圖後可能 > 邊長/5（640 上 137），仍要留著跟 1 格比誤差。
    max_p = max(approx + 4, length // 4)
    max_p = min(max_p, max(8, length // 2 - 1))
    cands: set[int] = set()
    for k in (1, 2, 3, 4):
        base = int(round(approx / k)) if k > 1 else approx
        if base < 8:
            continue
        for d in range(-4, 5):
            p = base + d
            if 8 <= p <= max_p:
                cands.add(p)
    twice = 2 * approx
    if 8 <= twice <= max_p:
        for d in range(-4, 5):
            p = twice + d
            if 8 <= p <= max_p:
                cands.add(p)
    scored: list[tuple[float, int]] = []
    for p in cands:
        scored.append((_repeat_error(gray, p, axis), p))
    if not scored:
        return approx
    min_err = min(err for err, _p in scored)
    ok = [(p, err) for err, p in scored if err <= min_err * 1.22]
    ok.sort(key=lambda t: (t[0], t[1]))
    return int(ok[0][0])


def _luma_square_grid_pitch(arr: np.ndarray) -> tuple[int, int] | None:
    """
    近正方形細格（gingham／小棋盤）的亮度週期。

    棋盤的列平均是平的，必須看單列／單行。前景可到 50–95%，不能靠墨水
    質心。圖示格捷徑的 w//4 會把 46px 的格子裁錯。
    """
    if arr.ndim < 2 or min(arr.shape[:2]) < 64:
        return None
    gray = _luminance_map(arr)
    h, w = gray.shape
    side = max(h, w)
    scale = 1.0
    if side > 512:
        scale = side / 512.0
        nw = max(64, int(round(w / scale)))
        nh = max(64, int(round(h / scale)))
        gray = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)
        h, w = gray.shape
    max_p = min(h, w) // 5
    min_p = 12
    if max_p <= min_p + 4:
        return None

    def _vote(sigs: list[np.ndarray]) -> tuple[list[tuple[int, float]], int]:
        acc: dict[int, float] = {}
        strong = 0
        for sig in sigs:
            peaks = _autocorr_best_periods(sig, min_p, max_p, top_k=4)
            if peaks and peaks[0][1] >= 0.45:
                strong += 1
            for p, s in peaks:
                acc[int(p)] = max(acc.get(int(p), 0.0), float(s))
        return sorted(acc.items(), key=lambda t: -t[1]), strong

    xs, nx = _vote([gray[y] for y in (h // 5, h // 3, h // 2, (2 * h) // 3, (4 * h) // 5)])
    ys, ny = _vote([gray[:, x] for x in (w // 5, w // 3, w // 2, (2 * w) // 3, (4 * w) // 5)])
    # 格紋幾乎每列都有週期；散點圓只有少數列打到圖章。
    if nx < 3 or ny < 3:
        return None
    if not xs or not ys or xs[0][1] < 0.35 or ys[0][1] < 0.35:
        return None
    scored: list[tuple[float, int, float, int, int]] = []
    for px, sx in xs[:5]:
        if sx < 0.35:
            continue
        for py, sy in ys[:5]:
            if sy < 0.35:
                continue
            lo, hi = (px, py) if px <= py else (py, px)
            if lo < 8 or hi / max(lo, 1) > 1.22:
                continue
            err = (
                _repeat_error(gray, int(px), 0)
                + _repeat_error(gray, int(py), 1)
            )
            scored.append((err, abs(int(px) - int(py)), -(sx + sy), int(px), int(py)))
    if not scored:
        return None
    scored.sort()
    _err, _d, _s, px, py = scored[0]
    if _err > 55.0:
        return None
    px = max(8, int(round(px * scale)))
    py = max(8, int(round(py * scale)))
    full = gray if scale == 1.0 else _luminance_map(arr)
    px = _refine_axis_pitch(full, 0, px)
    py = _refine_axis_pitch(full, 1, py)
    if min(px, py) < 16:
        return None
    # 真格紋偏一個週期誤差會暴衝；圓點陣列的 8～12px 假峰幾乎不變。
    off_x = _repeat_error(full, px + max(4, px // 8), 0)
    off_y = _repeat_error(full, py + max(4, py // 8), 1)
    if _repeat_error(full, px, 0) > off_x * 0.72:
        return None
    if _repeat_error(full, py, 1) > off_y * 0.72:
        return None
    lo, hi = (px, py) if px <= py else (py, px)
    if hi / max(lo, 1) > 1.22:
        mid = int(round(0.5 * (px + py)))
        if abs(px - mid) <= 3 and abs(py - mid) <= 3:
            px = py = mid
        else:
            return None
    return px, py


def _fine_grid_aligned(arr: np.ndarray, *, rem_max: float = 0.12) -> bool:
    """細格紋且邊長已是格距整數倍：wrap 熱點是格子本身，不是錯位。"""
    fine = _luma_square_grid_pitch(arr)
    if fine is None:
        return False
    h, w = arr.shape[:2]
    return max(_len_period_rem(w, fine[0]), _len_period_rem(h, fine[1])) <= rem_max


def _mae_copy_period(arr: np.ndarray, axis: int, tol: float = 2.0) -> int | None:
    """像素幾乎完全重複的最小位移（paste 單元，不是花距自相關）。"""
    a = arr if axis == 1 else np.swapaxes(arr, 0, 1)
    h, w = a.shape[:2]
    if w < 96:
        return None
    min_p = max(64, min(h, w) // 10)
    # 只比 32～80px 時，四邊留白會讓「幾乎整張位移」MAE≈0，
    # 點綴圖就被當成有 paste 週期，週期裁切空轉一兩分鐘。
    min_overlap = max(int(round(w * 0.38)), max(32, min(w // 6, 80)))
    max_p = w - min_overlap
    if max_p <= min_p:
        return None
    row_step = max(1, h // 40)
    sl = a[::row_step].astype(np.int16)
    probe = sl[:, 0]
    col_mae = np.mean(np.abs(sl - probe[:, None, :]), axis=(0, 2))
    for p in range(min_p, max_p + 1):
        if float(col_mae[p]) > tol:
            continue
        mae = float(np.mean(np.abs(sl[:, :-p] - sl[:, p:])))
        if mae <= tol:
            return int(p)
    return None


def _strong_axis_periods(gray: np.ndarray, axis: int) -> list[int]:
    """
    全圖自相關的強週期，並用重複誤差丟掉半週期。

    格紋的半格（101 vs 203）自相關分數甚至更高，但相隔半格的內容對不上。
    只靠分數會裁出「顏色接得上、結構接不上」的假單元。
    """
    length = gray.shape[1] if axis == 0 else gray.shape[0]
    sig = gray.mean(0) if axis == 0 else gray.mean(1)
    peaks = _autocorr_best_periods(
        sig, max(16, length // 40), length // 2, top_k=5
    )
    scored: list[tuple[int, float, float]] = []
    for p, score in peaks:
        if score < _STRONG_AC:
            continue
        # 細格紋週期常 < 5% 邊長（1254 上 46px 只有 3.7%），0.05 會整組丟掉。
        if not (max(12, length * 0.028) <= p <= length * 0.48):
            continue
        err = _repeat_error(gray, int(p), axis)
        scored.append((int(p), float(score), err))
    if not scored:
        return []
    best_err = min(err for _p, _s, err in scored)
    return [p for p, _s, err in scored if err <= best_err * _REPEAT_ERR_SLACK]


def _axis_tiles_length(length: int, period: int) -> bool:
    """單元邊長是否為週期的整數倍（容許抗鋸齒級誤差）。"""
    if period < 8:
        return False
    slack = max(8, period // 16)
    k = max(1, int(round(length / period)))
    return abs(length - k * period) <= slack


def _compact_period_jobs(
    arr: np.ndarray, gray: np.ndarray
) -> list[tuple[int, int, int, int]]:
    """
    1～3 格的小單元裁切。

    舊搜尋強制覆蓋原圖 72%，密格紋／直條紋上會把 4～6 格疊在一起，
    相位一偏內部就裂。真週期常常是兩三格（406×458、一條紋寬）。
    """
    h, w = arr.shape[:2]
    xs = _strong_axis_periods(gray, 0)[:2]
    ys = _strong_axis_periods(gray, 1)[:2]
    mae_x = _mae_copy_period(arr, 1)
    mae_y = _mae_copy_period(arr, 0)
    if mae_x:
        xs = list(dict.fromkeys([mae_x, *xs]))
    if mae_y:
        ys = list(dict.fromkeys([mae_y, *ys]))
    fine = _luma_square_grid_pitch(arr)
    # 細格只以基本週期當單元；3x 諧波（137 vs 46）裁出來會差 3px。
    if fine is not None:
        xs = [fine[0]]
        ys = [fine[1]]
    jobs: list[tuple[int, int, int, int]] = []
    # 0.84：3 格波點（1641/2048）仍進得來；再高就接近「幾乎整張」假裁切。
    max_w = int(w * 0.84)
    max_h = int(h * 0.84)

    def _add(px: int, py: int, cw: int, ch: int) -> None:
        from app.select import compact_min_edge

        need = compact_min_edge(h, w)
        if cw < need or ch < need or cw > w or ch > h:
            return
        jobs.append((px, py, cw, ch))

    if xs and ys:
        for px in xs:
            for py in ys:
                if fine is not None and max(px, py) / max(min(px, py), 1) > 1.35:
                    continue
                for nx in (1, 2, 3):
                    for ny in (1, 2, 3):
                        cw, ch = nx * px, ny * py
                        if cw > max_w or ch > max_h:
                            continue
                        _add(px, py, cw, ch)
    if fine is not None:
        px, py = fine
        from app.select import compact_min_edge as _need_edge

        need = _need_edge(h, w)
        nx = max(1, int(np.ceil(need / max(px, 1))))
        ny = max(1, int(np.ceil(need / max(py, 1))))
        for _ in range(16):
            cw, ch = nx * px, ny * py
            if cw > w or ch > h:
                break
            if cw <= max_w and ch <= max_h:
                _add(px, py, cw, ch)
            if cw >= int(w * 0.72) and ch >= int(h * 0.72):
                break
            nx += 1
            ny += 1

    # 單軸條帶：部落紋／橫向幾何往往只有一軸是真週期，另一軸自相關是假峰。
    # 舊邏輯在「兩軸都有峰」時只產生 2D 小單元，假峰那一軸的 repeat error
    # 很高，裁出來 wrap 色差可以是 0、結構卻對不上；真週期那一軸反而沒被
    # 單獨裁。即使另一軸也有峰，仍把較強的一軸做成 1～3 格滿幅條帶。
    # 但滿幅那一軸必須本身接近週期整數倍，否則波點會在縫上擠成杏仁條。
    # 細格紋兩軸都是真週期，滿幅單軸條會把格子剪成長方形。
    if fine is None:
        for px in xs:
            for nx in (1, 2, 3):
                cw = nx * px
                if cw <= int(w * 0.90):
                    if ys and not _axis_tiles_length(h, ys[0]):
                        continue
                    _add(px, max(ys[0] if ys else h // 4, 16), cw, h)
        for py in ys:
            for ny in (1, 2, 3):
                ch = ny * py
                if ch <= int(h * 0.90):
                    if xs and not _axis_tiles_length(w, xs[0]):
                        continue
                    _add(max(xs[0] if xs else w // 4, 16), py, w, ch)
    # 去重、限制數量
    uniq: list[tuple[int, int, int, int]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for job in jobs:
        if job in seen:
            continue
        seen.add(job)
        uniq.append(job)
        if len(uniq) >= 18:
            break
    return uniq


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
    budget_s: float | None = None,
) -> tuple[np.ndarray | None, str]:
    """
    滿鋪幾何／斜紋／魚鱗：用亮度+梯度找獨立 xy 週期，再裁成整數倍並搜相位。
    以 structural_edge_score + 接縫色差驗證，沒改善則失敗。
    """
    del bg  # 不再依賴背景色二值化（滿鋪時易誤判）

    h, w = arr.shape[:2]
    t0 = time.perf_counter()

    def _over_budget() -> bool:
        return budget_s is not None and (time.perf_counter() - t0) >= budget_s
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
    mae_early_x = _mae_copy_period(arr, 1)
    mae_early_y = _mae_copy_period(arr, 0)
    if mae_early_x:
        xs = list(dict.fromkeys([mae_early_x, *xs]))
    if mae_early_y:
        ys = list(dict.fromkeys([mae_early_y, *ys]))
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
    # 細格紋不要走圖示格（w//4、w//6）捷徑：能量很高會被誤判，裁成大格就錯位。
    fine = _luma_square_grid_pitch(arr)
    gray_full = _luminance_map(arr)
    if fine is not None:
        fx, fy = fine
        xs = list(
            dict.fromkeys(
                [fx, 2 * fx, *[p for p in xs if _near_grid_pitch(p, fx)]]
            )
        )
        ys = list(
            dict.fromkeys(
                [fy, 2 * fy, *[p for p in ys if _near_grid_pitch(p, fy)]]
            )
        )
    else:
        xs = list(dict.fromkeys([*grid_px, *xs]))
        ys = list(dict.fromkeys([*grid_py, *ys]))
    icon_grid_likely = _looks_like_icon_checkerboard(arr) and fine is None
    if icon_grid_likely:
        xs = list(dict.fromkeys([*grid_px, *xs[:8]]))[:12]
        ys = list(dict.fromkeys([*grid_py, *ys[:8]]))[:12]
    else:
        xs = [p for p in xs if 16 <= p <= w // 2][:20]
        ys = [p for p in ys if 16 <= p <= h // 2][:20]
    mae_x = _mae_copy_period(arr, 1)
    mae_y = _mae_copy_period(arr, 0)
    if mae_x:
        xs = list(dict.fromkeys([mae_x, *xs]))[:20]
    if mae_y:
        ys = list(dict.fromkeys([mae_y, *ys]))[:20]

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
    if fine is not None:
        period_pairs.append(fine)
        period_pairs.append((2 * fine[0], 2 * fine[1]))
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
    mae_keys: set[tuple[int, int, int, int]] = set()
    px_mae = mae_x if mae_x else (xs[0] if xs else None)
    py_mae = mae_y if mae_y else (ys[0] if ys else None)
    if (mae_x or mae_y) and px_mae and py_mae:
        for dn in (0, 1):
            for dm in (0, 1):
                cw = (w // px_mae - dn) * px_mae
                ch = (h // py_mae - dm) * py_mae
                if cw < int(w * 0.72) or ch < int(h * 0.72):
                    continue
                if cw > w or ch > h or cw < 64 or ch < 64:
                    continue
                key = (px_mae, py_mae, cw, ch)
                mae_keys.add(key)
                guaranteed.add(key)
    if len(guaranteed) > g_cap:
        # 優先較小週期（細密紋）與較大覆蓋
        guaranteed = set(
            sorted(
                guaranteed,
                key=lambda t: (abs(t[0] - t[1]), t[0] + t[1], -(t[2] * t[3])),
            )[:g_cap]
        ) | mae_keys
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
                if _over_budget():
                    if log is not None:
                        log(
                            f"  → 週期裁切搜尋逾時 {budget_s:.0f}s，"
                            f"{'保留已找到的裁切' if best_tile is not None else '改走其他候選'}"
                        )
                    break
                tile, off, sc = _best_phase_for_size(
                    arr, cw, ch, px, py, sm=phase_sm, scale=phase_scale
                )
                sv, shs = _tile_seam_scores(tile)
                seam = sv + shs
                join_pen = _tile_join_run_penalty(tile)
                # 雙倍條紋等錯相位：直接淘汰，避免低色差假勝利
                if join_pen >= 40.0:
                    continue
                if fine is not None:
                    job_err = (
                        _repeat_error(gray_full, int(px), 0)
                        + _repeat_error(gray_full, int(py), 1)
                    )
                    fine_err = (
                        _repeat_error(gray_full, int(fine[0]), 0)
                        + _repeat_error(gray_full, int(fine[1]), 1)
                    )
                    if job_err > fine_err * 1.55 + 12.0:
                        continue
                    if max(
                        _len_period_rem(cw, fine[0]),
                        _len_period_rem(ch, fine[1]),
                    ) > 0.12:
                        continue
                elif max(_len_period_rem(cw, px), _len_period_rem(ch, py)) > 0.18:
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
                if fine is not None and _near_grid_pitch(px, fine[0]) and _near_grid_pitch(py, fine[1]):
                    square_bonus -= 12.0
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

    compact = None
    if not _over_budget():
        compact = _pick_compact_period_tile(arr, phase_sm, phase_scale)
    if compact is not None:
        tile, detail, wrap_ex, derr = compact
        from app.select import min_unit_edge as _min_unit_edge

        too_small = min(tile.shape[:2]) < _min_unit_edge(h, w)
        take = (best_tile is None) and (not too_small)
        if not take and not too_small:
            from app.quality import seam_report as _seam_report

            old_wrap = _seam_report(best_tile).wrap_excess
            take = wrap_ex + derr * 0.05 < old_wrap + 8.0
            if fine is not None and best_tile is not None:
                bh, bw = best_tile.shape[:2]
                rem = max(
                    _len_period_rem(bw, fine[0]),
                    _len_period_rem(bh, fine[1]),
                )
                if rem > 0.12:
                    take = True
        if take:
            best_tile = tile
            best_detail = detail

    if best_tile is None:
        return None, f"無穩定週期（接縫分 {base:.2f}，裁切無改善）"
    from app.quality import design_error as _design_error
    from app.select import DESIGN_MAX_CROP

    derr = _design_error(arr, best_tile)
    if derr > DESIGN_MAX_CROP:
        if log is not None:
            log(f"  → 週期裁切還原 {derr:.1f} 超過 {DESIGN_MAX_CROP:.0f}，當假週期放棄")
        return None, f"還原過差 {derr:.1f}"
    return best_tile, best_detail


def _pick_compact_period_tile(
    arr: np.ndarray,
    phase_sm: np.ndarray,
    phase_scale: float,
) -> tuple[np.ndarray, str, float, float] | None:
    """強自相關的 1～3 格單元。通過接縫與還原門檻才回傳。"""
    from app.quality import (
        design_error,
        seam_report,
        wrap_cut_ratio,
        wrap_density_ratio,
        wrap_gutter_error,
        wrap_period_remainder,
    )
    from app.select import (
        GUTTER_ERR_MAX,
        GUTTER_VISUAL_MAX,
        PERIOD_REM_MAX,
        WRAP_CUT_MAX,
        WRAP_DENSITY_MAX,
        WRAP_DENSITY_MIN,
        compact_min_edge,
    )

    del phase_sm, phase_scale
    gray = _luminance_map(arr)
    jobs = _compact_period_jobs(arr, gray)
    if not jobs:
        return None
    h, w = arr.shape[:2]
    src_rep = seam_report(arr)
    allow = max(src_rep.internal_excess, 6.0) * 1.15 + 2.0
    best: tuple[tuple[float, float], np.ndarray, str, float, float] | None = None
    need = compact_min_edge(h, w)
    for px, py, cw, ch in jobs:
        if min(cw, ch) < need:
            continue
        tile, off = _compact_phase(arr, cw, ch, px, py)
        rep = seam_report(tile)
        if rep.wrap_excess > 5.0:
            continue
        derr = design_error(arr, tile)
        if derr > 35.0:
            continue
        view = stamp_structure_view(tile)
        from app.quality import wrap_hotspot as _whs

        hot = _whs(view)
        # 滿鋪幾何真週期：色差／熱點都是 0，連通域卻把整片菱形當切圖。
        # 格紋 wrap/hot 可為 0，溝寬／密度／餘數仍可能錯——crop_clean 不能免。
        crop_clean = rep.wrap_excess <= 2.0 and hot <= 14.0
        gut = wrap_gutter_error(view)
        dens = wrap_density_ratio(view)
        prem = wrap_period_remainder(tile)
        if gut > GUTTER_ERR_MAX:
            continue
        if max(_len_period_rem(cw, px), _len_period_rem(ch, py)) > PERIOD_REM_MAX:
            continue
        # 縫上雙樹黏合／十字過疏：色差縫為 0 時也要擋。
        if dens > WRAP_DENSITY_MAX or dens < WRAP_DENSITY_MIN:
            continue
        cut_v = wrap_cut_ratio(view)
        # 幾何／密度／溝寬已好時，允許略高的 period_rem（大象 726×794 prem≈0.16）。
        prem_lim = PERIOD_REM_MAX
        if (
            cut_v <= WRAP_CUT_MAX
            and gut <= GUTTER_VISUAL_MAX * 0.5
            and WRAP_DENSITY_MIN <= dens <= WRAP_DENSITY_MAX
        ):
            prem_lim = max(PERIOD_REM_MAX, 0.16)
        if prem > prem_lim:
            continue
        # 點綴規則格 cut≈1 是假接／錯相位，不要贏過 cut=0 的候選。
        if cut_v >= 0.85:
            continue
        if not crop_clean:
            if rep.internal_excess > allow:
                continue
            if cut_v > WRAP_CUT_MAX:
                continue
        key = (
            rep.wrap_excess,
            0 if crop_clean else 1,
            round(cut_v, 3),
            round(gut, 3),
            round(abs(dens - 0.90), 3),
            round(prem, 3),
            derr,
            -(cw * ch),
        )
        if best is None or key < best[0]:
            detail = (
                f"週期 {px}×{py}px → 單元 {cw}×{ch}"
                f"（偏移 {off[0]},{off[1]}，小單元還原 {derr:.1f}）"
            )
            best = (key, tile, detail, rep.wrap_excess, derr)
    if best is None:
        return None
    _key, tile, detail, wrap_ex, derr = best
    return tile, detail, wrap_ex, derr


def _compact_phase(
    arr: np.ndarray, cw: int, ch: int, px: int, py: int
) -> tuple[np.ndarray, tuple[int, int]]:
    """環面相位：未裁的軸要滾開原 wrap，否則那一軸的舊縫會留在新單元邊上。

    白底規則格色差縫常多組相位都接近 0；在近最佳色差裡再比溝寬／十字密度。
    """
    from app.quality import wrap_density_ratio, wrap_gutter_error
    from app.seamless_core import torus_crop
    from app.select import WRAP_DENSITY_MAX, WRAP_DENSITY_MIN

    h, w = arr.shape[:2]
    stepx = max(2, px // 10)
    stepy = max(2, py // 10)
    xs: list[int]
    ys: list[int]
    # 滿幅那一軸仍要滾相位：條帶圖的左右縫常常只要水平移到對的位置就消失。
    if ch >= h and cw >= w:
        xs = list(range(0, w, max(8, w // 12)))
        ys = list(range(0, h, max(8, h // 12)))
    elif ch >= h:
        ys = list(range(0, h, max(8, h // 12)))
        xs = list(range(0, max(px, 1), stepx))
    elif cw >= w:
        xs = list(range(0, w, max(8, w // 12)))
        ys = list(range(0, max(py, 1), stepy))
    else:
        xs = list(range(0, max(px, 1), stepx))
        ys = list(range(0, max(py, 1), stepy))

    coarse: list[tuple[float, int, int]] = []
    best_sc = 1e9
    bxy = (xs[0], ys[0])
    for ox in xs:
        for oy in ys:
            sc = sum(_tile_seam_scores(torus_crop(arr, oy, ox, ch, cw)))
            coarse.append((sc, ox, oy))
            if sc < best_sc:
                best_sc, bxy = sc, (ox, oy)
    ox, oy = bxy
    rx = max(2, stepx // 2)
    ry = max(2, stepy // 2)
    for dx in range(-rx, rx + 1, 2):
        for dy in range(-ry, ry + 1, 2):
            x = (ox + dx) % w
            y = (oy + dy) % h
            sc = sum(_tile_seam_scores(torus_crop(arr, y, x, ch, cw)))
            coarse.append((sc, x, y))
            if sc < best_sc:
                best_sc, bxy = sc, (x, y)

    lim = best_sc + 0.75
    near = [(sc, x, y) for sc, x, y in coarse if sc <= lim]
    # 色差同分時最多比 24 個相位的結構，避免拖慢。
    near.sort()
    near = near[:24]
    best_key: tuple[float, float, float, float] | None = None
    for sc, x, y in near:
        tile = torus_crop(arr, y, x, ch, cw)
        view = stamp_structure_view(tile)
        dens = wrap_density_ratio(view)
        gut = wrap_gutter_error(view)
        dens_pen = 0.0
        if dens < WRAP_DENSITY_MIN:
            dens_pen = WRAP_DENSITY_MIN - dens
        elif dens > WRAP_DENSITY_MAX:
            dens_pen = dens - WRAP_DENSITY_MAX
        key = (sc, gut, dens_pen, abs(dens - 0.90))
        if best_key is None or key < best_key:
            best_key, bxy = key, (x, y)
    ox, oy = bxy
    return torus_crop(arr, oy, ox, ch, cw), (ox, oy)


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
    # 密花／滿鋪覆蓋高：走滿鋪週期，勿誤判點綴晶格（12k 密花會卡死）。
    # 格紋／波點例外：前景可以超過 42%，仍是規則晶格。
    if float(np.mean(fg)) > 0.42:
        return _luma_square_grid_pitch(arr) is not None
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


def _has_separated_stamps(
    arr: np.ndarray,
    bg: Sequence[int],
    threshold: float,
) -> bool:
    """
    彼此分開的圖章，即使前景到 50%（貓頭鷹）也能清邊補花。

    `_looks_like_discrete_motifs` 在 42% 就改走滿鋪週期，但這種圖的週期
    裁切常把整隻動物剖開；清邊補花才是對的工具。滿版連通底紋仍排除。
    """
    fg = _stamp_foreground(arr, bg, threshold)
    frac = float(np.mean(fg))
    if frac > 0.62 or frac < 0.04:
        return False
    h, w = fg.shape[:2]
    n, _, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    areas = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < 200:
            continue
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        # 滿版波紋／條紋連通域會拉滿一邊，不是圖章。
        if bw >= int(w * 0.85) or bh >= int(h * 0.85):
            continue
        areas.append(area)
    if len(areas) < 6:
        return False
    total = int(sum(areas)) or 1
    if max(areas) > total * 0.45:
        return False
    return True


def is_stamp_sheet(
    arr: np.ndarray,
    bg: Sequence[int],
    threshold: float = 40.0,
) -> bool:
    """分開的圖章／規則點綴：最小誤差切只會剖開，必須清邊補花。"""
    return _looks_like_discrete_motifs(arr, bg, threshold) or _has_separated_stamps(
        arr, bg, threshold
    )


def _erase_interior_fragments(
    cleaned: np.ndarray,
    bg: Sequence[int],
    threshold: float,
    margin_px: int,
    templates: list[MotifStamp],
    min_area: int = 200,
) -> tuple[np.ndarray, list[tuple[float, float, int]]]:
    """用完整同伴換掉內部缺一塊的圖章（不碰邊的殘缺雪人）。"""
    from app.quality import _component_solidity

    extra: list[tuple[float, float, int]] = []
    if len(templates) < 2:
        return cleaned, extra
    fg = _stamp_foreground(cleaned, bg, threshold)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(
        fg, connectivity=8
    )
    edge = _edge_band_mask(*cleaned.shape[:2], margin_px)
    touching = set(int(i) for i in np.unique(labels[edge]) if int(i) > 0)
    recs: list[tuple[int, int, float, float, float]] = []
    for i in range(1, n):
        if i in touching:
            continue
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < max(200, min_area):
            continue
        x0 = int(stats[i, cv2.CC_STAT_LEFT])
        y0 = int(stats[i, cv2.CC_STAT_TOP])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        roi = labels[y0 : y0 + bh, x0 : x0 + bw] == i
        solid = _component_solidity(roi)
        if solid is None:
            continue
        recs.append(
            (
                i,
                area,
                solid,
                float(centroids[i][1]),
                float(centroids[i][0]),
            )
        )
    if len(recs) < 4:
        tsols = []
        for t in templates:
            s = _component_solidity(t.mask)
            if s is not None:
                tsols.append(s)
        if tsols and recs:
            med = float(np.median(np.asarray(tsols, dtype=np.float64)))
            scored: list[tuple[float, int, float, float, int]] = []
            for i, area, solid, cy, cx in recs:
                if solid < med - 0.08 and solid < 0.80:
                    scored.append((med - solid, i, cy, cx, area))
            scored.sort(reverse=True)
            kill = [i for _, i, _, _, _ in scored]
            extra = [
                (
                    cy,
                    cx,
                    area,
                    False,
                    _component_mean_rgb(cleaned, labels, stats, i),
                )
                for _, i, cy, cx, area in scored
            ]
            if kill:
                out = cleaned.copy()
                lut = np.zeros(n, dtype=bool)
                lut[np.asarray(kill, dtype=int)] = True
                ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                mask = cv2.dilate(lut[labels].astype(np.uint8), ker).astype(bool)
                if _overlay_stamp_mask(cleaned, bg, threshold) is not None:
                    out[mask] = cv2.medianBlur(cleaned, 21)[mask]
                else:
                    plate = background_plate(
                        cleaned,
                        _stamp_foreground(cleaned, bg, threshold).astype(bool),
                    )
                    out = fill_kill_with_plate(out, mask, plate)
                extra_jobs = extra
                return out, extra_jobs
        return cleaned, extra
    # 密鋪不規則斑點不是「缺一塊的雪人」：不該整批當成內部殘缺換掉。
    if len(recs) > 24:
        return cleaned, extra
    areas = np.array([a for _, a, _, _, _ in recs], dtype=np.float64)
    sols = np.array([s for _, _, s, _, _ in recs], dtype=np.float64)
    scored: list[tuple[float, int, float, float, int]] = []
    for i, area, solid, cy, cx in recs:
        peer = (areas >= area / 1.80) & (areas <= area * 1.80)
        if int(peer.sum()) < 3:
            continue
        med = float(np.median(sols[peer]))
        if not (solid < med - 0.08 and solid < 0.80):
            continue
        if not any(
            abs(t.area - area) / max(float(area), 1.0) < 0.55 for t in templates
        ):
            continue
        scored.append((med - solid, i, cy, cx, area))
    scored.sort(reverse=True)
    scored = scored[:32]
    kill = [i for _, i, _, _, _ in scored]
    extra = [
        (
            cy,
            cx,
            area,
            False,
            _component_mean_rgb(cleaned, labels, stats, i),
        )
        for _, i, cy, cx, area in scored
    ]
    if not kill:
        return cleaned, extra
    out = cleaned.copy()
    lut = np.zeros(n, dtype=bool)
    lut[np.asarray(kill, dtype=int)] = True
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.dilate(lut[labels].astype(np.uint8), ker).astype(bool)
    if _overlay_stamp_mask(cleaned, bg, threshold) is not None:
        out[mask] = cv2.medianBlur(cleaned, 21)[mask]
    else:
        plate = background_plate(
            cleaned, _stamp_foreground(cleaned, bg, threshold).astype(bool)
        )
        out = fill_kill_with_plate(out, mask, plate)
    return out, extra


def _erase_unwrapped_frame(
    arr: np.ndarray,
    bg: Sequence[int],
    threshold: float,
    min_area: int,
) -> tuple[np.ndarray, list[tuple[float, float, int, bool]]]:
    """第二輪：畫框上仍沒接到對邊、或對邊假接的殘片再清掉。"""
    from app.quality import wrap_cut_repair

    extra: list[tuple[float, float, int, bool]] = []
    overlay = _overlay_stamp_mask(arr, bg, threshold)
    view = (
        stamp_structure_view(arr, bg, threshold)
        if overlay is not None
        else _stamp_flat_view(arr, bg, threshold)
    )
    repair = wrap_cut_repair(view)
    h, w = view.shape[:2]
    wrap_band = max(2, min(8, min(h, w) // 80))
    kill = []
    for i in repair.kill_ids:
        if int(repair.stats[i, cv2.CC_STAT_AREA]) < 80:
            continue
        x0 = int(repair.stats[i, cv2.CC_STAT_LEFT])
        y0 = int(repair.stats[i, cv2.CC_STAT_TOP])
        bw = int(repair.stats[i, cv2.CC_STAT_WIDTH])
        bh = int(repair.stats[i, cv2.CC_STAT_HEIGHT])
        if (
            x0 <= wrap_band
            or x0 + bw >= w - wrap_band
            or y0 <= wrap_band
            or y0 + bh >= h - wrap_band
        ):
            kill.append(i)
    if not kill:
        return arr, extra
    labels = repair.labels
    stats = repair.stats
    centroids = repair.centroids
    for i in kill:
        area_i = int(stats[i, cv2.CC_STAT_AREA])
        extra.append(
            (
                float(centroids[i][1]),
                float(centroids[i][0]),
                int(max(area_i, min_area)),
                True,
                _component_mean_rgb(arr, labels, stats, i),
            )
        )
    out = arr.copy()
    n = int(labels.max()) + 1
    lut = np.zeros(n, dtype=bool)
    lut[np.asarray(kill, dtype=int)] = True
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.dilate(lut[labels].astype(np.uint8), ker).astype(bool)
    if overlay is not None:
        out = _fill_overlay_with_lattice(arr, mask.astype(np.uint8))
    else:
        raw = _foreground_mask(arr, bg, threshold)
        stamp = _stamp_foreground(arr, bg, threshold)
        bridges = raw & ~stamp.astype(bool)
        if bridges.any():
            band = _edge_band_mask(*arr.shape[:2], max(16, min(*arr.shape[:2]) // 40))
            mask = mask | (bridges & band)
        plate = background_plate(arr, stamp.astype(bool))
        out = fill_kill_with_plate(out, mask, plate)
    return out, extra


_LAST_REFILL_TAG = "清邊補花"


def _refill_wrap_is_ok(
    arr: np.ndarray, bg: Sequence[int], threshold: float
) -> bool:
    """第一輪補花若已經跨縫接好，不要再 leftover 拆真圖章。"""
    from app.quality import (
        edge_void_ratio,
        ink_frac,
        seam_report,
        wrap_cut_ratio,
        wrap_density_ratio,
        wrap_edge_color_break,
        wrap_gutter_error,
        wrap_hotspot,
        wrap_orphan_run,
    )
    from app.select import (
        EDGE_COLOR_BREAK_MAX,
        GUTTER_VISUAL_MAX,
        HOTSPOT_REFILL_OK,
        ORPHAN_RUN_MAX,
        SEAM_REFILL_MAX,
        WRAP_CUT_REFILL_MAX,
        WRAP_CUT_VISUAL_MAX,
        WRAP_DENSITY_MAX,
        WRAP_DENSITY_MIN,
        WRAP_DENSITY_SPARSE_MIN,
    )

    view = _stamp_flat_view(arr, bg, threshold)
    cut = wrap_cut_ratio(view)
    dens = wrap_density_ratio(view)
    dens_floor = (
        WRAP_DENSITY_SPARSE_MIN if ink_frac(view) < 0.22 else WRAP_DENSITY_MIN
    )
    hot_ok = wrap_hotspot(view) <= HOTSPOT_REFILL_OK or cut <= WRAP_CUT_REFILL_MAX
    return (
        hot_ok
        and cut <= WRAP_CUT_VISUAL_MAX
        and wrap_orphan_run(view) <= ORPHAN_RUN_MAX
        and dens_floor <= dens <= WRAP_DENSITY_MAX
        and edge_void_ratio(arr) <= 0.26
        and wrap_gutter_error(view) <= GUTTER_VISUAL_MAX
        and wrap_edge_color_break(view) <= EDGE_COLOR_BREAK_MAX
        and seam_report(arr).wrap_excess <= SEAM_REFILL_MAX
    )


def _keep_better_refill(
    prev: np.ndarray | None,
    nxt: np.ndarray | None,
    bg: Sequence[int],
    threshold: float,
    src: np.ndarray | None = None,
) -> np.ndarray | None:
    """後續 heal／dens-jobs 若把已過密度門的結果打成過疏或掏空，退回前一版。"""
    if nxt is None:
        return prev
    if prev is None:
        return nxt
    from app.quality import edge_void_ratio, wrap_cut_ratio, wrap_density_ratio
    from app.select import VOID_MAX, WRAP_CUT_VISUAL_MAX, WRAP_DENSITY_MIN

    vp = _stamp_flat_view(prev, bg, threshold)
    vn = _stamp_flat_view(nxt, bg, threshold)
    dp = wrap_density_ratio(vp)
    dn = wrap_density_ratio(vn)
    cp = wrap_cut_ratio(vp)
    cn = wrap_cut_ratio(vn)
    # 切圖已過目視門：dens／掏空／分數都不得把 390% origin 打回來。
    cut_nxt_wins = cn <= WRAP_CUT_VISUAL_MAX < cp
    if dp >= WRAP_DENSITY_MIN and dn < WRAP_DENSITY_MIN and not cut_nxt_wins:
        return prev
    void_p = edge_void_ratio(prev)
    void_n = edge_void_ratio(nxt)
    if (
        void_p <= VOID_MAX < void_n
        and dp >= WRAP_DENSITY_MIN - 0.02
        and not cut_nxt_wins
    ):
        return prev
    if (
        dp >= WRAP_DENSITY_MIN
        and dn >= WRAP_DENSITY_MIN
        and cp <= WRAP_CUT_VISUAL_MAX
        and cn > WRAP_CUT_VISUAL_MAX
    ):
        return prev
    if (
        dn >= WRAP_DENSITY_MIN
        and cn <= WRAP_CUT_VISUAL_MAX
        and cp > WRAP_CUT_VISUAL_MAX
    ):
        return nxt
    if cut_nxt_wins:
        return nxt
    if _score_refill(nxt, bg, threshold, src=src) < _score_refill(
        prev, bg, threshold, src=src
    ):
        return nxt
    return prev


def _match_wrap_edge_colors(
    arr: np.ndarray | None,
    bg: Sequence[int],
    threshold: float,
) -> np.ndarray | None:
    """切圖已過關仍陰陽臉：把右／下緣對邊著墨像素改成左／上緣色。

    量測必須與 sweep 一致：用輸出圖自動估背景的 stamp_structure_view。
    若硬塞原稿 bg，overlay 會偏、假切圖 800% 會讓本修復被跳過，
    而 sweep 仍報 cut=0／ecb 超標（如陰陽臉近關項）。
    """
    if arr is None:
        return None
    from app.quality import ink_mask, wrap_cut_ratio, wrap_edge_color_break
    from app.select import EDGE_COLOR_BREAK_MAX, WRAP_CUT_VISUAL_MAX

    del bg, threshold  # 簽名保留給呼叫端；量測對齊 sweep 不用原稿 bg。
    view = stamp_structure_view(arr)
    cut0 = wrap_cut_ratio(view)
    if cut0 > WRAP_CUT_VISUAL_MAX:
        return arr
    ecb0 = wrap_edge_color_break(view)
    if ecb0 <= EDGE_COLOR_BREAK_MAX:
        return arr
    out = arr.copy()
    h0, w0 = out.shape[:2]
    ink = ink_mask(out)
    band = max(2, min(3, min(h0, w0) // 80))
    both = ink[:, 0] & ink[:, -1]
    if int(both.sum()) >= 12:
        ys = np.flatnonzero(both)
        for t in range(band):
            m = ink[ys, t] & ink[ys, w0 - 1 - t]
            if m.any():
                yy = ys[m]
                out[yy, w0 - 1 - t] = out[yy, t]
    both_h = ink[0, :] & ink[-1, :]
    if int(both_h.sum()) >= 12:
        xs = np.flatnonzero(both_h)
        for t in range(band):
            m = ink[t, xs] & ink[h0 - 1 - t, xs]
            if m.any():
                xx = xs[m]
                out[h0 - 1 - t, xx] = out[t, xx]
    view2 = stamp_structure_view(out)
    cut2 = wrap_cut_ratio(view2)
    ecb2 = wrap_edge_color_break(view2)
    # 切圖不得惡化；錯色至少降 8%。
    if cut2 > max(WRAP_CUT_VISUAL_MAX, cut0 + 0.02) or ecb2 >= ecb0 * 0.92:
        return arr
    return out

def _seal_wrap_background(
    arr: np.ndarray | None,
    bg: Sequence[int],
    threshold: float,
    plate: np.ndarray | None = None,
    *,
    aggressive: bool = False,
) -> np.ndarray | None:
    """把接縫上非圖章、又接近地色的像素封回背景版。

    淡底雪花／抗鋸齒殘邊進不了 stamp mask，卻能讓 wrap_excess 卡在 5～10。
    圖章本體（含膨脹一圈）不動。aggressive：第二輪略加寬，用來壓 9.8→9.0。
    """
    from app.color_utils import color_distance

    if arr is None:
        return None
    fg = _stamp_foreground(arr, bg, threshold).astype(np.uint8)
    ksz = 3 if aggressive else 5
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
    prot = cv2.dilate(fg, ker).astype(bool)
    dist = color_distance(arr, bg)
    # 抗鋸齒殘邊常略高於 threshold；過寬會啃淡色圖章邊。
    near = dist < float(threshold) * (1.65 if aggressive else 1.35)
    h, w = arr.shape[:2]
    band = 10 if aggressive else 6
    mask = np.zeros((h, w), dtype=bool)
    mask[:band, :] = True
    mask[h - band :, :] = True
    mask[:, :band] = True
    mask[:, w - band :] = True
    hit = mask & near & ~prot
    if not hit.any():
        return arr
    out = arr.copy()
    if plate is None:
        plate = background_plate(arr, fg.astype(bool))
    out[hit] = plate[hit]
    return out


def _blend_wrap_seam(arr: np.ndarray, *, n: int = 2) -> np.ndarray:
    """把對邊 1～2px 設成同一中點，讓 wrap 線差掉到門檻下（花色1 9.8）。"""
    out = arr.astype(np.float32)
    h, w = out.shape[:2]
    n = max(1, min(n, h // 8, w // 8))
    for t in range(n):
        mid = (out[t] + out[h - 1 - t]) * 0.5
        out[t] = mid
        out[h - 1 - t] = mid
        mid = (out[:, t] + out[:, w - 1 - t]) * 0.5
        out[:, t] = mid
        out[:, w - 1 - t] = mid
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def _close_refill_seams(
    filled: np.ndarray | None,
    bg: Sequence[int],
    threshold: float,
    plate: np.ndarray | None,
) -> np.ndarray | None:
    """交卷前封邊、配色，殘縫>9 再 periodize／對邊中點。早退路徑也要走這步。"""
    if filled is None:
        return None
    from app.quality import (
        seam_report as _sr_per,
        wrap_cut_ratio as _wcr_per,
        wrap_density_ratio as _wdr_per,
    )
    from app.select import (
        SEAM_OK as _SEAM_OK_SEAL,
        SEAM_REFILL_MAX as _SEAM_PER,
        WRAP_CUT_VISUAL_MAX as _CUT_PER,
        WRAP_DENSITY_MIN as _DMIN_PER,
    )
    from app.seamless_core import periodize as _periodize_end

    filled = _seal_wrap_background(filled, bg, threshold, plate)
    if _sr_per(filled).wrap_excess > _SEAM_OK_SEAL:
        filled = _seal_wrap_background(
            filled, bg, threshold, plate, aggressive=True
        )
    filled = _match_wrap_edge_colors(filled, bg, threshold)
    we0 = _sr_per(filled).wrap_excess
    if we0 <= _SEAM_PER:
        return filled
    v0 = stamp_structure_view(filled)
    dens0 = _wdr_per(v0)
    cut0 = _wcr_per(v0)
    cand = filled
    per, _pi = _periodize_end(filled)
    we1 = _sr_per(per).wrap_excess
    vp = stamp_structure_view(per)
    if (
        (we1 <= _SEAM_PER or we1 < we0 - 0.15)
        and _wcr_per(vp) <= max(_CUT_PER, cut0)
        and _wdr_per(vp) >= min(_DMIN_PER, dens0) - 0.02
    ):
        cand = per
    if _sr_per(cand).wrap_excess > _SEAM_PER:
        blended = _blend_wrap_seam(cand, n=2)
        vb = stamp_structure_view(blended)
        if (
            _wcr_per(vb) <= max(_CUT_PER, cut0)
            and _wdr_per(vb) >= min(_DMIN_PER, dens0) - 0.02
        ):
            cand = blended
    return cand


def _repair_wrap_cuts(
    arr: np.ndarray,
    bg: Sequence[int],
    threshold: float,
    templates: list,
    min_area: int,
) -> np.ndarray:
    """清掉仍停在畫框上的切圖，把完整圖章吸到 wrap 線上重貼。"""
    out = arr
    for _ in range(2):
        leftover, jobs = _erase_unwrapped_frame(out, bg, threshold, min_area)
        if not jobs:
            break
        occ = _stamp_foreground(leftover, bg, threshold).astype(bool)
        h, w = occ.shape
        band = max(8, min(h, w) // 30)
        occ[:band, :] = False
        occ[h - band :, :] = False
        occ[:, :band] = False
        occ[:, w - band :] = False
        nxt = refill_with_wrapped_motifs(
            leftover,
            bg,
            threshold,
            templates,
            jobs,
            seed=42,
            occupied=occ,
        )
        sn = _score_refill(nxt, bg, threshold)
        so = _score_refill(out, bg, threshold)
        if sn <= so:
            out = nxt
        elif sn[2] < so[2] and sn[1] <= so[1] + 2.0:
            # 切圖變小就留下，不要被色調／密度的次要項一票否決
            out = nxt
        else:
            break
        if not _refill_needs_repair(out, bg, threshold):
            break
    return out


def _refill_needs_repair(
    arr: np.ndarray, bg: Sequence[int], threshold: float
) -> bool:
    """結構切圖還沒好，或對邊顏色假接但 wrap_cut 比值是 0。"""
    from app.quality import (
        seam_report,
        wrap_cut_repair,
        wrap_edge_color_break,
    )
    from app.select import EDGE_COLOR_BREAK_MAX, SEAM_OK

    view = _stamp_flat_view(arr, bg, threshold)
    # cut=0 的陰陽臉：color_fake 只進 kill_ids，必須當未修好。
    if wrap_edge_color_break(view) > EDGE_COLOR_BREAK_MAX:
        return True
    if _refill_wrap_is_ok(arr, bg, threshold):
        return False
    from app.quality import wrap_cut_ratio as _wcr_need
    from app.select import WRAP_CUT_REFILL_MAX as _CUT_NEED

    if _wcr_need(view) > _CUT_NEED:
        return True
    if seam_report(arr).wrap_excess > SEAM_OK:
        return bool(wrap_cut_repair(view).kill_ids)
    # dens／void／gutter／切圖門檻未過：需要修。
    return True


def _dens_jobs_deadline(deadline: float | None, dens: float) -> float:
    """dens-jobs 牆鐘：過疏 22s、近門 14s、已過 12s；不得超過父截止−3s。"""
    now = time.perf_counter()
    # 已近 0.70 只留短窗，避免 10_43_58 把 90s 吃光還交不出圖。
    if dens < 0.50:
        extra = 40.0
    elif dens < 0.70:
        extra = 16.0
    else:
        extra = 8.0
    wall = now + extra
    if deadline is None:
        return wall
    return min(float(deadline) - 3.0, wall)


def _orphan_wrap_centers(arr: np.ndarray) -> list[tuple[float, float]]:
    """殘片（一側厚、對邊幾乎沒墨）連續段中點，當作跨縫 stamp 優先點。"""
    from app.quality import ink_mask

    ink = ink_mask(arr)
    h, w = ink.shape
    if h < 8 or w < 8:
        return []
    maxd = min(24, w // 4, h // 4)

    def depth(rows: np.ndarray) -> np.ndarray:
        is_bg = ~rows
        has_bg = is_bg.any(axis=1)
        first_bg = np.argmax(is_bg, axis=1)
        return np.where(has_bg, first_bg, rows.shape[1]).astype(np.int32)

    left = depth(ink[:, :maxd])
    right = depth(ink[:, -maxd:][:, ::-1])
    top = depth(ink[:maxd].T)
    bot = depth(ink[-maxd:][::-1].T)
    thick, thin = 8, 2
    v = ((left >= thick) & (right <= thin)) | ((right >= thick) & (left <= thin))
    hz = ((top >= thick) & (bot <= thin)) | ((bot >= thick) & (top <= thin))

    def runs(mask: np.ndarray) -> list[tuple[int, int]]:
        m = np.empty(mask.size + 2, dtype=np.uint8)
        m[0] = 0
        m[-1] = 0
        m[1:-1] = np.asarray(mask, dtype=np.uint8)
        d = np.diff(m.view(np.int8))
        starts = np.flatnonzero(d == 1)
        ends = np.flatnonzero(d == -1)
        return [(int(s), int(e)) for s, e in zip(starts, ends) if e - s >= 16]

    out: list[tuple[float, float]] = []
    for s, e in runs(v):
        out.append((float((s + e) // 2), 0.0))
    for s, e in runs(hz):
        out.append((0.0, float((s + e) // 2)))
    return out


def _copy_orphan_thin_side(arr: np.ndarray) -> np.ndarray:
    """殘片對邊幾乎沒墨時，把厚側最外帶抄到薄側（環面延續）。"""
    from app.quality import ink_mask

    out = arr.copy()
    ink = ink_mask(out)
    h, w = ink.shape
    if h < 8 or w < 8:
        return out
    maxd = min(24, w // 4, h // 4)
    band = min(10, maxd)

    def depth(rows: np.ndarray) -> np.ndarray:
        is_bg = ~rows
        has_bg = is_bg.any(axis=1)
        first_bg = np.argmax(is_bg, axis=1)
        return np.where(has_bg, first_bg, rows.shape[1]).astype(np.int32)

    left = depth(ink[:, :maxd])
    right = depth(ink[:, -maxd:][:, ::-1])
    top = depth(ink[:maxd].T)
    bot = depth(ink[-maxd:][::-1].T)
    thick, thin = 8, 2
    for y in range(h):
        if left[y] >= thick and right[y] <= thin:
            out[y, -band:] = out[y, :band]
        elif right[y] >= thick and left[y] <= thin:
            out[y, :band] = out[y, -band:]
    for x in range(w):
        if top[x] >= thick and bot[x] <= thin:
            out[-band:, x] = out[:band, x]
        elif bot[x] >= thick and top[x] <= thin:
            out[:band, x] = out[-band:, x]
    return out


def _maybe_repair_wrap(
    src: np.ndarray,
    filled: np.ndarray | None,
    bg: Sequence[int],
    threshold: float,
    templates: list,
    min_area: int,
    layout=None,
    plate: np.ndarray | None = None,
    deadline: float | None = None,
) -> tuple[np.ndarray | None, bool]:
    if filled is None:
        return None, False
    if not _refill_needs_repair(filled, bg, threshold):
        return filled, _refill_wrap_is_ok(filled, bg, threshold)
    from app.quality import wrap_density_ratio as _wdr_m, wrap_cut_ratio as _wcr_m
    from app.select import WRAP_CUT_REFILL_MAX as _CUT_M, WRAP_DENSITY_MIN as _DMIN_M

    dens_before = _wdr_m(_stamp_flat_view(filled, bg, threshold))
    cut_before = _wcr_m(_stamp_flat_view(filled, bg, threshold))
    origin = filled.copy()
    # 優先：殺假接後立刻跨縫補回，避免舊 repair 把密度打穿。
    if layout is not None and templates:
        healed = _heal_wrap_cuts(
            filled, bg, threshold, templates, layout, plate, deadline=deadline
        )
        view_h = _stamp_flat_view(healed, bg, threshold)
        dens_h = _wdr_m(view_h)
        cut_h = _wcr_m(view_h)
        if dens_h >= _DMIN_M and cut_h <= _CUT_M:
            return healed, True
        from app.select import WRAP_CUT_VISUAL_MAX as _CUT_VIS_M

        # 爆切圖殺邊後 dens 常掉到 0.4x；舊門檻 dens_before-0.03 會把 cut→0
        # 整段丟掉，外層又交回 cut 390%。
        cut_fixed_explosive = (
            cut_before >= 2.0 and cut_h <= _CUT_VIS_M and dens_h >= 0.18
        )
        cut_improved = cut_h < cut_before and dens_h >= dens_before - 0.03
        from app.quality import wrap_orphan_run as _wor_gate
        from app.select import ORPHAN_RUN_MAX as _ORPH_GATE

        orphan_h = int(_wor_gate(view_h))
        if (
            cut_fixed_explosive
            or cut_improved
            or dens_h < _DMIN_M
            or orphan_h > _ORPH_GATE
        ):
            filled = healed
            if templates and (dens_h < _DMIN_M or orphan_h > _ORPH_GATE):
                from dataclasses import replace
                from app.motif_layout import _place_along_wrap

                dens_deadline = _dens_jobs_deadline(deadline, dens_h)
                orphan_only = dens_h + 1e-9 >= 0.68
                if orphan_only:
                    dens_deadline = min(
                        float(dens_deadline), time.perf_counter() + 6.0
                    )
                    if deadline is not None:
                        dens_deadline = min(
                            dens_deadline, float(deadline) - 3.0
                        )
                # 用跨縫 jobs 沿 wrap 線補回（比盲 place 不易造假接）。
                h_f, w_f = filled.shape[:2]
                typ_a = max(80, int(_hero_typical_area(templates) * 0.70))
                jobs: list[tuple[float, float, int, bool]] = []
                # 0 與 h-1 是同一條環縫，勿雙貼造成 two_wholes。
                # 避開四角：橫向跨縫與直向跨縫在角上會併成 2 個大 CC。
                step = max(
                    10,
                    int(
                        round(
                            float(layout.d_nn)
                            * (0.65 if dens_h < 0.50 else 0.95)
                        )
                    ),
                )
                margin = max(int(step * (0.40 if dens_h < 0.50 else 0.48)), 20)
                for x in range(step // 2, w_f, step):
                    if margin <= x <= w_f - margin:
                        jobs.append((0.0, float(x), typ_a, True))
                for y in range(step // 2, h_f, step):
                    if margin <= y <= h_f - margin:
                        jobs.append((float(y), 0.0, typ_a, True))
                # 殘片段中點優先跨縫，避免 dens 已 0.67 卻 orphan 46 從不進 jobs。
                for cy, cx in _orphan_wrap_centers(view_h):
                    jobs.insert(0, (float(cy), float(cx), typ_a, True))

                from app.quality import ink_frac as _ink_dj
                from app.select import WRAP_DENSITY_SPARSE_MIN as _DSPARSE_M

                dens_need_dj = (
                    _DSPARSE_M
                    if _ink_dj(_stamp_flat_view(filled, bg, threshold)) < 0.22
                    else _DMIN_M
                )
                band_j = max(16, min(h_f, w_f) // 10)
                heal_fg = _stamp_foreground(filled, bg, threshold).astype(bool)
                best_f = filled
                best_d = float(dens_h)
                best_c = float(cut_h)
                best_orphan = int(orphan_h)
                accepted = 0
                n_cut = n_noop = 0
                snaps: list[tuple[np.ndarray, float, float]] = [
                    (best_f.copy(), best_d, best_c)
                ]

                def _dens_occ(cur: np.ndarray) -> np.ndarray:
                    fg_now = _stamp_foreground(cur, bg, threshold).astype(bool)
                    occ = np.zeros((h_f, w_f), dtype=bool)
                    occ[band_j : h_f - band_j, band_j : w_f - band_j] = fg_now[
                        band_j : h_f - band_j, band_j : w_f - band_j
                    ]
                    new_ink = fg_now & ~heal_fg
                    occ[:band_j] |= new_ink[:band_j]
                    occ[h_f - band_j :] |= new_ink[h_f - band_j :]
                    occ[:, :band_j] |= new_ink[:, :band_j]
                    occ[:, w_f - band_j :] |= new_ink[:, w_f - band_j :]
                    return occ

                heroes = sorted(
                    templates, key=lambda t: int(getattr(t, "area", 0)), reverse=True
                )
                n_h = len(heroes)
                probes: list = []
                if n_h:
                    seen_m = set()
                    for idx in (0, n_h // 3, min(n_h - 1, (2 * n_h) // 3)):
                        m = heroes[idx]
                        mid = id(m)
                        if mid in seen_m:
                            continue
                        seen_m.add(mid)
                        probes.append(m)
                if not probes:
                    probes = list(templates[:4])
                if dens_h >= 0.55 and n_h:
                    for m in heroes[max(0, n_h // 2) : max(0, n_h // 2) + 3]:
                        if id(m) not in seen_m:
                            seen_m.add(id(m))
                            probes.append(m)

                from app.quality import wrap_cut_repair as _wcr_sliver
                from app.motif_layout import fill_kill_with_plate as _fk_sliver

                if not orphan_only:
                    sl = _wcr_sliver(_stamp_flat_view(best_f, bg, threshold))
                    typ_sl = max(float(sl.typical), 80.0)
                    small_ids = [
                        i
                        for i in sl.kill_ids
                        if int(sl.stats[i, cv2.CC_STAT_AREA]) < typ_sl * 0.55
                    ]
                    if small_ids:
                        kill_sl = np.isin(
                            sl.labels, np.asarray(small_ids, dtype=np.int32)
                        )
                        if plate is None:
                            plate = background_plate(
                                best_f,
                                _stamp_foreground(best_f, bg, threshold).astype(bool),
                            )
                        best_f = _fk_sliver(best_f, kill_sl, plate)
                        best_d = float(_wdr_m(_stamp_flat_view(best_f, bg, threshold)))
                        best_c = float(_wcr_m(_stamp_flat_view(best_f, bg, threshold)))
                        snaps[0] = (best_f.copy(), best_d, best_c)

                smalls = sorted(
                    templates, key=lambda t: int(getattr(t, "area", 0))
                )[:8]
                for cy, cx in _orphan_wrap_centers(
                    _stamp_flat_view(best_f, bg, threshold)
                ):
                    if dens_deadline is not None and time.perf_counter() >= float(
                        dens_deadline
                    ) - 1.0:
                        break
                    if best_orphan <= _ORPH_GATE and best_d + 1e-9 >= dens_need_dj:
                        break
                    for motif in smalls or probes:
                        nxt = best_f.copy()
                        stamp_motif_wrapped(nxt, motif, cy, cx)
                        vn = _stamp_flat_view(nxt, bg, threshold)
                        cut_n = _wcr_m(vn)
                        if cut_n > _CUT_VIS_M:
                            continue
                        dens_n = _wdr_m(vn)
                        orphan_n = int(_wor_gate(vn))
                        if orphan_n >= best_orphan and dens_n < best_d + 0.002:
                            continue
                        if dens_n + 1e-9 < best_d - 0.02:
                            continue
                        best_f, best_d, best_c = nxt, dens_n, cut_n
                        best_orphan = orphan_n
                        accepted += 1
                        snaps.append((best_f.copy(), best_d, best_c))
                        break

                if best_orphan > _ORPH_GATE:
                    cpy = _copy_orphan_thin_side(best_f)
                    vn_c = _stamp_flat_view(cpy, bg, threshold)
                    cut_c = _wcr_m(vn_c)
                    if cut_c <= _CUT_VIS_M:
                        dens_c = _wdr_m(vn_c)
                        orphan_c = int(_wor_gate(vn_c))
                        if orphan_c < best_orphan and dens_c + 1e-9 >= best_d - 0.03:
                            best_f, best_d, best_c = cpy, dens_c, cut_c
                            best_orphan = orphan_c
                            snaps.append((best_f.copy(), best_d, best_c))

                if orphan_only:
                    filled = best_f
                    filled = _keep_better_refill(
                        origin, filled, bg, threshold, src=src
                    )
                    return filled, _refill_wrap_is_ok(filled, bg, threshold)

                for bi, job in enumerate(jobs):
                    if dens_deadline is not None and time.perf_counter() >= float(
                        dens_deadline
                    ) - 1.0:
                        break
                    if (
                        best_d + 1e-9 >= dens_need_dj
                        and best_c <= _CUT_VIS_M
                        and best_orphan <= _ORPH_GATE
                    ):
                        break
                    if not probes:
                        break
                    cy, cx = float(job[0]), float(job[1])
                    occ_now = _dens_occ(best_f)
                    ov_lim = 0.45 if best_d < 0.50 else (0.22 if best_d < 0.62 else 0.18)
                    use = probes if best_d < 0.50 else (smalls if best_d < 0.55 and smalls else probes)
                    placed = False
                    for motif in use:
                        if best_d >= 0.50 and _overlap_ratio(
                            occ_now, motif, cy, cx, occ_now
                        ) > ov_lim:
                            continue
                        nxt = best_f.copy()
                        stamp_motif_wrapped(nxt, motif, cy, cx)
                        vn = _stamp_flat_view(nxt, bg, threshold)
                        dens_n = _wdr_m(vn)
                        if dens_n + 1e-9 <= best_d:
                            continue
                        if best_d < 0.50:
                            cut_n = best_c
                        else:
                            cut_n = _wcr_m(vn)
                            if cut_n > _CUT_VIS_M:
                                continue
                        orphan_n = int(_wor_gate(vn))
                        dens_gain = dens_n >= best_d + 0.002
                        orphan_gain = (
                            orphan_n + 4 < best_orphan
                            and dens_n + 1e-9 >= best_d - 0.01
                        )
                        if not dens_gain and not orphan_gain:
                            continue
                        best_f, best_d, best_c = nxt, dens_n, cut_n
                        best_orphan = orphan_n
                        accepted += 1
                        snaps.append((best_f.copy(), best_d, best_c))
                        placed = True
                        break
                    if not placed:
                        n_cut += 1
                if best_d + 1e-9 < dens_need_dj and probes:
                    hero = probes[0]
                    burst = best_f.copy()
                    n_b = 0
                    for job in jobs:
                        if dens_deadline is not None and time.perf_counter() >= float(
                            dens_deadline
                        ) - 1.0:
                            break
                        stamp_motif_wrapped(
                            burst, hero, float(job[0]), float(job[1])
                        )
                        n_b += 1
                        if n_b % 4 != 0:
                            continue
                        dens_b2 = _wdr_m(_stamp_flat_view(burst, bg, threshold))
                        if dens_b2 > 1.15:
                            burst = best_f.copy()
                            continue
                        if dens_b2 > best_d:
                            best_f = burst.copy()
                            best_d = float(dens_b2)
                            snaps.append((best_f.copy(), best_d, best_c))
                        if best_d + 1e-9 >= dens_need_dj:
                            break
                if best_d + 1e-9 < dens_need_dj or best_orphan > _ORPH_GATE:
                    step2 = max(8, step // 2)
                    extra: list[tuple[float, float, int, bool]] = []
                    off = max(4, step2 // 2)
                    for x in range(off, w_f, step2):
                        if margin <= x <= w_f - margin:
                            extra.append((0.0, float(x), typ_a, True))
                    for y in range(off, h_f, step2):
                        if margin <= y <= h_f - margin:
                            extra.append((float(y), 0.0, typ_a, True))
                    for job in extra:
                        if dens_deadline is not None and time.perf_counter() >= float(
                            dens_deadline
                        ) - 1.0:
                            break
                        if (
                            best_d + 1e-9 >= dens_need_dj
                            and best_c <= _CUT_VIS_M
                            and best_orphan <= _ORPH_GATE
                        ):
                            break
                        cy, cx = float(job[0]), float(job[1])
                        occ_now = _dens_occ(best_f)
                        placed = False
                        use_e = smalls if best_d < 0.55 and smalls else probes
                        for motif in use_e:
                            if best_d >= 0.50 and _overlap_ratio(
                                occ_now, motif, cy, cx, occ_now
                            ) > 0.22:
                                continue
                            nxt = best_f.copy()
                            stamp_motif_wrapped(nxt, motif, cy, cx)
                            vn = _stamp_flat_view(nxt, bg, threshold)
                            dens_n = _wdr_m(vn)
                            cut_n = _wcr_m(vn)
                            if (cut_n > _CUT_VIS_M and best_d >= 0.50) or dens_n < best_d + 0.002:
                                continue
                            best_f, best_d, best_c = nxt, dens_n, cut_n
                            accepted += 1
                            snaps.append((best_f.copy(), best_d, best_c))
                            placed = True
                            break
                        if not placed:
                            n_cut += 1
                from app.quality import seam_report as _sr_dj

                if 0.55 <= best_d + 1e-9 < dens_need_dj:
                    mile = list(jobs)
                    for job in mile:
                        if dens_deadline is not None and time.perf_counter() >= float(
                            dens_deadline
                        ) - 0.8:
                            break
                        if best_d + 1e-9 >= dens_need_dj:
                            break
                        cy, cx = float(job[0]), float(job[1])
                        occ_now = _dens_occ(best_f)
                        for motif in probes:
                            if _overlap_ratio(occ_now, motif, cy, cx, occ_now) > 0.38:
                                continue
                            nxt = best_f.copy()
                            stamp_motif_wrapped(nxt, motif, cy, cx)
                            vn = _stamp_flat_view(nxt, bg, threshold)
                            dens_n = _wdr_m(vn)
                            cut_n = _wcr_m(vn)
                            if cut_n > _CUT_VIS_M or dens_n < best_d + 0.001:
                                continue
                            best_f, best_d, best_c = nxt, dens_n, cut_n
                            accepted += 1
                            snaps.append((best_f.copy(), best_d, best_c))
                            break

                allow_ie = max(_sr_dj(src).internal_excess, 6.0) * 1.15 + 2.0
                good: list[tuple[np.ndarray, float, float]] = []
                dens_ok_snaps: list[tuple[float, np.ndarray, float, float]] = []
                for arr_s, d_s, c_s in snaps:
                    ie_s = _sr_dj(arr_s).internal_excess
                    if d_s + 1e-9 >= dens_need_dj and c_s <= _CUT_VIS_M:
                        dens_ok_snaps.append((ie_s, arr_s, d_s, c_s))
                        if ie_s <= allow_ie + 0.05:
                            good.append((arr_s, d_s, c_s))
                if good:
                    best_f, best_d, best_c = max(good, key=lambda t: t[1])
                elif dens_ok_snaps:
                    dens_ok_snaps.sort(key=lambda t: t[0])
                    _, best_f, best_d, best_c = dens_ok_snaps[0]
                else:
                    # dens_need 可能是 sparse 0.78；目視門 0.70+cut≤0.30 的中間態仍要留。
                    vis0 = [
                        (a, d, c)
                        for a, d, c in snaps
                        if d + 1e-9 >= _DMIN_M and c <= _CUT_VIS_M
                    ]
                    if vis0:
                        best_f, best_d, best_c = max(vis0, key=lambda t: t[1])
                filled = best_f
                import os as _os_dj
                if _os_dj.environ.get("SEAMLESS_HEAL_TRACE"):
                    print(
                        f"  dens-jobs dens {dens_h:.3f}->{best_d:.3f} "
                        f"cut {best_c:.3f} ok={accepted} cutrej={n_cut} noop={n_noop} "
                        f"ie={_sr_dj(filled).internal_excess:.1f}/{allow_ie:.1f}",
                        flush=True,
                    )
                for seed_r in (71, 79, 89, 97, 103, 109, 127, 139, 149, 157, 163, 173):
                    if dens_deadline is not None and time.perf_counter() >= float(
                        dens_deadline
                    ) - 1.5:
                        break
                    dens_now = _wdr_m(_stamp_flat_view(filled, bg, threshold))
                    if dens_now >= dens_need_dj:
                        break
                    snap = filled.copy()
                    occ_r = _dens_occ(filled)
                    n_put = _place_along_wrap(
                        filled,
                        occ_r,
                        templates,
                        replace(
                            layout,
                            d_nn=max(4.0, float(layout.d_nn) * (0.12 if dens_now < 0.45 else 0.18)),
                            d_min=max(3.5, float(layout.d_min) * (0.12 if dens_now < 0.45 else 0.18)),
                        ),
                        np.random.default_rng(seed_r),
                        prefer_hero=False,
                        relax_inner=0.12,
                        max_place=4 if dens_now < 0.45 else 2,
                        deadline=dens_deadline,
                    )
                    if n_put <= 0:
                        filled = snap
                        continue
                    vb2 = _stamp_flat_view(filled, bg, threshold)
                    if _wcr_m(vb2) > _CUT_VIS_M:
                        filled = snap
                        continue
                    if _wdr_m(vb2) < dens_now + 0.005:
                        filled = snap
                        continue
                    snaps.append(
                        (filled.copy(), float(_wdr_m(vb2)), float(_wcr_m(vb2)))
                    )
                snap_cut0 = filled.copy()
                if _wdr_m(_stamp_flat_view(filled, bg, threshold)) + 1e-9 < dens_need_dj:
                    filled = _boost_wrap_fill(
                        filled,
                        bg,
                        threshold,
                        templates,
                        layout,
                        deadline=dens_deadline,
                    )
                view_b = _stamp_flat_view(filled, bg, threshold)
                dens_b = _wdr_m(view_b)
                cut_b = _wcr_m(view_b)
                if cut_b > _CUT_VIS_M and dens_b + 1e-9 < _DMIN_M:
                    filled = snap_cut0
                    view_b = _stamp_flat_view(filled, bg, threshold)
                    dens_b = _wdr_m(view_b)
                    cut_b = _wcr_m(view_b)
                snaps.append((filled.copy(), float(dens_b), float(cut_b)))
                orphan_b = int(_wor_gate(_stamp_flat_view(filled, bg, threshold)))
                if dens_b >= _DMIN_M and cut_b <= _CUT_M and orphan_b <= _ORPH_GATE:
                    return filled, True
                # 任一中間態已過 dens≥0.70 且 cut≤0.30：留下，勿退回 cut 爆表的 origin。
                vis_pass = [
                    (a, d, c)
                    for a, d, c in snaps
                    if d + 1e-9 >= _DMIN_M
                    and c <= _CUT_VIS_M
                    and int(_wor_gate(_stamp_flat_view(a, bg, threshold)))
                    <= _ORPH_GATE
                ]
                if vis_pass:
                    filled, dens_b, cut_b = max(
                        vis_pass, key=lambda t: (t[1], -t[2])
                    )
                    return filled, True
                # 沒補回 0.70，但切圖已過目視門：留下 dens 最高的 cut=0，勿退回 390%。
                cut_ok = [
                    (a, d, c)
                    for a, d, c in snaps
                    if c <= _CUT_VIS_M
                ]
                if cut_ok:
                    filled, dens_b, cut_b = max(cut_ok, key=lambda t: t[1])
                    filled = _keep_better_refill(
                        origin, filled, bg, threshold, src=src
                    )
                    return filled, _refill_wrap_is_ok(filled, bg, threshold)
                import os as _os_end
                if _os_end.environ.get("SEAMLESS_HEAL_TRACE"):
                    print(
                        f"  dens-end dens={dens_b:.3f} cut={cut_b:.3f}",
                        flush=True,
                    )
                if dens_b + 1e-9 < dens_need_dj and cut_b > _CUT_VIS_M:
                    filled = origin
    # 舊 repair 極慢且常打穿 dens；時間緊或 heal 已試過就少跑。
    filled = _keep_better_refill(origin, filled, bg, threshold, src=src)
    if not _refill_needs_repair(filled, bg, threshold):
        return filled, _refill_wrap_is_ok(filled, bg, threshold)
    if deadline is not None and time.perf_counter() >= float(deadline) - 3.0:
        return filled, _refill_wrap_is_ok(filled, bg, threshold)
    # 殺邊後仍過疏：不要把白十字當修好，也不要再跑舊 repair 打穿。
    view_keep = _stamp_flat_view(filled, bg, threshold)
    if (
        _wdr_m(view_keep) + 1e-9 < _DMIN_M
        and dens_before >= _DMIN_M
    ):
        return origin, _refill_wrap_is_ok(origin, bg, threshold)
    repaired = _repair_wrap_cuts(filled, bg, threshold, templates, min_area)
    if layout is not None and templates:
        repaired = _heal_wrap_cuts(
            repaired, bg, threshold, templates, layout, plate, deadline=deadline
        )
    if not _refill_needs_repair(repaired, bg, threshold):
        return repaired, _refill_wrap_is_ok(repaired, bg, threshold)
    dens_rep = _wdr_m(_stamp_flat_view(repaired, bg, threshold))
    # 修切圖修到過疏：寧願退回 heal／原稿補花，交給外層整張重排。
    if dens_before >= _DMIN_M and dens_rep < _DMIN_M - 0.02:
        if _clear_edge_ruins_motifs(src, repaired, bg, threshold):
            return origin, False
        kept = _keep_better_refill(origin, filled, bg, threshold, src=src)
        return kept, _refill_wrap_is_ok(kept, bg, threshold)
    if _clear_edge_ruins_motifs(src, repaired, bg, threshold):
        return origin, False
    kept = _keep_better_refill(origin, repaired, bg, threshold, src=src)
    return kept, _refill_wrap_is_ok(kept, bg, threshold)


def _score_refill(
    cand: np.ndarray,
    bg: Sequence[int],
    threshold: float,
    src: np.ndarray | None = None,
) -> tuple:
    from app.quality import (
        edge_void_ratio,
        seam_report,
        tone_shift,
        wrap_cut_ratio,
        wrap_density_ratio,
        wrap_edge_color_break,
        wrap_gutter_error,
        wrap_hotspot,
    )
    from app.select import (
        EDGE_COLOR_BREAK_MAX,
        GUTTER_VISUAL_MAX,
        SEAM_OK,
        TONE_REFILL_MAX,
        WRAP_CUT_REFILL_MAX,
        WRAP_DENSITY_MIN,
    )

    view = _stamp_flat_view(cand, bg, threshold)
    dens = wrap_density_ratio(view)
    wrap_ex = seam_report(cand).wrap_excess
    cut = wrap_cut_ratio(view)
    errs = 0 if _refill_wrap_is_ok(cand, bg, threshold) else 3
    if wrap_ex > SEAM_OK:
        errs += 2
    if dens > 1.15:
        errs += 8
    if dens > 1.30:
        errs += 4
    if cut > WRAP_CUT_REFILL_MAX:
        errs += 16
        # 爆切圖（≥200%）必須壓過 void／過疏罰分，否則 heal 切到 0 會被
        # dart／erase 候選用 dens0.97+cut3.9 的較低 errs 換回去。
        if cut >= 2.0:
            errs += 24 + int(min(float(cut), 8.0) * 4.0)
    elif cut > WRAP_CUT_REFILL_MAX * 0.75:
        errs += 4
    from app.select import WRAP_CUT_VISUAL_MAX as _CUT_VIS_SCR

    if cut > _CUT_VIS_SCR:
        errs += 6
    if dens < 0.85:
        errs += 3
    if dens < WRAP_DENSITY_MIN:
        errs += 8
    if dens < 0.55:
        errs += 8
    # 十字相對內部的偏離：擠圈（>1.15 已罰）與白十字（偏疏）都加重。
    dens_dev = abs(float(dens) - 0.92)
    if dens_dev > 0.18:
        errs += 2
    void = edge_void_ratio(cand)
    if void > 0.26:
        errs += 10
    if void > 0.40:
        errs += 8
    gut = wrap_gutter_error(view)
    if gut > GUTTER_VISUAL_MAX:
        errs += 12
    elif gut > GUTTER_VISUAL_MAX * 0.5:
        errs += 4
    ecb = wrap_edge_color_break(view)
    if ecb > EDGE_COLOR_BREAK_MAX:
        errs += 10
    elif ecb > EDGE_COLOR_BREAK_MAX * 0.7:
        errs += 3
    # 陰陽臉是一眼假接：權重必須能壓過 dens 微幅變差，否則 heal 永遠被退回。
    if ecb > EDGE_COLOR_BREAK_MAX:
        errs += 12 + int(min(20.0, (ecb - EDGE_COLOR_BREAK_MAX) / 3.0))
    if src is not None and tone_shift(src, cand, edge_frac=0.20) > TONE_REFILL_MAX:
        errs += 4
    return (
        errs,
        wrap_ex,
        0.0 if cut <= WRAP_CUT_REFILL_MAX else cut,
        dens_dev,
        wrap_hotspot(view),
        void,
        gut,
        ecb,
    )


def _boost_wrap_fill(
    canvas: np.ndarray | None,
    bg: Sequence[int],
    threshold: float,
    templates: list,
    layout,
    deadline: float | None = None,
) -> np.ndarray | None:
    """補花若留下十字空帶／邊帶掏空，再往 wrap 環帶塞完整圖章。"""
    if canvas is None or not templates:
        return canvas
    from dataclasses import replace
    from app.motif_layout import _place_along_wrap
    from app.quality import (
        edge_void_ratio,
        wrap_cut_ratio,
        wrap_density_ratio,
        wrap_edge_color_break,
        wrap_gutter_error,
    )
    from app.quality import ink_frac
    from app.select import (
        EDGE_COLOR_BREAK_MAX,
        GUTTER_VISUAL_MAX,
        WRAP_CUT_REFILL_MAX,
        WRAP_DENSITY_MAX,
        WRAP_DENSITY_MIN,
        WRAP_DENSITY_SPARSE_MIN,
    )

    def _oot(reserve: float = 2.0) -> bool:
        return deadline is not None and time.perf_counter() >= float(deadline) - reserve

    if _oot(2.0):
        return canvas

    view = _stamp_flat_view(canvas, bg, threshold)
    dens = wrap_density_ratio(view)
    void = edge_void_ratio(canvas)
    gut = wrap_gutter_error(view)
    ecb = wrap_edge_color_break(view)
    dens_need = (
        WRAP_DENSITY_SPARSE_MIN if ink_frac(view) < 0.22 else WRAP_DENSITY_MIN
    )
    if (
        dens_need <= dens <= WRAP_DENSITY_MAX
        and void <= 0.26
        and gut <= GUTTER_VISUAL_MAX
        and ecb <= EDGE_COLOR_BREAK_MAX
    ):
        return canvas
    # 已過密：勿再塞花。
    if dens > WRAP_DENSITY_MAX and ecb <= EDGE_COLOR_BREAK_MAX:
        return canvas
    # 多輪漸緊：錯色／清邊後 dens 常停在 0.55–0.69，一輪塞不滿。
    base_factor = (
        0.55 if void > 0.40 or gut > GUTTER_VISUAL_MAX else (0.62 if dens < 0.55 else 0.78)
    )
    if dens < dens_need:
        base_factor = min(base_factor, 0.58 if dens < 0.62 else 0.70)
    if ecb > EDGE_COLOR_BREAK_MAX:
        base_factor = min(base_factor, 0.55)
    base_relax = (
        0.50 if (gut > GUTTER_VISUAL_MAX or ecb > EDGE_COLOR_BREAK_MAX) else 1.0
    )
    if dens < dens_need:
        base_relax = min(
            base_relax,
            0.42 if dens < 0.60 else (0.50 if dens < 0.66 else 0.58),
        )
    passes = (
        (base_factor, base_relax, 11),
        (min(base_factor, 0.42), min(base_relax, 0.36), 19),
        (min(base_factor, 0.28), min(base_relax, 0.24), 29),
    )
    for fac, relax, seed in passes:
        if _oot(2.5):
            break
        view = _stamp_flat_view(canvas, bg, threshold)
        dens = wrap_density_ratio(view)
        void = edge_void_ratio(canvas)
        gut = wrap_gutter_error(view)
        ecb = wrap_edge_color_break(view)
        dens_need = (
            WRAP_DENSITY_SPARSE_MIN if ink_frac(view) < 0.22 else WRAP_DENSITY_MIN
        )
        if (
            dens >= dens_need
            and void <= 0.26
            and gut <= GUTTER_VISUAL_MAX
            and ecb <= EDGE_COLOR_BREAK_MAX
        ):
            break
        before = canvas.copy()
        dens_b = dens
        cut_b = wrap_cut_ratio(view)
        ecb_b = ecb
        occ = _stamp_foreground(canvas, bg, threshold).astype(bool)
        # dens 差一截時一次只貼 1～2 枚，避免 0.56→1.5 被護欄整輪打回。
        # 切圖已過目視門：一律護切圖（不限稀墨），貼壞立刻退。
        from app.select import WRAP_CUT_VISUAL_MAX as _CUT_VIS_B

        cut_guard = cut_b <= _CUT_VIS_B
        if cut_guard and dens_b < 0.45:
            cap = 6
        elif cut_guard and dens_b < 0.62:
            cap = 4
        elif cut_guard:
            cap = 2
        elif dens_b < dens_need:
            cap = 2
        else:
            cap = None
        n_put = _place_along_wrap(
            canvas,
            occ,
            templates,
            replace(
                layout,
                d_nn=max(8.0, float(layout.d_nn) * fac),
                d_min=max(6.0, float(layout.d_min) * fac),
            ),
            np.random.default_rng(seed),
            prefer_hero=True,
            relax_inner=relax,
            max_place=cap,
            deadline=deadline,
        )
        if n_put <= 0:
            canvas[:] = before
            continue
        view_n = _stamp_flat_view(canvas, bg, threshold)
        dens_n = wrap_density_ratio(view_n)
        cut_n = wrap_cut_ratio(view_n)
        ecb_n = wrap_edge_color_break(view_n)
        # 補 dens 不得把錯色貼回來。
        if ecb_n > max(EDGE_COLOR_BREAK_MAX, ecb_b + 0.5):
            canvas[:] = before
            continue
        if cut_guard and cut_n > _CUT_VIS_B:
            canvas[:] = before
            continue
        score_worse = _score_refill(canvas, bg, threshold) > _score_refill(
            before, bg, threshold
        )
        # dens 往門檻爬且切圖沒爆：即使分數微差也留下（避免互打退回過疏）。
        dens_help = (
            dens_b < dens_need
            and dens_n >= dens_b + (0.008 if cut_guard else 0.025)
            and dens_n <= WRAP_DENSITY_MAX
            and cut_n <= (_CUT_VIS_B if cut_guard else WRAP_CUT_REFILL_MAX)
            and (cut_b > WRAP_CUT_REFILL_MAX or cut_n <= cut_b + 0.08)
            and ecb_n <= EDGE_COLOR_BREAK_MAX
        )
        if dens_n > WRAP_DENSITY_MAX and dens_b <= WRAP_DENSITY_MAX:
            canvas[:] = before
            continue
        if score_worse and not dens_help:
            canvas[:] = before
    # dens 仍災難級：最後一搏極緊貼（鎖色已在 place 內）。
    view = _stamp_flat_view(canvas, bg, threshold)
    dens = wrap_density_ratio(view)
    ecb = wrap_edge_color_break(view)
    dens_need = (
        WRAP_DENSITY_SPARSE_MIN if ink_frac(view) < 0.22 else WRAP_DENSITY_MIN
    )
    if dens < dens_need and ecb <= EDGE_COLOR_BREAK_MAX and not _oot(3.0):
        occ = _stamp_foreground(canvas, bg, threshold).astype(bool)
        from app.select import WRAP_CUT_VISUAL_MAX as _CUT_VIS_E

        cut_lock = wrap_cut_ratio(view) <= _CUT_VIS_E
        # 一次一枚爬 dens；輪數封頂，避免掃掃單張拖到數百秒。
        for seed_e, fac_e, ri_e in (
            (41, 0.55, 0.45),
            (43, 0.35, 0.30),
            (47, 0.22, 0.20),
            (53, 0.12, 0.14),
            (59, 0.10, 0.12),
            (61, 0.08, 0.10),
        ):
            if _oot(2.0):
                break
            dens0 = wrap_density_ratio(_stamp_flat_view(canvas, bg, threshold))
            if dens0 >= dens_need:
                break
            snap = canvas.copy()
            n_put = _place_along_wrap(
                canvas,
                occ,
                templates,
                replace(
                    layout,
                    d_nn=max(5.0, float(layout.d_nn) * fac_e),
                    d_min=max(4.0, float(layout.d_min) * fac_e),
                ),
                np.random.default_rng(seed_e),
                prefer_hero=False,
                relax_inner=ri_e,
                max_place=1,
                deadline=deadline,
            )
            if n_put <= 0:
                canvas[:] = snap
                continue
            view_n = _stamp_flat_view(canvas, bg, threshold)
            dens_n = wrap_density_ratio(view_n)
            cut_n = wrap_cut_ratio(view_n)
            if (
                wrap_edge_color_break(view_n) > EDGE_COLOR_BREAK_MAX
                or dens_n < dens0 + 0.008
                or dens_n > WRAP_DENSITY_MAX
                or (cut_lock and cut_n > _CUT_VIS_E)
            ):
                canvas[:] = snap
            else:
                occ = _stamp_foreground(canvas, bg, threshold).astype(bool)
    return canvas


def _heal_wrap_cuts(
    canvas: np.ndarray,
    bg: Sequence[int],
    threshold: float,
    templates: list,
    layout,
    plate: np.ndarray | None,
    deadline: float | None = None,
) -> np.ndarray:
    """
    清掉 wrap_cut 標成假接／切圖的邊件，再以主圖章跨縫補回。

    舊路徑 `_repair_wrap_cuts` 反覆 erase 後常把密度打到 0.6 以下；
    cut 爆表時最多兩輪：殺切圖 → 跨縫重貼 → dens 補回。變差則退回。
    """
    from dataclasses import replace
    from app.motif_layout import _place_along_wrap, fill_kill_with_plate
    from app.quality import ink_frac, wrap_cut_ratio, wrap_cut_repair, wrap_density_ratio
    from app.select import (
        WRAP_CUT_REFILL_MAX,
        WRAP_CUT_VISUAL_MAX,
        WRAP_DENSITY_MAX,
        WRAP_DENSITY_MIN,
        WRAP_DENSITY_SPARSE_MIN,
    )

    def _oot(reserve: float = 2.0) -> bool:
        return deadline is not None and time.perf_counter() >= float(deadline) - reserve

    if _oot(2.0):
        return canvas

    import os as _os_heal
    if _os_heal.environ.get("SEAMLESS_HEAL_TRACE"):
        import traceback
        chain = []
        for fr in traceback.extract_stack(limit=16)[:-1]:
            if "processor.py" in fr.filename:
                chain.append(f"{fr.name}:{fr.lineno}")
        print(f"  heal-stack {' <- '.join(chain[-6:])}", flush=True)

    view0 = _stamp_flat_view(canvas, bg, threshold)
    dens0 = wrap_density_ratio(view0)
    cut0 = wrap_cut_ratio(view0)
    dens_need0 = (
        WRAP_DENSITY_SPARSE_MIN if ink_frac(view0) < 0.22 else WRAP_DENSITY_MIN
    )

    # 目視切圖帶也要進 heal；稀墨 dens 目標對齊 sparse 0.78。
    if cut0 <= WRAP_CUT_VISUAL_MAX and dens0 >= dens_need0:
        return canvas

    def _badness(dens: float, cut: float) -> tuple:
        dens_err = 0
        dens_floor = dens_need0
        if dens < dens_floor:
            dens_err = 2 + int((dens_floor - dens) * 10)
        elif dens > WRAP_DENSITY_MAX:
            dens_err = 2 + int((dens - WRAP_DENSITY_MAX) * 10)
        cut_err = 0
        if cut > WRAP_CUT_REFILL_MAX:
            cut_err = 3 + int(cut)
        elif cut > WRAP_CUT_VISUAL_MAX:
            cut_err = 1 + int((cut - WRAP_CUT_VISUAL_MAX) * 8)
        return (dens_err + cut_err, cut, abs(dens - 0.92))

    best = canvas
    best_dens, best_cut = dens0, cut0
    out = canvas.copy()
    # cut≥2 一輪常殺不完；mild 0.30～0.55 給三輪壓到目視門。時間緊至少兩輪 mild。
    mild0 = WRAP_CUT_VISUAL_MAX < cut0 < 2.0
    if _oot(8.0):
        rounds = 1
    elif mild0 and ink_frac(view0) < 0.42:
        rounds = 3
    elif cut0 >= 2.0 or mild0:
        rounds = 2
    else:
        rounds = 1
    for rnd in range(rounds):
        if _oot(2.5):
            break
        view_r = _stamp_flat_view(out, bg, threshold)
        dens_r = wrap_density_ratio(view_r)
        cut_r = wrap_cut_ratio(view_r)
        dens_need = (
            WRAP_DENSITY_SPARSE_MIN if ink_frac(view_r) < 0.22 else WRAP_DENSITY_MIN
        )
        if cut_r <= WRAP_CUT_VISUAL_MAX and dens_r >= dens_need:
            break
        only_boost = cut_r <= WRAP_CUT_VISUAL_MAX and dens_r < dens_need
        explosive = cut_r >= 2.0
        mild_cut = (not explosive) and cut_r > WRAP_CUT_VISUAL_MAX
        sparse_ink = ink_frac(view_r) < 0.42
        import os as _os_heal
        if _os_heal.environ.get("SEAMLESS_HEAL_TRACE"):
            print(
                f"  heal-rnd{rnd} cut={cut_r:.3f} dens={dens_r:.3f} "
                f"expl={explosive} mild={mild_cut} only_boost={only_boost}",
                flush=True,
            )
        if not only_boost:
            rep = wrap_cut_repair(view_r)
            if rep.kill_ids:
                kill = np.isin(rep.labels, np.asarray(rep.kill_ids, dtype=np.int32))
                ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                kill = cv2.dilate(kill.astype(np.uint8), ker).astype(bool)
                if plate is None:
                    plate = background_plate(
                        out, _stamp_foreground(out, bg, threshold).astype(bool)
                    )
                out = fill_kill_with_plate(out, kill, plate)
        view_k = _stamp_flat_view(out, bg, threshold)
        dens_k = wrap_density_ratio(view_k)
        cut_k = wrap_cut_ratio(view_k)
        # 殺完切圖已過目視門：禁止大把 place 把半隻貼回來。
        # 爆切圖常見 dens0 已夠 → 殺完 dens 掉到 0.4x；粗補會讓 cut 回到 200%+。
        cut_cleared = cut_k <= WRAP_CUT_VISUAL_MAX and (
            mild_cut or explosive or (sparse_ink and cut_r > WRAP_CUT_VISUAL_MAX)
        )
        if cut_cleared:
            # 切圖已過目視門：立刻把時間留給 dens-jobs 用小章補密度。
            # 舊路徑在 heal 裡連貼 10 顆 hero，65s 後 dens 仍 0.44。
            best = out.copy()
            best_dens, best_cut = dens_k, cut_k
            break

        occ = _stamp_foreground(out, bg, threshold).astype(bool)
        tight = dens_r < dens_need or only_boost or explosive or mild_cut
        if only_boost:
            factor, relax = 0.50, 0.34
        elif explosive:
            factor = 0.48 if rnd == 0 else 0.38
            relax = 0.36 if rnd == 0 else 0.30
        elif mild_cut and dens_r < dens_need + 0.10:
            # dens／cut 互卡：殺切圖同時放寬間距，免得修完變白十字。
            factor, relax = (0.45, 0.32) if rnd == 0 else (0.38, 0.28)
        elif mild_cut:
            # 密度已夠：少貼、先吃掉假接件。
            factor, relax = (0.55, 0.40) if rnd == 0 else (0.42, 0.30)
        elif tight:
            factor, relax = 0.68, 0.55
        else:
            factor, relax = 1.0, 1.0
        # 密花／小間距：hero 太大貼不满十字，改用全模板。
        use_hero = (not explosive) and float(layout.d_nn) >= 28.0
        # 爆切圖／稀疏 mild：限量貼，避免單輪 place 拖時或貼回半隻。
        if explosive or _oot(12.0):
            place_cap = 6
        elif mild_cut and sparse_ink:
            place_cap = 3 if rnd == 0 else 5
        elif mild_cut:
            place_cap = 10
        else:
            place_cap = None
        _place_along_wrap(
            out,
            occ,
            templates,
            replace(
                layout,
                d_nn=max(8.0, float(layout.d_nn) * factor),
                d_min=max(6.0, float(layout.d_min) * factor),
            ),
            np.random.default_rng(17 + rnd * 11),
            prefer_hero=use_hero,
            relax_inner=relax,
            max_place=place_cap,
            deadline=deadline,
        )
        view1 = _stamp_flat_view(out, bg, threshold)
        dens1 = wrap_density_ratio(view1)
        cut1 = wrap_cut_ratio(view1)
        # 殺切圖後仍過疏：僅在切圖已近目視門時補 dens，避免又貼回半隻。
        dens_cut_ok = (
            cut1 <= WRAP_CUT_VISUAL_MAX + 0.04
            if ink_frac(view1) < 0.42
            else cut1 <= WRAP_CUT_REFILL_MAX + 0.20
        )
        if dens1 < dens_need and dens_cut_ok:
            occ2 = _stamp_foreground(out, bg, threshold).astype(bool)
            _place_along_wrap(
                out,
                occ2,
                templates,
                replace(
                    layout,
                    d_nn=max(8.0, float(layout.d_nn) * 0.34),
                    d_min=max(6.0, float(layout.d_min) * 0.34),
                ),
                np.random.default_rng(23 + rnd * 7),
                prefer_hero=use_hero,
                relax_inner=0.26,
                deadline=deadline,
                max_place=4 if ink_frac(view1) < 0.42 else None,
            )
            view1 = _stamp_flat_view(out, bg, threshold)
            dens1 = wrap_density_ratio(view1)
            cut1 = wrap_cut_ratio(view1)
        if _badness(dens1, cut1) < _badness(best_dens, best_cut):
            best = out.copy()
            best_dens, best_cut = dens1, cut1
        if cut1 <= WRAP_CUT_VISUAL_MAX and dens1 >= dens_need:
            break
        # 第二輪沒明顯進步就停，避免越修越空。
        if rnd > 0 and cut1 >= cut_r * 0.92 and dens1 <= dens_r + 0.03:
            break

    dens1, cut1 = best_dens, best_cut
    # 原稿 dens 已過門檻：heal 後 dens 打穿則拒，除非切圖從爆表／目視帶修回。
    if dens0 >= dens_need0 and dens1 < dens_need0 - 0.02:
        cut_fixed = cut0 > WRAP_CUT_VISUAL_MAX and cut1 <= WRAP_CUT_VISUAL_MAX
        cut_big_drop = cut0 >= 2.0 and cut1 <= max(WRAP_CUT_REFILL_MAX, cut0 * 0.40)
        # 爆切圖殺邊後 dens 可掉到 0.2x；0.40 會把 cut→0 整段丟掉。
        if cut0 >= 2.0 and cut_fixed:
            dens_floor = 0.18
        elif cut_fixed or cut_big_drop:
            dens_floor = 0.58
        else:
            dens_floor = 0.62
        if not (cut_fixed or cut_big_drop) or dens1 < dens_floor:
            import os
            if os.environ.get("SEAMLESS_HEAL_TRACE"):
                print(
                    f"  heal-reject dens0={dens0:.3f} dens1={dens1:.3f} "
                    f"cut0={cut0:.3f} cut1={cut1:.3f} floor={dens_floor} "
                    f"fixed={cut_fixed}",
                    flush=True,
                )
            return canvas
    if _badness(dens1, cut1) < _badness(dens0, cut0):
        import os
        if os.environ.get("SEAMLESS_HEAL_TRACE"):
            print(
                f"  heal-accept dens {dens0:.3f}->{dens1:.3f} "
                f"cut {cut0:.3f}->{cut1:.3f} bad {_badness(dens0,cut0)}->{_badness(dens1,cut1)}",
                flush=True,
            )
        return best
    import os
    if os.environ.get("SEAMLESS_HEAL_TRACE"):
        print(
            f"  heal-keep dens {dens0:.3f}/{dens1:.3f} cut {cut0:.3f}/{cut1:.3f} "
            f"bad {_badness(dens0,cut0)} vs {_badness(dens1,cut1)}",
            flush=True,
        )
    return canvas


def _clear_and_refill(
    arr: np.ndarray,
    bg: Sequence[int],
    threshold: float,
    margin_px: int,
    deadline: float | None = None,
) -> np.ndarray | None:
    """
    清掉碰到畫框的圖章群組，再用完整圖案在環帶上藍噪聲貼回。

    優先只動邊緣一圈；填不均勻則擴大環帶，再不行整張重排。
    圖章不鏡射、不旋轉。格紋點綴先用週期拷貝填回底紋。

    deadline：牆鐘截止；逾時跳過整張重排／多次種子重試。
    """
    from app.quality import edge_void_ratio

    global _LAST_REFILL_TAG
    _LAST_REFILL_TAG = "清邊補花"
    del margin_px

    def _out_of_time(reserve: float = 3.0) -> bool:
        return deadline is not None and time.perf_counter() >= float(deadline) - reserve

    # 已幾乎沒時間：別開清邊補花，交給其它基底／choose。
    if _out_of_time(18.0):
        return None

    def _tick(msg: str) -> None:
        # 診斷用：僅在設了 SEAMLESS_REFILL_TRACE 時輸出。
        import os
        if os.environ.get("SEAMLESS_REFILL_TRACE"):
            print(f"  refill+{time.perf_counter():.1f} {msg}", flush=True)

    _tick("enter")
    overlay = _overlay_stamp_mask(arr, bg, threshold)
    fg = _stamp_foreground(arr, bg, threshold)
    touch = _touch_margin_px(fg)
    _tick(f"touch={touch}")
    # 散點星：15px 帶會把整圈星都當碰框，補回去縫上過密。
    if overlay is not None and _luma_square_grid_pitch(arr) is None:
        touch = 2
    motifs = extract_interior_motifs(arr, bg, threshold, touch)
    _tick(f"motifs={len(motifs)}")
    if len(motifs) < 2:
        return None
    templates = _complete_templates(motifs)
    _tick(f"templates={len(templates)}")
    if len(templates) < 2 or touch >= 8:
        templates = motifs
    typ = _hero_typical_area(templates)
    min_area = max(80, int(typ * 0.03))
    areas_all = np.array([m.area for m in motifs], dtype=np.float64)
    if areas_all.size:
        hero = typ
        smalls = [m for m in motifs if 80 <= m.area < hero * 0.28]
        mixed = float(np.percentile(areas_all, 90)) > 6.0 * max(
            float(np.percentile(areas_all, 50)), 1.0
        )
        if mixed and len(smalls) >= 8:
            templates = list(templates) + smalls
            min_area = min(
                min_area,
                max(80, int(np.percentile([m.area for m in smalls], 20) * 0.4)),
            )
    h, w = arr.shape[:2]
    edge = _edge_band_mask(h, w, touch)
    _gmap, groups, labels, stats, _cents = group_components_into_motifs(
        fg, min_area, edge=edge
    )
    _tick(f"groups={len(groups)}")
    if _out_of_time(4.0):
        return None
    attach_group_colors(arr, labels, stats, groups)
    layout = layout_stats_from_groups(groups, h, w, interior_only=True)
    _tick(f"layout dnn={layout.d_nn:.1f}")
    ring_place = max(float(touch), 1.2 * layout.span)
    cap = 0.22 * min(h, w)
    ring_place = float(min(ring_place, cap))
    ring_clear = ring_place
    if layout.d_nn >= 16.0:
        # 第一圈若離 wrap 近到無法再貼跨縫件，會留下「貼在縫上卻不跨」的圖章。
        ring_clear = max(ring_place, min(0.62 * layout.d_nn, cap))
        ring_place = max(ring_place, ring_clear)
    # 中大圖章：0.62×d_nn 清太深，補回後不是擠在縫上就是修切圖修到過疏。
    if layout.span >= 0.10 * min(h, w):
        ring_clear = min(
            ring_clear,
            max(float(touch) + 4.0, 0.08 * min(h, w), 0.42 * float(layout.span)),
        )
        ring_place = max(ring_place, ring_clear)
    plate = None if overlay is not None else background_plate(arr, fg.astype(bool))
    _tick("plate")

    def _pick_layout(cleaned: np.ndarray, occ: np.ndarray, full: bool, ring_w: float):
        best = None
        best_key = None
        # 整張多種子各跑 heal+boost 極慢；時間緊縮到 1～2 顆，且先評原圖再只修最佳。
        if _out_of_time(4.0):
            seeds = (42,)
        elif full:
            seeds = (42,) if _out_of_time(22.0) else (42, 55)
        else:
            seeds = (42,)
        raws: list[tuple[np.ndarray, tuple]] = []
        for seed in seeds:
            if _out_of_time(2.0) and seed != 42:
                break
            cand = relayout_ring(
                cleaned,
                templates,
                layout,
                ring_w,
                seed,
                occupied=occ.copy(),
                full=full,
            )
            raws.append((cand, _score_refill(cand, bg, threshold, src=arr)))
        raws.sort(key=lambda x: x[1])
        # 時間緊：只修最佳一顆；不夠時間則直接交 raw。
        n_fix = 0 if _out_of_time(8.0) else (1 if full else 0)
        for cand0, _rk in raws[: max(1, n_fix) if n_fix else 0]:
            cand = cand0
            if full and templates and not _out_of_time(4.0):
                cand = _heal_wrap_cuts(
                    cand, bg, threshold, templates, layout, plate, deadline=deadline
                )
                cand = _boost_wrap_fill(
                    cand, bg, threshold, templates, layout, deadline=deadline
                )
            key = _score_refill(cand, bg, threshold, src=arr)
            if best is None or key < best_key:
                best, best_key = cand, key
            if key[0] == 0:
                return cand, True
        if best is None and raws:
            best, best_key = raws[0][0], raws[0][1]
        return best, bool(best_key is not None and best_key[0] == 0)

    def _finish(cand: np.ndarray | None) -> tuple[np.ndarray | None, bool]:
        if cand is None:
            return None, False
        # 已過關（含邊帶）就不要再進 repair——會把剛補好的跨縫件拆掉。
        if (
            _refill_wrap_is_ok(cand, bg, threshold)
            and edge_void_ratio(cand) <= 0.26
        ):
            cand = _close_refill_seams(cand, bg, threshold, plate)
            return cand, _refill_wrap_is_ok(cand, bg, threshold)
        # 時間緊：跳過昂貴 repair，直接交卷。
        if _out_of_time(4.0):
            cand = _close_refill_seams(cand, bg, threshold, plate)
            return cand, _refill_wrap_is_ok(cand, bg, threshold)
        before_fix = cand.copy()
        cand, _ok = _maybe_repair_wrap(
            arr,
            cand,
            bg,
            threshold,
            templates,
            min_area,
            layout=layout,
            plate=plate,
            deadline=deadline,
        )
        cand = _keep_better_refill(before_fix, cand, bg, threshold, src=arr)
        cand = _close_refill_seams(cand, bg, threshold, plate)
        if cand is None:
            return None, False
        return cand, _refill_wrap_is_ok(cand, bg, threshold)

    def _wipe_full() -> tuple[np.ndarray | None, bool]:
        """地色整張重鋪：只清環帶留不住構圖框，2×2 仍是九宮格。"""
        wiped = plate.copy() if plate is not None else background_plate(
            arr, fg.astype(bool)
        )
        occ0 = np.zeros((h, w), dtype=bool)
        cand, _ = _pick_layout(wiped, occ0, True, ring_place)
        cand, okw = _finish(cand)
        return cand, okw

    from app.quality import wrap_cut_ratio as _wcr_src, wrap_density_ratio as _wdr_src

    n_fg, lab_fg, st_fg, _ = cv2.connectedComponentsWithStats(
        fg.astype(np.uint8), 8
    )
    hero_min = max(80, int(float(typ) * 0.28))
    hero_arr = arr.copy()
    bg_px = np.asarray(bg, dtype=np.uint8)
    for i in range(1, n_fg):
        if int(st_fg[i, cv2.CC_STAT_AREA]) < hero_min:
            hero_arr[lab_fg == i] = bg_px
    hero_view = _stamp_flat_view(hero_arr, bg, threshold)
    hero_dens = _wdr_src(hero_view)
    hero_cut = _wcr_src(hero_view)
    band8 = max(8, int(touch))
    has_slivers = False
    for i in range(1, n_fg):
        area = int(st_fg[i, cv2.CC_STAT_AREA])
        if not (80 <= area < hero_min):
            continue
        x0 = int(st_fg[i, cv2.CC_STAT_LEFT])
        y0 = int(st_fg[i, cv2.CC_STAT_TOP])
        bw = int(st_fg[i, cv2.CC_STAT_WIDTH])
        bh = int(st_fg[i, cv2.CC_STAT_HEIGHT])
        touches = (
            x0 < band8
            or y0 < band8
            or x0 + bw > w - band8
            or y0 + bh > h - band8
        )
        if touches and (bw <= 8 or bh <= 8) and area >= 200:
            has_slivers = True
            break
    all_dens = _wdr_src(_stamp_flat_view(arr, bg, threshold))
    # 主圖章停在畫心、十字帶真的疏：整張重鋪。
    # ATT 聖誕樹／雪花原稿 2×2 已經是完整構圖，全墨密度 ≥0.80，
    # 舊閘 hero_dens<0.95 會誤判成九宮格，打散原稿。
    framed_stamps = (
        overlay is None
        and all_dens < 0.70
        and hero_dens < 0.70
        and hero_cut <= 0.42
        and not has_slivers
        and (
            _has_separated_stamps(arr, bg, threshold)
            or _looks_like_discrete_motifs(arr, bg, threshold)
        )
    )
    if framed_stamps and not _out_of_time(20.0):
        wipe, _okw = _wipe_full()  # framed empty wrap
        if wipe is not None:
            wipe = _heal_wrap_cuts(
                wipe, bg, threshold, templates, layout, plate, deadline=deadline
            )
            wipe = _boost_wrap_fill(
                wipe, bg, threshold, templates, layout, deadline=deadline
            )
            wipe = _seal_wrap_background(wipe, bg, threshold, plate)
            from app.select import (
                WRAP_CUT_REFILL_MAX as _CUT_FR,
                WRAP_DENSITY_MAX as _DMAX_FR,
                WRAP_DENSITY_MIN as _DMIN_FR,
            )

            dens_fr = _wdr_src(_stamp_flat_view(wipe, bg, threshold))
            cut_fr = _wcr_src(_stamp_flat_view(wipe, bg, threshold))
            # 舊路徑無條件 return，常交出 dens 好看但 cut≈7 的假解。
            if (
                _DMIN_FR <= dens_fr <= _DMAX_FR
                and cut_fr <= _CUT_FR
                and edge_void_ratio(wipe) <= 0.26
            ):
                _LAST_REFILL_TAG = "清邊補花（整張重排）"
                return wipe
            # 結構未過：別提早交卷，落到後面環帶清邊／dens 補。
    elif framed_stamps:
        # 時間不夠就不整張重排，改走下面環帶清邊。
        pass
    _tick(f"framed={framed_stamps}")

    if overlay is not None:
        jobs = _stamp_component_jobs(overlay, touch, min_area)
        if not any(job[3] for job in jobs):
            return None
        # 格紋點綴：整張清掉再重貼。散點星疊在水彩上沒有格子可拷，
        # 全清會把星堆到縫上（過密／切圖）。
        if _luma_square_grid_pitch(arr) is not None:
            cleaned = _fill_overlay_with_lattice(arr, overlay)
            occ = np.zeros(cleaned.shape[:2], dtype=bool)
            filled, ok = _pick_layout(cleaned, occ, True, ring_place)
            if filled is None:
                return None
            snapped = refill_with_wrapped_motifs(
                cleaned,
                bg,
                threshold,
                templates,
                jobs,
                seed=42,
                occupied=np.zeros(cleaned.shape[:2], dtype=bool),
            )
            if _score_refill(snapped, bg, threshold, src=arr) < _score_refill(
                filled, bg, threshold, src=arr
            ):
                filled = snapped
            filled, ok = _finish(filled)
            if not ok:
                _LAST_REFILL_TAG = "清邊補花（整張重排）"
        else:
            cleaned, occ, ov_jobs = _erase_touching_overlay(
                arr, overlay, touch, min_area
            )
            edge_jobs = [j for j in ov_jobs if j[3]]
            filled = cleaned
            if edge_jobs:
                filled = refill_with_wrapped_motifs(
                    cleaned,
                    bg,
                    threshold,
                    templates,
                    edge_jobs,
                    seed=42,
                    occupied=occ.copy(),
                )
            filled, ok = _finish(filled)
            if not ok:
                dart, _ = _pick_layout(cleaned, occ, False, ring_place)
                dart, dart_ok = _finish(dart)
                if dart is not None and (
                    filled is None
                    or _score_refill(dart, bg, threshold, src=arr)
                    < _score_refill(filled, bg, threshold, src=arr)
                ):
                    filled, ok = dart, dart_ok
            if not ok:
                leftover, more = _erase_unwrapped_frame(
                    filled if filled is not None else cleaned,
                    bg,
                    threshold,
                    min_area,
                )
                if more:
                    occ_l = _stamp_foreground(leftover, bg, threshold).astype(bool)
                    snap = refill_with_wrapped_motifs(
                        leftover,
                        bg,
                        threshold,
                        templates,
                        more,
                        seed=42,
                        occupied=occ_l,
                    )
                    snap, ok2 = _finish(snap)
                    if snap is not None and (
                        filled is None
                        or _score_refill(snap, bg, threshold, src=arr)
                        < _score_refill(filled, bg, threshold, src=arr)
                    ):
                        filled, ok = snap, ok2
    else:
        _tick("erase_edge begin")
        cleaned, removed = remove_edge_touching_components(
            arr,
            bg,
            threshold,
            touch,
            min_area=min_area,
            ring_px=int(round(ring_clear)),
            plate=plate,
        )
        _tick(f"erase_edge removed={len(removed)}")
        if _out_of_time(3.0):
            return _close_refill_seams(cleaned, bg, threshold, plate)
        cleaned, extra = _erase_interior_fragments(
            cleaned, bg, threshold, touch, templates, min_area=min_area
        )
        _tick(f"erase_frag extra={len(extra)}")
        if _out_of_time(3.0):
            return _close_refill_seams(cleaned, bg, threshold, plate)
        inner = [
            j
            for j in (removed + extra)
            if len(j) > 3 and not bool(j[3])
        ]
        if inner:
            cleaned = refill_with_wrapped_motifs(
                cleaned, bg, threshold, templates, inner, seed=42
            )
            _tick("refill_inner")
        occ = _stamp_foreground(cleaned, bg, threshold).astype(bool)
        filled = cleaned
        if removed:
            _tick("refill_removed begin")
            filled = refill_with_wrapped_motifs(
                cleaned, bg, threshold, templates, removed, seed=42, occupied=occ.copy()
            )
            _tick("refill_removed done")
        if _out_of_time(3.0):
            return _close_refill_seams(filled, bg, threshold, plate)
        _tick("finish begin")
        filled, ok = _finish(filled)
        _tick(f"finish ok={ok}")
        # 稀疏切圖還在目視帶：優先再 heal 一輪，勿直接交 0.43 假過關。
        if (
            filled is not None
            and templates
            and not ok
            and not _out_of_time(5.0)
        ):
            from app.quality import ink_frac as _ink_m, wrap_cut_ratio as _wcr_m
            from app.select import WRAP_CUT_VISUAL_MAX as _CUT_VIS_M

            vm = _stamp_flat_view(filled, bg, threshold)
            if _ink_m(vm) < 0.42 and _wcr_m(vm) > _CUT_VIS_M:
                filled = _heal_wrap_cuts(
                    filled, bg, threshold, templates, layout, plate, deadline=deadline
                )
                filled = _boost_wrap_fill(
                    filled, bg, threshold, templates, layout, deadline=deadline
                )
                ok = _refill_wrap_is_ok(filled, bg, threshold)
                _tick(f"mild_heal ok={ok}")
        # 第一輪後若時間所剩無幾，勿再 erase／整圈重排（常再吃 30s+）。
        if _out_of_time(10.0):
            return _close_refill_seams(filled, bg, threshold, plate)
        if not ok:
            leftover, more = _erase_unwrapped_frame(
                filled if filled is not None else cleaned,
                bg,
                threshold,
                min_area,
            )
            if more:
                occ_l = _stamp_foreground(leftover, bg, threshold).astype(bool)
                snap = refill_with_wrapped_motifs(
                    leftover,
                    bg,
                    threshold,
                    templates,
                    more,
                    seed=42,
                    occupied=occ_l,
                )
                snap, ok2 = _finish(snap)
                if snap is not None and (
                    filled is None
                    or _score_refill(snap, bg, threshold, src=arr)
                    < _score_refill(filled, bg, threshold, src=arr)
                ):
                    filled, ok = snap, ok2
        if not ok:
            dart, _ = _pick_layout(cleaned, occ, False, ring_place)
            dart, dart_ok = _finish(dart)
            if dart is not None and (
                filled is None
                or _score_refill(dart, bg, threshold, src=arr)
                < _score_refill(filled, bg, threshold, src=arr)
            ):
                filled, ok = dart, dart_ok
        if filled is None:
            leftover, more = _erase_unwrapped_frame(arr, bg, threshold, min_area)
            if not more:
                return None
            occ_l = _stamp_foreground(leftover, bg, threshold).astype(bool)
            filled, _ = _pick_layout(leftover, occ_l, False, ring_place)
            filled, ok = _finish(filled)
            if filled is None:
                return None
        if not ok:
            ring2 = min(ring_place * 1.8, 0.32 * min(h, w))
            cleaned2, rem2 = remove_edge_touching_components(
                arr,
                bg,
                threshold,
                touch,
                min_area=min_area,
                ring_px=int(round(ring2)),
                plate=plate,
            )
            occ2 = _stamp_foreground(cleaned2, bg, threshold).astype(bool)
            filled2 = cleaned2
            if rem2:
                filled2 = refill_with_wrapped_motifs(
                    cleaned2,
                    bg,
                    threshold,
                    templates,
                    rem2,
                    seed=42,
                    occupied=occ2.copy(),
                )
            filled2, ok2 = _finish(filled2)
            if not ok2:
                dart2, _ = _pick_layout(cleaned2, occ2, False, ring2)
                dart2, ok2d = _finish(dart2)
                if dart2 is not None and (
                    filled2 is None
                    or _score_refill(dart2, bg, threshold, src=arr)
                    < _score_refill(filled2, bg, threshold, src=arr)
                ):
                    filled2, ok2 = dart2, ok2d
            if filled2 is not None and (
                filled is None
                or _score_refill(filled2, bg, threshold, src=arr)
                < _score_refill(filled, bg, threshold, src=arr)
            ):
                filled, ok = filled2, ok2
        if not ok and layout.d_nn >= 90.0 and not _out_of_time(10.0):
            # 大間距圖：整張重排改走後面 plate wipe，避免對 cleaned 再跑一輪多種子。
            pass
        # 環帶補花失敗仍把結果交給閘門，不要在這裡直接丟掉。

    # 時間將盡：交出現有結果，留給 choose；避免硬超時整張作廢。
    if filled is not None and _out_of_time(8.0):
        return _close_refill_seams(filled, bg, threshold, plate)

    # 先微調 dens／void（便宜，逾時也做），再跑昂貴整張重排。
    if (
        filled is not None
        and templates
        and not _refill_wrap_is_ok(filled, bg, threshold)
    ):
        from app.quality import wrap_density_ratio as _wdr_nudge
        from app.quality import wrap_cut_ratio as _wcr_nudge
        from app.select import WRAP_CUT_REFILL_MAX as _CUT_N
        from app.select import WRAP_DENSITY_MIN as _DMIN_N

        v0 = _stamp_flat_view(filled, bg, threshold)
        d0 = _wdr_nudge(v0)
        c0 = _wcr_nudge(v0)
        vo0 = edge_void_ratio(filled)
        if c0 <= _CUT_N and (d0 < _DMIN_N or vo0 > 0.26):
            before = filled.copy()
            occ = _stamp_foreground(filled, bg, threshold).astype(bool)
            factor = 0.48 if d0 < 0.60 or vo0 > 0.40 else 0.55
            # 0.66～0.70：差一點，大幅放寬畫心間距硬補。
            if 0.66 <= d0 < _DMIN_N:
                relax = 0.36
                factor = min(factor, 0.40)
            elif 0.60 <= d0 < _DMIN_N:
                relax = 0.48
                factor = min(factor, 0.48)
            else:
                relax = 0.72 if d0 < _DMIN_N else 1.0
            _place_along_wrap(
                filled,
                occ,
                templates,
                replace(
                    layout,
                    d_nn=max(8.0, float(layout.d_nn) * factor),
                    d_min=max(6.0, float(layout.d_min) * factor),
                ),
                np.random.default_rng(103),
                prefer_hero=True,
                relax_inner=relax,
            )
            # 仍差一點再補一次
            d1 = _wdr_nudge(_stamp_flat_view(filled, bg, threshold))
            if d1 < _DMIN_N and _wcr_nudge(_stamp_flat_view(filled, bg, threshold)) <= _CUT_N:
                occ2 = _stamp_foreground(filled, bg, threshold).astype(bool)
                _place_along_wrap(
                    filled,
                    occ2,
                    templates,
                    replace(
                        layout,
                        d_nn=max(8.0, float(layout.d_nn) * 0.38),
                        d_min=max(6.0, float(layout.d_min) * 0.38),
                    ),
                    np.random.default_rng(107),
                    prefer_hero=float(layout.d_nn) >= 28.0,
                    relax_inner=0.32,
                )
            if _score_refill(filled, bg, threshold, src=arr) > _score_refill(
                before, bg, threshold, src=arr
            ):
                # dens 已跨門檻且切圖仍可接受：留下，勿被次要分打回過疏。
                d2 = _wdr_nudge(_stamp_flat_view(filled, bg, threshold))
                c2 = _wcr_nudge(_stamp_flat_view(filled, bg, threshold))
                if not (
                    d0 < _DMIN_N
                    and d2 >= _DMIN_N
                    and c2 <= _CUT_N + 0.05
                ):
                    filled = before

    if filled is not None and (
        not _refill_wrap_is_ok(filled, bg, threshold)
        or edge_void_ratio(filled) > 0.26
    ) and not _out_of_time(8.0):
        wipe, okw = _wipe_full()
        if wipe is not None:
            wipe = _heal_wrap_cuts(
                wipe, bg, threshold, templates, layout, plate, deadline=deadline
            )
            wipe = _boost_wrap_fill(
                wipe, bg, threshold, templates, layout, deadline=deadline
            )
            wipe = _seal_wrap_background(wipe, bg, threshold, plate)
            from app.quality import wrap_density_ratio as _wdr_w
            from app.quality import wrap_cut_ratio as _wcr_w
            from app.select import (
                WRAP_CUT_REFILL_MAX as _CUT_W,
                WRAP_DENSITY_MAX as _DMAX_W,
                WRAP_DENSITY_MIN as _DMIN_W,
            )

            dens_w = _wdr_w(_stamp_flat_view(wipe, bg, threshold))
            dens_f = _wdr_w(_stamp_flat_view(filled, bg, threshold))
            cut_w = _wcr_w(_stamp_flat_view(wipe, bg, threshold))
            cut_f = _wcr_w(_stamp_flat_view(filled, bg, threshold))
            void_w = edge_void_ratio(wipe)
            void_f = edge_void_ratio(filled)
            better = _score_refill(wipe, bg, threshold, src=arr) < _score_refill(
                filled, bg, threshold, src=arr
            )
            struct_ok = (
                _DMIN_W <= dens_w <= _DMAX_W and cut_w <= _CUT_W
            )
            wipe_ok = struct_ok and void_w <= 0.26
            # 環帶過疏：整張 dens/cut 過關就採用（void 可再補）。
            if struct_ok and dens_f < _DMIN_W:
                better = True
            if wipe_ok and (
                dens_f < _DMIN_W
                or cut_f > _CUT_W
                or void_f > 0.26
                or dens_w >= dens_f + 0.04
            ):
                better = True
            if dens_w < _DMIN_W and dens_f >= dens_w + 0.05:
                better = False
            if dens_w > _DMAX_W:
                better = False
            # 切圖爆表絕不以「dens 較好」換進來（會留下 cut≈7–8 假過疏解）。
            if cut_w > _CUT_W:
                better = False
            if void_w > 0.26 and dens_w < dens_f + 0.08 and dens_f >= _DMIN_W:
                better = False
            if better:
                filled = wipe
                ok = okw or wipe_ok or (struct_ok and dens_f < _DMIN_W)
                _LAST_REFILL_TAG = "清邊補花（整張重排）"
                if edge_void_ratio(filled) > 0.26:
                    filled = _boost_wrap_fill(
                        filled,
                        bg,
                        threshold,
                        templates,
                        layout,
                        deadline=deadline,
                    )
                    filled = _seal_wrap_background(filled, bg, threshold, plate)

    # 邊帶／過疏／錯格／錯色最後一搏：略收步長多貼跨縫件。
    if (
        filled is not None
        and templates
        and not _refill_wrap_is_ok(filled, bg, threshold)
    ):
        from app.quality import (
            wrap_cut_ratio as _wcr_end,
            wrap_density_ratio as _wdr_end,
            wrap_edge_color_break as _ecb_end,
            wrap_gutter_error as _gut_end,
        )
        from app.select import (
            EDGE_COLOR_BREAK_MAX as _ECB_END,
            GUTTER_VISUAL_MAX as _GUT_END,
            WRAP_CUT_REFILL_MAX as _CUT_END,
            WRAP_DENSITY_MIN as _DMIN_END,
        )

        view0 = _stamp_flat_view(filled, bg, threshold)
        dens0 = _wdr_end(view0)
        cut0 = _wcr_end(view0)
        void0 = edge_void_ratio(filled)
        gut0 = _gut_end(view0)
        ecb0 = _ecb_end(view0)
        need_wrap = (
            dens0 < _DMIN_END
            or void0 > 0.26
            or gut0 > _GUT_END
            or ecb0 > _ECB_END
        )
        if cut0 <= _CUT_END and need_wrap:
            before = filled.copy()
            occ = _stamp_foreground(filled, bg, threshold).astype(bool)
            factor = 0.48 if dens0 < 0.60 or void0 > 0.40 or gut0 > _GUT_END else 0.58
            # dens 只差一點：放寬與畫心間距，才能再塞跨縫件。
            relax = 0.62 if dens0 >= 0.62 or gut0 > _GUT_END else 1.0
            if gut0 > _GUT_END or ecb0 > _ECB_END:
                relax = min(relax, 0.50)
                factor = min(factor, 0.50)
            _place_along_wrap(
                filled,
                occ,
                templates,
                replace(
                    layout,
                    d_nn=max(8.0, float(layout.d_nn) * factor),
                    d_min=max(6.0, float(layout.d_min) * factor),
                ),
                np.random.default_rng(101),
                prefer_hero=True,
                relax_inner=relax,
            )
            if _score_refill(filled, bg, threshold, src=arr) > _score_refill(
                before, bg, threshold, src=arr
            ):
                filled = before
            elif _refill_wrap_is_ok(filled, bg, threshold):
                pass
            else:
                # 變好但未過關也留下；變差已退回。
                pass

    if (
        filled is not None
        and templates
        and not _refill_wrap_is_ok(filled, bg, threshold)
        and not _out_of_time(6.0)
    ):
        filled = _heal_wrap_cuts(
            filled, bg, threshold, templates, layout, plate, deadline=deadline
        )
        filled = _boost_wrap_fill(
            filled, bg, threshold, templates, layout, deadline=deadline
        )
        from app.quality import (
            wrap_cut_ratio as _wcr_end,
            wrap_density_ratio as _wdr_end,
            wrap_edge_color_break as _ecb_end,
            wrap_gutter_error as _gut_end,
        )
        from app.select import (
            EDGE_COLOR_BREAK_MAX as _ECB_END,
            GUTTER_VISUAL_MAX as _GUT_END,
            WRAP_CUT_REFILL_MAX as _CUT_END,
            WRAP_DENSITY_MIN as _DMIN_END,
        )

        view_end = _stamp_flat_view(filled, bg, threshold)
        still_cut = _wcr_end(view_end) > _CUT_END
        dens_end = _wdr_end(view_end)
        void_end = edge_void_ratio(filled)
        gut_end = _gut_end(view_end)
        ecb_end = _ecb_end(view_end)
        need_more = (
            dens_end < _DMIN_END
            or void_end > 0.26
            or gut_end > _GUT_END
            or ecb_end > _ECB_END
        )
        if need_more and not still_cut:
            from app.select import WRAP_CUT_VISUAL_MAX as _CUT_VIS_MORE

            occ = _stamp_foreground(filled, bg, threshold).astype(bool)
            factor = 0.55 if dens_end < 0.55 or void_end > 0.40 else 0.72
            if gut_end > _GUT_END or ecb_end > _ECB_END:
                factor = min(factor, 0.52)
            tight = replace(
                layout,
                d_nn=max(8.0, float(layout.d_nn) * factor),
                d_min=max(6.0, float(layout.d_min) * factor),
            )
            before = filled.copy()
            cut_b = _wcr_end(view_end)
            dens_b = dens_end
            _place_along_wrap(
                filled,
                occ,
                templates,
                tight,
                np.random.default_rng(99),
                prefer_hero=True,
                relax_inner=0.48 if (gut_end > _GUT_END or ecb_end > _ECB_END) else 1.0,
                max_place=4 if dens_end < 0.55 else 8,
                deadline=deadline,
            )
            view_n = _stamp_flat_view(filled, bg, threshold)
            cut_n = _wcr_end(view_n)
            dens_n = _wdr_end(view_n)
            dens_help = (
                dens_b < _DMIN_END
                and dens_n >= dens_b + 0.03
                and dens_n <= 1.15
                and cut_n <= max(_CUT_VIS_MORE, cut_b + 0.02)
            )
            if cut_b <= _CUT_VIS_MORE and cut_n > _CUT_VIS_MORE:
                filled = before
            elif (
                _score_refill(filled, bg, threshold, src=arr)
                > _score_refill(before, bg, threshold, src=arr)
            ) and not dens_help:
                filled = before

    # 時間將盡：跳過昂貴尾段。但切圖已過關只剩錯色時仍要修陰陽臉。
    if filled is not None and _out_of_time(5.0):
        from app.quality import (
            wrap_cut_ratio as _wcr_t,
            wrap_edge_color_break as _ecb_t,
        )
        from app.select import (
            EDGE_COLOR_BREAK_MAX as _ECB_T,
            WRAP_CUT_VISUAL_MAX as _CUT_T,
        )

        vt = _stamp_flat_view(filled, bg, threshold)
        only_ecb = (
            templates
            and _wcr_t(vt) <= _CUT_T
            and _ecb_t(vt) > _ECB_T
            and not _out_of_time(1.5)
        )
        if not only_ecb:
            return _close_refill_seams(filled, bg, threshold, plate)

    # 對邊異色（陰陽臉）：清掉假接邊件，用帶色 jobs 跨縫重貼（勿丟 more）。
    # wrap_cut 已低時 erase 常殺不到「兩側各貼半隻異色」，再清最外環帶。
    # 順序：清邊→色匹配跨縫貼→先補 dens→最後才 heal 切圖（避免 dens/cut 互打）。
    if filled is not None and templates:
        from app.quality import wrap_edge_color_break as _ecb_fix
        from app.motif_layout import fill_kill_with_plate
        from app.select import EDGE_COLOR_BREAK_MAX as _ECB_FIX

        view_ecb = _stamp_flat_view(filled, bg, threshold)
        if _ecb_fix(view_ecb) > _ECB_FIX and not _out_of_time(6.0):
            before = filled.copy()
            from app.quality import wrap_cut_ratio as _wcr_ecb0
            from app.select import WRAP_CUT_VISUAL_MAX as _CUT_VIS_ECB

            cut_ecb0 = _wcr_ecb0(view_ecb)
            leftover, more = _erase_unwrapped_frame(
                filled, bg, threshold, min_area
            )
            healed = leftover
            # 切圖已過關卻陰陽臉：幾何已接上，直接把右／下緣異色像素改成左／上緣色。
            if (
                cut_ecb0 <= _CUT_VIS_ECB
                and _ecb_fix(_stamp_flat_view(healed, bg, threshold)) > _ECB_FIX
            ):
                from app.quality import ink_mask as _ink_ecb

                h0, w0 = healed.shape[:2]
                snap_ecb = healed.copy()
                ink2 = _ink_ecb(healed)
                band = max(2, min(3, min(h0, w0) // 80))
                both2 = ink2[:, 0] & ink2[:, -1]
                if int(both2.sum()) >= 12:
                    ys = np.flatnonzero(both2)
                    for t in range(band):
                        m = ink2[ys, t] & ink2[ys, w0 - 1 - t]
                        if m.any():
                            yy = ys[m]
                            healed[yy, w0 - 1 - t] = healed[yy, t]
                both2h = ink2[0, :] & ink2[-1, :]
                if int(both2h.sum()) >= 12:
                    xs = np.flatnonzero(both2h)
                    for t in range(band):
                        m = ink2[t, xs] & ink2[h0 - 1 - t, xs]
                        if m.any():
                            xx = xs[m]
                            healed[h0 - 1 - t, xx] = healed[t, xx]
                vh_e = _stamp_flat_view(healed, bg, threshold)
                if (
                    _wcr_ecb0(vh_e) > _CUT_VIS_ECB
                    or _ecb_fix(vh_e) >= _ecb_fix(view_ecb) * 0.92
                ):
                    healed = snap_ecb
            if _ecb_fix(_stamp_flat_view(healed, bg, threshold)) > _ECB_FIX:
                # 只殺 wrap_cut_repair 標的假接／錯色件，勿整圈 wipe（dens→0.3）。
                from app.quality import wrap_cut_repair as _wcr_ecb

                view_k = _stamp_flat_view(healed, bg, threshold)
                rep_k = _wcr_ecb(view_k)
                kill = None
                if rep_k.kill_ids:
                    kill = np.isin(
                        rep_k.labels, np.asarray(rep_k.kill_ids, dtype=np.int32)
                    )
                    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                    kill = cv2.dilate(kill.astype(np.uint8), ker).astype(bool)
                if kill is None or not kill.any():
                    # 後備：只清最外 1～2px 細邊，保留環帶圖章本體。
                    h0, w0 = healed.shape[:2]
                    band = max(2, min(4, min(h0, w0) // 200))
                    edge_kill = np.zeros((h0, w0), dtype=bool)
                    edge_kill[:band, :] = True
                    edge_kill[h0 - band :, :] = True
                    edge_kill[:, :band] = True
                    edge_kill[:, w0 - band :] = True
                    fg_e = _stamp_foreground(healed, bg, threshold).astype(bool)
                    kill = edge_kill & fg_e
                if kill is not None and kill.any():
                    use_plate = plate
                    if use_plate is None:
                        use_plate = background_plate(
                            healed,
                            _stamp_foreground(healed, bg, threshold).astype(bool),
                        )
                    healed = fill_kill_with_plate(healed, kill, use_plate)
            # 先用 erase 帶回的色相 jobs 跨縫貼回。
            # more 常是對邊各一截異色——必須 collapse 成一筆，否則再造陰陽臉。
            # erase 後 ecb 已過關但 dens／void 崩了：仍要用 collapse 後的 jobs 補回。
            if more:
                from app.quality import (
                    edge_void_ratio as _void_ecb,
                    wrap_density_ratio as _wdr_ecb0,
                )
                from app.select import WRAP_DENSITY_MIN as _DMIN_ECB0

                view_ae = _stamp_flat_view(healed, bg, threshold)
                ecb_after = _ecb_fix(view_ae)
                dens_after = _wdr_ecb0(view_ae)
                void_after = _void_ecb(healed)
                h_m, w_m = healed.shape[:2]
                more_use = _collapse_opposite_edge_jobs(more, h_m, w_m)
                need_put = (
                    ecb_after > _ECB_FIX
                    or dens_after < _DMIN_ECB0
                    or void_after > 0.26
                )
                if more_use and need_put:
                    occ_jobs = _stamp_foreground(healed, bg, threshold).astype(bool)
                    h_j, w_j = occ_jobs.shape
                    band_j = max(8, min(h_j, w_j) // 30)
                    occ_jobs[:band_j, :] = False
                    occ_jobs[h_j - band_j :, :] = False
                    occ_jobs[:, :band_j] = False
                    occ_jobs[:, w_j - band_j :] = False
                    nxt_put = refill_with_wrapped_motifs(
                        healed,
                        bg,
                        threshold,
                        templates,
                        more_use,
                        seed=113,
                        occupied=occ_jobs,
                    )
                    ecb_put = _ecb_fix(_stamp_flat_view(nxt_put, bg, threshold))
                    # 貼回不得把剛清掉的錯色弄回來；ecb 變差就放棄這輪 more。
                    if ecb_put <= max(_ECB_FIX, ecb_after + 0.5):
                        healed = nxt_put
                    elif ecb_after > _ECB_FIX and ecb_put < ecb_after * 0.85:
                        healed = nxt_put
            from app.quality import wrap_cut_ratio as _wcr_p
            from app.quality import wrap_density_ratio as _wdr_p
            from app.select import WRAP_CUT_REFILL_MAX as _CUT_P
            from app.select import WRAP_DENSITY_MIN as _DMIN_P
            from app.select import WRAP_DENSITY_MAX as WRAP_DENSITY_MAX

            # 色匹配貼回後：勿再盲 place（對邊空時亂抽會再造陰陽臉）。
            # dens 交給 _boost_wrap_fill（已鎖色＋護 ecb）。
            healed = _boost_wrap_fill(
                healed, bg, threshold, templates, layout, deadline=deadline
            )
            # boost 若又把錯色貼回：再 erase＋色匹配一輪（不再盲 place）。
            if _ecb_fix(_stamp_flat_view(healed, bg, threshold)) > _ECB_FIX:
                leftover2, more2 = _erase_unwrapped_frame(
                    healed, bg, threshold, min_area
                )
                if more2:
                    h2s, w2s = leftover2.shape[:2]
                    more2 = _collapse_opposite_edge_jobs(more2, h2s, w2s)
                    occ2 = _stamp_foreground(leftover2, bg, threshold).astype(bool)
                    h2, w2 = occ2.shape
                    b2 = max(8, min(h2, w2) // 30)
                    occ2[:b2, :] = False
                    occ2[h2 - b2 :, :] = False
                    occ2[:, :b2] = False
                    occ2[:, w2 - b2 :] = False
                    nxt2 = refill_with_wrapped_motifs(
                        leftover2,
                        bg,
                        threshold,
                        templates,
                        more2,
                        seed=131,
                        occupied=occ2,
                    )
                    ecb_n2 = _ecb_fix(_stamp_flat_view(nxt2, bg, threshold))
                    ecb_h0 = _ecb_fix(_stamp_flat_view(healed, bg, threshold))
                    if ecb_n2 <= _ECB_FIX or ecb_n2 < ecb_h0 * 0.85:
                        healed = nxt2
                        healed = _boost_wrap_fill(
                            healed, bg, threshold, templates, layout, deadline=deadline
                        )
            vh0 = _stamp_flat_view(healed, bg, threshold)
            dens_p = _wdr_p(vh0)
            cut_p = _wcr_p(vh0)
            ecb_p = _ecb_fix(vh0)
            if cut_p > _CUT_P and dens_p >= 0.58:
                nxt = _heal_wrap_cuts(
                    healed, bg, threshold, templates, layout, plate, deadline=deadline
                )
                dens_n = _wdr_p(_stamp_flat_view(nxt, bg, threshold))
                ecb_n = _ecb_fix(_stamp_flat_view(nxt, bg, threshold))
                cut_n = _wcr_p(_stamp_flat_view(nxt, bg, threshold))
                # dens 必須仍靠近門檻；切圖有改善才收。
                if (
                    dens_n >= max(0.68, dens_p * 0.88)
                    and ecb_n <= max(_ECB_FIX, ecb_p)
                    and cut_n <= cut_p + 0.05
                ):
                    healed = nxt
                    healed = _boost_wrap_fill(
                        healed, bg, threshold, templates, layout, deadline=deadline
                    )
            from app.quality import wrap_density_ratio as _wdr_e

            vb = _stamp_flat_view(before, bg, threshold)
            vh = _stamp_flat_view(healed, bg, threshold)
            ecb_b = _ecb_fix(vb)
            ecb_h = _ecb_fix(vh)
            dens_h = _wdr_e(vh)
            # 錯色明顯下降就採用（分數常因暫疏退回陰陽臉）；之後再補 dens。
            adopt = (
                _score_refill(healed, bg, threshold, src=arr)
                <= _score_refill(before, bg, threshold, src=arr)
                or (ecb_h <= _ECB_FIX and dens_h >= 0.30)
                or (
                    ecb_h < ecb_b * 0.72
                    and ecb_h <= max(_ECB_FIX * 1.15, ecb_b - 6.0)
                    and dens_h >= 0.40
                )
            )
            if adopt:
                filled = healed
                from app.quality import ink_frac as _ink_ad

                # 目標略高於門檻；稀墨對齊 sparse 0.78。
                dens_target = (
                    0.82
                    if _ink_ad(_stamp_flat_view(filled, bg, threshold)) < 0.22
                    else max(_DMIN_P + 0.06, 0.76)
                )
                for seed_i, fac in enumerate(
                    (0.50, 0.42, 0.34, 0.28, 0.22)
                ):
                    if _out_of_time(3.0):
                        break
                    dens_now = _wdr_e(_stamp_flat_view(filled, bg, threshold))
                    if dens_now >= dens_target:
                        break
                    if dens_now > WRAP_DENSITY_MAX:
                        break
                    if _ecb_fix(_stamp_flat_view(filled, bg, threshold)) > _ECB_FIX:
                        break
                    before2 = filled.copy()
                    occ2 = _stamp_foreground(filled, bg, threshold).astype(bool)
                    n_put = _place_along_wrap(
                        filled,
                        occ2,
                        templates,
                        replace(
                            layout,
                            d_nn=max(8.0, float(layout.d_nn) * fac),
                            d_min=max(6.0, float(layout.d_min) * fac),
                        ),
                        np.random.default_rng(127 + seed_i),
                        prefer_hero=True,
                        relax_inner=0.22,
                        max_place=1,
                    )
                    if n_put <= 0:
                        continue
                    from app.quality import wrap_cut_ratio as _wcr_d
                    from app.select import (
                        WRAP_CUT_REFILL_MAX as _CUT_D,
                        WRAP_DENSITY_MAX as _DMAX_D,
                    )

                    cut_now = _wcr_d(_stamp_flat_view(filled, bg, threshold))
                    dens_put = _wdr_e(_stamp_flat_view(filled, bg, threshold))
                    if dens_put > _DMAX_D:
                        filled = before2
                        break
                    if cut_now > _CUT_D:
                        # 切圖爆了：優先退回本輪貼花，勿 heal 打穿 dens。
                        if dens_put < dens_now + 0.04:
                            filled = before2
                            continue
                        nxt = _heal_wrap_cuts(
                            filled,
                            bg,
                            threshold,
                            templates,
                            layout,
                            plate,
                            deadline=deadline,
                        )
                        dens_h2 = _wdr_e(_stamp_flat_view(nxt, bg, threshold))
                        if dens_h2 >= max(0.68, dens_now * 0.90) and dens_h2 <= _DMAX_D:
                            filled = nxt
                        else:
                            filled = before2
                            continue
                    if (
                        _ecb_fix(_stamp_flat_view(filled, bg, threshold))
                        > _ECB_FIX
                    ):
                        filled = before2
                        break
                # 尾段仍略疏：再 boost 一次（不再進 cut heal）。
                dens_fin = _wdr_e(_stamp_flat_view(filled, bg, threshold))
                if dens_fin < dens_target and dens_fin <= WRAP_DENSITY_MAX:
                    filled = _boost_wrap_fill(
                        filled, bg, threshold, templates, layout, deadline=deadline
                    )

    # 切圖仍偏高（含目視帶 0.28～0.60）或 dens 差一點：再強制一輪。
    # 舊門檻 cut>1.5 會漏掉蝴蝶類 cut≈0.30／dens≈0.72。
    if filled is not None and templates and not _out_of_time(4.0):
        from app.quality import wrap_cut_ratio as _wcr_fin
        from app.quality import wrap_density_ratio as _wdr_fin
        from app.quality import wrap_edge_color_break as _ecb_fin
        from app.select import (
            EDGE_COLOR_BREAK_MAX as _ECB_FIN,
            WRAP_CUT_REFILL_MAX as _CUT_FIN,
            WRAP_CUT_VISUAL_MAX as _CUT_VIS,
            WRAP_DENSITY_MIN as _DMIN_FIN,
            WRAP_DENSITY_SPARSE_MIN as _DMIN_SP,
        )
        from app.quality import ink_frac as _ink_fin

        v_fin = _stamp_flat_view(filled, bg, threshold)
        cut_fin = _wcr_fin(v_fin)
        dens_fin0 = _wdr_fin(v_fin)
        ecb_fin0 = _ecb_fin(v_fin)
        ink_fin = _ink_fin(v_fin)
        sparse_need = ink_fin < 0.22 and dens_fin0 < _DMIN_SP
        # 稀疏必須壓到目視 0.30；密墨仍用補花 0.60。
        cut_target = _CUT_VIS if ink_fin < 0.42 else _CUT_FIN
        need_cut = cut_fin > cut_target
        need_dens = dens_fin0 < (_DMIN_SP if sparse_need else _DMIN_FIN)
        if need_cut or need_dens:
            before_fin = filled.copy()
            healed_fin = filled
            if need_cut:
                healed_fin = _heal_wrap_cuts(
                    healed_fin,
                    bg,
                    threshold,
                    templates,
                    layout,
                    plate,
                    deadline=deadline,
                )
            # 無論剛修切圖或本來就疏：一律 boost，避免 cut 修好 dens 掉下去。
            healed_fin = _boost_wrap_fill(
                healed_fin, bg, threshold, templates, layout, deadline=deadline
            )
            vh = _stamp_flat_view(healed_fin, bg, threshold)
            cut_h = _wcr_fin(vh)
            dens_h = _wdr_fin(vh)
            ecb_h = _ecb_fin(vh)
            score_ok = _score_refill(
                healed_fin, bg, threshold, src=arr
            ) <= _score_refill(before_fin, bg, threshold, src=arr)
            cut_help = (not need_cut) or (
                cut_h <= cut_target
                and (cut_h <= cut_fin + 0.02 or cut_h < cut_fin * 0.85)
            )
            # 爆切圖殺邊後 dens 常掉到 0.4x；若仍用 dens_fin0-0.10（≈0.87）當地板，
            # 會把 cut 3.9→0 的結果整段打回原圖（見 SEAMLESS_HEAL_TRACE）。
            cut_fixed_explosive = (
                need_cut and cut_fin >= 2.0 and cut_h <= cut_target
            )
            if cut_fixed_explosive:
                dens_ok = dens_h >= 0.18
            else:
                dens_ok = dens_h >= max(0.62, dens_fin0 - 0.10)
            dens_help = (
                need_dens and dens_h >= dens_fin0 + 0.03 and cut_h <= cut_target
            )
            ecb_ok = ecb_h <= max(_ECB_FIN, ecb_fin0 + 1.0)
            # 爆切圖修好：即使 score 因 void／過疏變差也收下，再靠後續 boost。
            accept = (score_ok or dens_help or cut_fixed_explosive) and cut_help and dens_ok and ecb_ok
            import os as _os_fin
            if _os_fin.environ.get("SEAMLESS_HEAL_TRACE"):
                print(
                    f"  finish-gate cut_fin={cut_fin:.3f} cut_h={cut_h:.3f} "
                    f"dens_fin={dens_fin0:.3f} dens_h={dens_h:.3f} "
                    f"score_ok={score_ok} cut_help={cut_help} dens_ok={dens_ok} "
                    f"ecb_ok={ecb_ok} expl_fix={cut_fixed_explosive} accept={accept} "
                    f"ecb {ecb_fin0:.1f}->{ecb_h:.1f}",
                    flush=True,
                )
            if accept:
                filled = healed_fin
                dens_h2 = _wdr_fin(_stamp_flat_view(filled, bg, threshold))
                if dens_h2 < _DMIN_FIN:
                    filled = _boost_wrap_fill(
                        filled, bg, threshold, templates, layout, deadline=deadline
                    )
                    # boost 不得把切圖貼回來
                    v_b = _stamp_flat_view(filled, bg, threshold)
                    if _wcr_fin(v_b) > cut_target:
                        filled = healed_fin
                filled = _keep_better_refill(
                    before_fin, filled, bg, threshold, src=arr
                )

    return _close_refill_seams(filled, bg, threshold, plate)



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
        Base,
        apply_recipe,
        choose,
        gen_budget_for_shape,
        source_facts,
        source_looks_seamless,
        timed,
    )

    def _lg(msg: str) -> None:
        if log is not None:
            log(msg)

    t_gen = time.perf_counter()
    if bg is None:
        bg = detect_background(image)

    ctx = context_of(image)
    arr = _to_rgb_array(image, bg)
    native = _native_array(image, bg)
    h, w = arr.shape[:2]
    gen_budget = gen_budget_for_shape(h, w)
    # 硬殺 90s；軟截止再提前，確保 choose 前清邊補花會交卷。
    deadline = t_gen + min(float(gen_budget), 52.0)
    _lg(f"  → 時間預算 {gen_budget:.0f}s（截止 {deadline - t_gen:.0f}s）")

    if margin_is_percent:
        margin_px = int(round(min(h, w) * float(margin)))
    else:
        margin_px = int(round(float(margin)))
    margin_px = max(0, margin_px)

    ratio = foreground_ratio(arr, bg, threshold)
    discrete = _looks_like_discrete_motifs(arr, bg, threshold)
    fine_src = _luma_square_grid_pitch(arr)
    src = source_facts(arr, needs_native=ctx.mode == "CMYK")
    _lg(
        f"  → 分類：{'點綴/晶格' if discrete else '滿鋪或稀疏'}，"
        f"前景 {ratio:.0%}，尺寸 {w}×{h}，{ctx.describe()}，{src.rep.describe()}"
    )

    # 原稿已經夠好時直接採用：它的成本天生最低，不可能被贏過。門檻刻意
    # 訂得比「可見」還嚴，灰帶要進候選競賽讓成本函數權衡，別讓 4.0 那種
    # 平坦大色塊上看得見的小台階從這裡溜走。
    from app.quality import wrap_edge_color_break as _ecb_src

    src_ecb = float(_ecb_src(stamp_structure_view(arr)))
    if source_looks_seamless(
        src.rep,
        src.hotspot,
        orphan=src.orphan,
        fragment=src.fragment,
        wrap_cut=src.wrap_cut,
        gutter=src.gutter,
        period_rem=src.period_rem,
        wrap_density=src.wrap_density,
        edge_color_break=src_ecb,
    ):
        _lg("  → 原稿已無縫，直接採用")
        return (
            _unit_image(native, ctx),
            f"前景 {ratio:.0%}｜原稿已無縫（接縫超出 {src.rep.wrap_excess:.1f}，熱點 {src.hotspot:.1f}）",
        )

    # GUI 預設邊緣帶 0%（「不改圖」）時仍要能清邊補花：否則動物／雪花
    # 卡在畫框只會原圖直出。0 表示「用自動帶寬」，不是關掉補花。
    if margin_px <= 0:
        margin_px = max(8, int(round(min(h, w) * 0.03)))
        _lg(f"  → 邊緣帶 0，改用自動 {margin_px}px 做去邊／補花")

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

    # 2048 水彩碎花的全解析度相位搜尋可空轉 20+ 分鐘，最後仍常選清邊補花。
    # 點綴圖章更不該在週期裡耗死：錯週期會把刺蝟／動物剖開。
    # 1254 散點以前預算 180s，GUI 一批圖會感覺整台卡住。
    # 只缺單軸 paste 週期時仍可做 compact（大象 X=366、Y 無 MAE 仍能裁出 732×794）。
    crop = None
    mae_x = _mae_copy_period(arr, 1)
    mae_y = _mae_copy_period(arr, 0)
    if discrete and mae_x is None and mae_y is None:
        _lg("  → 點綴圖無 paste 週期，跳過週期裁切搜尋")
    else:
        # 週期搜尋吃滿預算會擠掉後面的補花／擇優；留給後面至少 ~25s。
        remain = max(8.0, deadline - time.perf_counter() - 25.0)
        crop_budget = min(
            12.0 if discrete else (22.0 if max(h, w) < 1800 else 30.0),
            remain,
        )
        crop = timed(
            "週期裁切搜尋",
            lambda: try_period_crop(arr, log=log, budget_s=crop_budget),
            log,
        )
    if crop is not None and crop[0] is not None:
        tile_c, detail_c = crop
        from app.quality import (
            seam_report as _sr_c,
            wrap_cut_ratio as _wcr_c,
            wrap_density_ratio as _wdr_c,
            wrap_hotspot as _wh_c,
            wrap_gutter_error as _wge_c,
            wrap_period_remainder as _wpr_c,
        )
        from app.select import (
            GUTTER_VISUAL_MAX,
            HOTSPOT_OK,
            PERIOD_REM_MAX,
            SEAM_OK,
            WRAP_CUT_MAX,
            WRAP_DENSITY_MAX,
            WRAP_DENSITY_MIN,
        )

        rep_c = _sr_c(tile_c)
        view_c = stamp_structure_view(tile_c)
        dens_c = _wdr_c(view_c)
        gut_c = _wge_c(view_c)
        prem_c = _wpr_c(tile_c)
        # 雲朵假週期 313×126：接縫色差有降，圖章仍剖在新邊上。
        # 幾何真週期 wrap/hot 都是 0，即使 wrap_cut 連通域誤報也要留。
        fake_stamp_crop = discrete and (
            rep_c.wrap_excess > SEAM_OK
            or _wcr_c(view_c) >= 0.85
            or (
                _wcr_c(view_c) > WRAP_CUT_MAX
                and _wh_c(view_c) > HOTSPOT_OK
            )
        )
        # 進候選用目視溝寬：GUTTER_ERR_MAX=0.35 太鬆，錯格週期會當選後再未達標。
        bad_struct = (
            dens_c < WRAP_DENSITY_MIN
            or dens_c > WRAP_DENSITY_MAX
            or gut_c > GUTTER_VISUAL_MAX
            or (
                prem_c > PERIOD_REM_MAX
                and not (
                    dens_c >= WRAP_DENSITY_MIN
                    and _wcr_c(view_c) <= WRAP_CUT_MAX
                    and gut_c <= GUTTER_VISUAL_MAX * 0.5
                    and prem_c <= 0.15
                )
            )
        )
        if fake_stamp_crop:
            _lg(f"  → 週期裁切仍切圖章，放棄（{detail_c}）")
        elif bad_struct:
            _lg(
                f"  → 週期裁切密度／錯格不合格，放棄"
                f"（dens={dens_c:.2f}）"
            )
        else:
            bases.append(_crop_base(tile_c, f"週期裁切（{detail_c}）"))

    if discrete and fine_src is None:
        from app.select import HOTSPOT_OK as _HOT_LAT

        # 十字已密但切圖／熱點差：晶格裁切通常幫不上，省時間直奔清邊補花。
        skip_lattice_cut = (
            float(src.wrap_density) >= 0.80
            and (
                float(src.wrap_cut) > 0.50
                or float(src.hotspot) > _HOT_LAT
            )
        )
        if skip_lattice_cut:
            _lg(
                f"  → 跳過點綴晶格（dens={src.wrap_density:.2f} "
                f"cut={src.wrap_cut:.0%}，改清邊補花）"
            )
        elif time.perf_counter() >= deadline - 28.0:
            _lg("  → 時間不夠，跳過點綴晶格")
        else:
            got = timed(
                "點綴晶格",
                lambda: try_make_discrete_seamless(arr, bg, threshold, log=log),
                log,
            )
            if got is not None and got[0] is not None:
                unit_d, detail_d = got
                same = unit_d.shape == arr.shape and np.array_equal(unit_d, arr)
                if not same and "FAIL" not in str(detail_d):
                    from app.quality import wrap_cut_ratio as _wcr_d, wrap_hotspot as _wh_d
                    from app.select import (
                        HOTSPOT_OK,
                        WRAP_CUT_MAX,
                        WRAP_DENSITY_MAX,
                        WRAP_DENSITY_MIN,
                    )

                    view_d = stamp_structure_view(unit_d)
                    from app.quality import wrap_density_ratio as _wdr_d

                    dens_d = _wdr_d(view_d)
                    if _wcr_d(view_d) > WRAP_CUT_MAX and _wh_d(view_d) > HOTSPOT_OK:
                        _lg(f"  → 點綴晶格仍切圖，放棄（{detail_d}）")
                    elif dens_d < WRAP_DENSITY_MIN or dens_d > WRAP_DENSITY_MAX:
                        _lg(
                            f"  → 點綴晶格密度不合格，放棄"
                            f"（dens={dens_d:.2f}）"
                        )
                    else:
                        bases.append(_crop_base(unit_d, f"點綴晶格（{detail_d}）"))

    # 「去邊」只裁掉外圈（像素仍是原稿），讓新邊緣較少殘肢，再交給最小誤差切。
    # 不限前景占比：滿版花布也只是少一圈邊，不是清掉花網。
    # 清邊補花會改色，同尺寸時色調閘門看得到；偏色超過門檻就不要進選單。
    # 細格紋去邊會切掉非整格，最小誤差切再把格子剪錯位。
    if margin_px > 0 and fine_src is None:
        insets = [(1, "去邊"), (2, "去寬邊")]
        if ratio < 0.20 or src.wrap_cut > 0.40:
            insets.append((3, "去更寬邊"))
        # 時間緊只留一層去邊，少幾個要展開刀法的基底。
        if time.perf_counter() >= deadline - 25.0:
            insets = insets[:1]
        for mul, tag in insets:
            m = margin_px * mul
            ih = h - 2 * m
            iw = w - 2 * m
            if min(ih, iw) < 64:
                continue
            inset = arr[m : m + ih, m : m + iw]
            bases.append(
                Base(inset, tag, True, [("inset", (m, m, ih, iw))])
            )
    # 雪人等散點前景可到 38%，0.34 會跳過清邊補花、只剩剖開圖章的最小誤差切。
    # 貓頭鷹這種分開的大圖章前景可到 50%，晶格分類會當成滿鋪而跳過補花。
    # 滿版連通底紋會在補花內因模板不足／毀圖直接放棄。
    # 原稿十字帶已經夠密且結構已乾淨：再清邊／整張重鋪只會把設計打散
    # （聖誕樹 derr 100+）。切圖／熱點／殘片仍高時，密十字多半是縫上擠花，
    # 不能因 dens≥0.80 就跳過——冬青剖開仍會未達標原圖。
    from app.quality import wrap_edge_color_break as _ecb_skip
    from app.select import (
        EDGE_COLOR_BREAK_MAX,
        HOTSPOT_OK,
        ORPHAN_RUN_MAX,
        SEAM_OK,
        SEAM_REFILL_MAX,
        WRAP_CUT_MAX as _WCM_SKIP,
        WRAP_DENSITY_VISUAL_MAX,
    )

    stamp_sheet = discrete or _has_separated_stamps(arr, bg, threshold)
    seam_struct_ok = (
        float(src.wrap_cut) <= _WCM_SKIP
        and float(src.hotspot) <= HOTSPOT_OK
        and int(src.orphan) <= ORPHAN_RUN_MAX
    )
    skip_ecb = float(_ecb_skip(stamp_structure_view(arr)))
    skip_wrap = float(src.rep.wrap_excess)
    skip_overdense = float(src.wrap_density) > WRAP_DENSITY_VISUAL_MAX
    skip_recolor = skip_ecb > EDGE_COLOR_BREAK_MAX
    # 原稿尚未補花對齊 SEAM_OK；SEAM_REFILL_MAX 是補花後上限。
    skip_excess = skip_wrap > SEAM_OK or skip_wrap > SEAM_REFILL_MAX
    skip_refill_dense = (
        stamp_sheet
        and src.wrap_density >= 0.80
        and seam_struct_ok
        and not skip_overdense
        and not skip_recolor
        and not skip_excess
    )
    if stamp_sheet and src.wrap_density >= 0.80 and not seam_struct_ok:
        _lg(
            f"  → 原稿接縫密度 {src.wrap_density:.2f} 但切圖/熱點仍差"
            f"（cut={src.wrap_cut:.0%} hot={src.hotspot:.0f}），仍清邊補花"
        )
    elif stamp_sheet and src.wrap_density >= 0.80 and (
        skip_overdense or skip_recolor or skip_excess
    ):
        reasons = []
        if skip_overdense:
            reasons.append(f"過密 dens={src.wrap_density:.2f}")
        if skip_recolor:
            reasons.append(f"錯色 ecb={skip_ecb:.1f}")
        if skip_excess:
            reasons.append(f"殘縫 wrap={skip_wrap:.1f}")
        _lg(
            f"  → 原稿接縫密度 {src.wrap_density:.2f} 但"
            f"{'／'.join(reasons)}，仍清邊補花"
        )
    elif skip_refill_dense:
        _lg(
            f"  → 原稿接縫密度 {src.wrap_density:.2f} 且結構已乾淨，"
            f"跳過清邊補花以免打散構圖"
        )
    if (
        margin_px > 0
        and fine_src is None
        and not skip_refill_dense
        and time.perf_counter() < deadline - 12.0
        and (
            ratio < 0.55
            or src.wrap_cut > 0.40
            or _overlay_stamp_mask(arr, bg, threshold) is not None
            or _has_separated_stamps(arr, bg, threshold)
        )
    ):
        # 清邊補花最多佔 ~28s，且須在父截止−4s 交卷，留給 choose。
        refill_deadline = min(float(deadline) - 4.0, time.perf_counter() + 28.0)
        filled = timed(
            "清邊補花",
            lambda: _clear_and_refill(
                arr, bg, threshold, margin_px, deadline=refill_deadline
            ),
            log,
        )
        if filled is not None:
            from app.quality import tone_shift as _ts_f, wrap_cut_ratio as _wcr_f, wrap_density_ratio as _wdr_f
            from app.select import (
                DESIGN_MAX,
                TONE_REFILL_MAX,
                WRAP_CUT_MAX,
                WRAP_CUT_REFILL_MAX,
                WRAP_DENSITY_MAX,
                WRAP_DENSITY_MIN,
            )

            view_f = stamp_structure_view(filled)
            cut_f = _wcr_f(view_f)
            dens_f = _wdr_f(view_f)
            tone_f = _ts_f(arr, filled, edge_frac=0.20)
            from app.quality import design_error as _derr_f

            derr_f = float(_derr_f(arr, filled))
            stampy = discrete or _has_separated_stamps(arr, bg, threshold)
            wipeish = "整張重排" in _LAST_REFILL_TAG
            fixes_cut = (
                cut_f <= WRAP_CUT_REFILL_MAX
                and WRAP_DENSITY_MIN <= dens_f <= WRAP_DENSITY_MAX
                and src.wrap_cut > WRAP_CUT_MAX
            )
            # 非 wipe 且高 derr：擋打散構圖；但雪人環帶吸附修掉原切圖要放行。
            if stampy and derr_f > 45.0 and not wipeish and not fixes_cut:
                _lg(
                    f"  → 清邊補花毀圖過大，放棄"
                    f"（derr={derr_f:.1f} cut={cut_f:.0%} dens={dens_f:.2f}）"
                )
            elif (
                tone_f > TONE_REFILL_MAX
                or cut_f > WRAP_CUT_REFILL_MAX
                or dens_f < WRAP_DENSITY_MIN
                or dens_f > WRAP_DENSITY_MAX
            ):
                if stampy:
                    # 切圖完全沒修好（仍爆表／幾乎沒降）：別進候選搶贏原圖／去邊。
                    cut_unfixed = (
                        cut_f > WRAP_CUT_REFILL_MAX
                        and float(src.wrap_cut) > WRAP_CUT_MAX
                        and cut_f >= min(8.0, float(src.wrap_cut) * 0.85)
                    )
                    if derr_f > DESIGN_MAX:
                        _lg(
                            f"  → 清邊補花毀圖過大，放棄"
                            f"（derr={derr_f:.1f} cut={cut_f:.0%} dens={dens_f:.2f}）"
                        )
                    elif cut_unfixed and dens_f < 0.85:
                        _lg(
                            f"  → 清邊補花切圖未修且偏疏，放棄"
                            f"（cut={cut_f:.0%} dens={dens_f:.2f}）"
                        )
                    elif cut_unfixed and cut_f >= 4.0:
                        _lg(
                            f"  → 清邊補花切圖仍爆表未降，放棄"
                            f"（cut={cut_f:.0%}←{src.wrap_cut:.0%} dens={dens_f:.2f}）"
                        )
                    else:
                        _lg(
                            f"  → 清邊補花未過硬門，仍進候選"
                            f"（cut={cut_f:.0%} tone={tone_f:.1f} dens={dens_f:.2f}）"
                        )
                        bases.append(Base(filled, _LAST_REFILL_TAG, False, None))
                else:
                    _lg(
                        f"  → 清邊補花切圖／色調／密度仍差，放棄"
                        f"（cut={cut_f:.0%} tone={tone_f:.1f} dens={dens_f:.2f}）"
                    )
            else:
                bases.append(Base(filled, _LAST_REFILL_TAG, False, None))

    best = choose(src, bases, log=log, deadline=deadline)
    if best.recipe is not None:
        unit = _unit_image(apply_recipe(native, best.recipe), ctx)
    else:
        unit = restore(Image.fromarray(best.arr, mode="RGB"), ctx)
    elapsed = time.perf_counter() - t_gen
    if elapsed > gen_budget * 0.95:
        _lg(f"  → 生成耗時 {elapsed:.1f}s（已觸及預算 {gen_budget:.0f}s）")
    return unit, f"前景 {ratio:.0%}｜{best.label}"
