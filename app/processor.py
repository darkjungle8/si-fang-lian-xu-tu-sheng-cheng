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
    dist = np.abs(areas - float(target_area))
    weights = (1.0 / (1.0 + dist / max(float(target_area), 1.0))) * (fill + 0.05)
    if target_rgb is not None and len(pool) > 1:
        tgt = np.asarray(target_rgb, dtype=np.float64)
        cdist = np.array(
            [float(np.linalg.norm(_stamp_mean_rgb(m) - tgt)) for m in pool],
            dtype=np.float64,
        )
        weights = weights * (1.0 / (1.0 + cdist / 18.0))
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
    pending_edge = _space_wrap_jobs(
        pending_edge, h, w, _interior_nn_separation(occupied)
    )
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
    max_p = w - max(32, min(w // 6, 80))
    if max_p <= min_p:
        return None
    row_step = max(1, h // 40)
    sl = a[::row_step]
    probe = sl[:, 0].astype(np.int16)
    for p in range(min_p, max_p + 1):
        if float(np.mean(np.abs(sl[:, p].astype(np.int16) - probe))) > tol:
            continue
        mae = float(np.mean(np.abs(sl[:, :-p].astype(np.int16) - sl[:, p:])))
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
    from app.select import GUTTER_ERR_MAX, PERIOD_REM_MAX, WRAP_CUT_MAX, WRAP_DENSITY_MAX, compact_min_edge

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
        # 格紋 7 格單元 wrap/hot 也可為 0，溝寬仍差 46%——crop_clean 不能免錯格。
        crop_clean = rep.wrap_excess <= 2.0 and hot <= 14.0
        gut = wrap_gutter_error(view)
        if gut > GUTTER_ERR_MAX:
            continue
        if max(_len_period_rem(cw, px), _len_period_rem(ch, py)) > PERIOD_REM_MAX:
            continue
        if not crop_clean:
            if rep.internal_excess > allow:
                continue
            if wrap_cut_ratio(view) > WRAP_CUT_MAX:
                continue
            if wrap_density_ratio(view) > WRAP_DENSITY_MAX:
                continue
            if wrap_period_remainder(tile) > PERIOD_REM_MAX:
                continue
        key = (
            rep.wrap_excess,
            0 if crop_clean else 1,
            round(gut, 3),
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
    """環面相位：未裁的軸要滾開原 wrap，否則那一軸的舊縫會留在新單元邊上。"""
    from app.seamless_core import torus_crop

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
    best, bxy = 1e9, (xs[0], ys[0])
    for ox in xs:
        for oy in ys:
            tile = torus_crop(arr, oy, ox, ch, cw)
            sc = sum(_tile_seam_scores(tile))
            if sc < best:
                best, bxy = sc, (ox, oy)
    ox, oy = bxy
    rx = max(2, stepx // 2)
    ry = max(2, stepy // 2)
    for dx in range(-rx, rx + 1, 2):
        for dy in range(-ry, ry + 1, 2):
            x = (ox + dx) % w
            y = (oy + dy) % h
            sc = sum(_tile_seam_scores(torus_crop(arr, y, x, ch, cw)))
            if sc < best:
                best, bxy = sc, (x, y)
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
    n, _, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    areas = [
        int(stats[i, cv2.CC_STAT_AREA])
        for i in range(1, n)
        if int(stats[i, cv2.CC_STAT_AREA]) >= 200
    ]
    if len(areas) < 6:
        return False
    total = int(sum(areas)) or 1
    if max(areas) > total * 0.45:
        return False
    return True


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
        wrap_cut_ratio,
        wrap_density_ratio,
        wrap_hotspot,
        wrap_orphan_run,
    )
    from app.select import HOTSPOT_REFILL_OK, ORPHAN_RUN_MAX, WRAP_CUT_MAX, WRAP_CUT_REFILL_MAX, WRAP_DENSITY_MAX

    view = _stamp_flat_view(arr, bg, threshold)
    cut = wrap_cut_ratio(view)
    hot_ok = wrap_hotspot(view) <= HOTSPOT_REFILL_OK or cut <= WRAP_CUT_REFILL_MAX
    return (
        hot_ok
        and cut <= WRAP_CUT_REFILL_MAX
        and wrap_orphan_run(view) <= ORPHAN_RUN_MAX
        and wrap_density_ratio(view) <= WRAP_DENSITY_MAX
    )


def _seal_wrap_background(
    arr: np.ndarray | None,
    bg: Sequence[int],
    threshold: float,
    plate: np.ndarray | None = None,
) -> np.ndarray | None:
    """把接縫上非圖章、又接近地色的像素封回背景版。

    淡底雪花／抗鋸齒殘邊進不了 stamp mask，卻能讓 wrap_excess 卡在 5～10。
    圖章本體（含膨脹一圈）不動。
    """
    from app.color_utils import color_distance

    if arr is None:
        return None
    fg = _stamp_foreground(arr, bg, threshold).astype(np.uint8)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    prot = cv2.dilate(fg, ker).astype(bool)
    dist = color_distance(arr, bg)
    near = dist < float(threshold)
    h, w = arr.shape[:2]
    band = 2
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


def _repair_wrap_cuts(
    arr: np.ndarray,
    bg: Sequence[int],
    threshold: float,
    templates: list,
    min_area: int,
) -> np.ndarray:
    """清掉仍停在畫框上的切圖，把完整圖章吸到 wrap 線上重貼。"""
    out = arr
    for _ in range(5):
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
    from app.quality import seam_report, wrap_cut_repair
    from app.select import SEAM_OK

    if not _refill_wrap_is_ok(arr, bg, threshold):
        from app.quality import wrap_cut_ratio as _wcr_need
        from app.select import WRAP_CUT_REFILL_MAX as _CUT_NEED

        view = _stamp_flat_view(arr, bg, threshold)
        if _wcr_need(view) > _CUT_NEED:
            return True
        if seam_report(arr).wrap_excess > SEAM_OK:
            return bool(wrap_cut_repair(view).kill_ids)
        return False
    if seam_report(arr).wrap_excess <= SEAM_OK:
        return False
    view = _stamp_flat_view(arr, bg, threshold)
    return bool(wrap_cut_repair(view).kill_ids)


def _maybe_repair_wrap(
    src: np.ndarray,
    filled: np.ndarray | None,
    bg: Sequence[int],
    threshold: float,
    templates: list,
    min_area: int,
) -> tuple[np.ndarray | None, bool]:
    if filled is None:
        return None, False
    if not _refill_needs_repair(filled, bg, threshold):
        return filled, _refill_wrap_is_ok(filled, bg, threshold)
    repaired = _repair_wrap_cuts(filled, bg, threshold, templates, min_area)
    if not _refill_needs_repair(repaired, bg, threshold):
        return repaired, _refill_wrap_is_ok(repaired, bg, threshold)
    if _clear_edge_ruins_motifs(src, repaired, bg, threshold):
        return filled, False
    return repaired, _refill_wrap_is_ok(repaired, bg, threshold)


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
        wrap_hotspot,
    )
    from app.select import SEAM_OK, TONE_REFILL_MAX, WRAP_CUT_REFILL_MAX

    view = _stamp_flat_view(cand, bg, threshold)
    dens = wrap_density_ratio(view)
    wrap_ex = seam_report(cand).wrap_excess
    cut = wrap_cut_ratio(view)
    errs = 0 if _refill_wrap_is_ok(cand, bg, threshold) else 3
    if wrap_ex > SEAM_OK:
        errs += 2
    if dens > 1.15:
        errs += 2
    if src is not None and tone_shift(src, cand, edge_frac=0.20) > TONE_REFILL_MAX:
        errs += 4
    return (
        errs,
        wrap_ex,
        0.0 if cut <= WRAP_CUT_REFILL_MAX else cut,
        dens,
        wrap_hotspot(view),
        edge_void_ratio(cand),
    )


def _clear_and_refill(
    arr: np.ndarray,
    bg: Sequence[int],
    threshold: float,
    margin_px: int,
) -> np.ndarray | None:
    """
    清掉碰到畫框的圖章群組，再用完整圖案在環帶上藍噪聲貼回。

    優先只動邊緣一圈；填不均勻則擴大環帶，再不行整張重排。
    圖章不鏡射、不旋轉。格紋點綴先用週期拷貝填回底紋。
    """
    from app.quality import edge_void_ratio

    global _LAST_REFILL_TAG
    _LAST_REFILL_TAG = "清邊補花"
    del margin_px

    overlay = _overlay_stamp_mask(arr, bg, threshold)
    fg = _stamp_foreground(arr, bg, threshold)
    touch = _touch_margin_px(fg)
    # 散點星：15px 帶會把整圈星都當碰框，補回去縫上過密。
    if overlay is not None and _luma_square_grid_pitch(arr) is None:
        touch = 2
    motifs = extract_interior_motifs(arr, bg, threshold, touch)
    if len(motifs) < 2:
        return None
    templates = _complete_templates(motifs)
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
    attach_group_colors(arr, labels, stats, groups)
    layout = layout_stats_from_groups(groups, h, w, interior_only=True)
    ring_place = max(float(touch), 1.2 * layout.span)
    cap = 0.22 * min(h, w)
    ring_place = float(min(ring_place, cap))
    ring_clear = ring_place
    if layout.d_nn >= 16.0:
        # 第一圈若離 wrap 近到無法再貼跨縫件，會留下「貼在縫上卻不跨」的圖章。
        ring_clear = max(ring_place, min(0.62 * layout.d_nn, cap))
        ring_place = max(ring_place, ring_clear)
    # 大圖章：0.62×d_nn 會清掉 20%+ 畫面，補回去就在縫上疊成一排。
    if layout.span >= 0.16 * min(h, w):
        ring_clear = min(
            ring_clear, max(float(touch) + 4.0, 0.08 * min(h, w))
        )
        ring_place = max(ring_place, ring_clear)
    plate = None if overlay is not None else background_plate(arr, fg.astype(bool))

    def _pick_layout(cleaned: np.ndarray, occ: np.ndarray, full: bool, ring_w: float):
        best = None
        best_key = None
        seeds = (42, 43, 44, 45, 46, 47) if full else (42, 43)
        for seed in seeds:
            cand = relayout_ring(
                cleaned,
                templates,
                layout,
                ring_w,
                seed,
                occupied=occ.copy(),
                full=full,
            )
            key = _score_refill(cand, bg, threshold, src=arr)
            if best is None or key < best_key:
                best, best_key = cand, key
            if key[0] == 0:
                return cand, True
        return best, bool(best_key is not None and best_key[0] == 0)

    def _finish(cand: np.ndarray | None) -> tuple[np.ndarray | None, bool]:
        cand, _ok = _maybe_repair_wrap(
            arr, cand, bg, threshold, motifs, min_area
        )
        cand = _seal_wrap_background(cand, bg, threshold, plate)
        if cand is None:
            return None, False
        return cand, _refill_wrap_is_ok(cand, bg, threshold)

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
        cleaned, removed = remove_edge_touching_components(
            arr,
            bg,
            threshold,
            touch,
            min_area=min_area,
            ring_px=int(round(ring_clear)),
            plate=plate,
        )
        cleaned, extra = _erase_interior_fragments(
            cleaned, bg, threshold, touch, templates, min_area=min_area
        )
        inner = [
            j
            for j in (removed + extra)
            if len(j) > 3 and not bool(j[3])
        ]
        if inner:
            cleaned = refill_with_wrapped_motifs(
                cleaned, bg, threshold, templates, inner, seed=42
            )
        occ = _stamp_foreground(cleaned, bg, threshold).astype(bool)
        filled = cleaned
        if removed:
            filled = refill_with_wrapped_motifs(
                cleaned, bg, threshold, templates, removed, seed=42, occupied=occ.copy()
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
        if not ok and layout.d_nn >= 90.0:
            occ_f = _stamp_foreground(cleaned, bg, threshold).astype(bool)
            full_cand, _ = _pick_layout(cleaned, occ_f, True, ring_place)
            full_cand, ok_f = _finish(full_cand)
            if full_cand is not None and (
                filled is None
                or _score_refill(full_cand, bg, threshold, src=arr)
                < _score_refill(filled, bg, threshold, src=arr)
            ):
                filled, ok = full_cand, ok_f
                if ok:
                    _LAST_REFILL_TAG = "清邊補花（整張重排）"
        # 環帶補花失敗仍把結果交給閘門，不要在這裡直接丟掉。

    if (
        filled is not None
        and templates
        and not _refill_wrap_is_ok(filled, bg, threshold)
    ):
        from app.quality import wrap_cut_ratio as _wcr_end
        from app.select import WRAP_CUT_REFILL_MAX as _CUT_END

        still_cut = (
            _wcr_end(_stamp_flat_view(filled, bg, threshold)) > _CUT_END
        )
        if still_cut:
            occ = _stamp_foreground(filled, bg, threshold).astype(bool)
            tight = replace(
                layout,
                d_nn=max(8.0, float(layout.d_nn) * 0.55),
                d_min=max(6.0, float(layout.d_min) * 0.65),
            )
            _place_along_wrap(
                filled, occ, templates, tight, np.random.default_rng(99)
            )

    filled = _seal_wrap_background(filled, bg, threshold, plate)
    if filled is None:
        return None
    if overlay is None:
        before = foreground_ratio(arr, bg, threshold)
        after = foreground_ratio(filled, bg, threshold)
        if after < max(0.03, before * 0.42):
            return None
    src_void = edge_void_ratio(arr)
    out_void = edge_void_ratio(filled)
    if (
        not _refill_wrap_is_ok(filled, bg, threshold)
        and out_void > 0.26
        and out_void > src_void + 0.12
    ):
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
        Base,
        apply_recipe,
        choose,
        source_facts,
        source_looks_seamless,
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
    fine_src = _luma_square_grid_pitch(arr)
    src = source_facts(arr, needs_native=ctx.mode == "CMYK")
    _lg(
        f"  → 分類：{'點綴/晶格' if discrete else '滿鋪或稀疏'}，"
        f"前景 {ratio:.0%}，尺寸 {w}×{h}，{ctx.describe()}，{src.rep.describe()}"
    )

    # 原稿已經夠好時直接採用：它的成本天生最低，不可能被贏過。門檻刻意
    # 訂得比「可見」還嚴，灰帶要進候選競賽讓成本函數權衡，別讓 4.0 那種
    # 平坦大色塊上看得見的小台階從這裡溜走。
    if source_looks_seamless(
        src.rep,
        src.hotspot,
        orphan=src.orphan,
        fragment=src.fragment,
        wrap_cut=src.wrap_cut,
        gutter=src.gutter,
        period_rem=src.period_rem,
        wrap_density=src.wrap_density,
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
    crop_budget = 90.0 if max(h, w) >= 1800 else 180.0
    crop = timed(
        "週期裁切搜尋",
        lambda: try_period_crop(arr, log=log, budget_s=crop_budget),
        log,
    )
    if crop is not None and crop[0] is not None:
        tile_c, detail_c = crop
        from app.quality import seam_report as _sr_c, wrap_cut_ratio as _wcr_c, wrap_hotspot as _wh_c
        from app.select import SEAM_OK, WRAP_CUT_MAX, HOTSPOT_OK

        rep_c = _sr_c(tile_c)
        view_c = stamp_structure_view(tile_c)
        # 雲朵假週期 313×126：接縫色差有降，圖章仍剖在新邊上。
        # 幾何真週期 wrap/hot 都是 0，即使 wrap_cut 連通域誤報也要留。
        fake_stamp_crop = discrete and (
            rep_c.wrap_excess > SEAM_OK
            or (
                _wcr_c(view_c) > WRAP_CUT_MAX
                and _wh_c(view_c) > HOTSPOT_OK
            )
        )
        if fake_stamp_crop:
            _lg(f"  → 週期裁切仍切圖章，放棄（{detail_c}）")
        else:
            bases.append(_crop_base(tile_c, f"週期裁切（{detail_c}）"))

    if discrete and fine_src is None:
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
                from app.select import WRAP_CUT_MAX, HOTSPOT_OK

                view_d = stamp_structure_view(unit_d)
                if _wcr_d(view_d) > WRAP_CUT_MAX and _wh_d(view_d) > HOTSPOT_OK:
                    _lg(f"  → 點綴晶格仍切圖，放棄（{detail_d}）")
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
    if (
        margin_px > 0
        and fine_src is None
        and (
            ratio < 0.55
            or src.wrap_cut > 0.40
            or _overlay_stamp_mask(arr, bg, threshold) is not None
            or _has_separated_stamps(arr, bg, threshold)
        )
    ):
        filled = timed(
            "清邊補花",
            lambda: _clear_and_refill(arr, bg, threshold, margin_px),
            log,
        )
        if filled is not None:
            from app.quality import tone_shift as _ts_f, wrap_cut_ratio as _wcr_f
            from app.select import TONE_REFILL_MAX, WRAP_CUT_REFILL_MAX

            view_f = stamp_structure_view(filled)
            cut_f = _wcr_f(view_f)
            tone_f = _ts_f(arr, filled, edge_frac=0.20)
            if tone_f > TONE_REFILL_MAX or cut_f > WRAP_CUT_REFILL_MAX:
                _lg(
                    f"  → 清邊補花切圖／色調仍差，放棄"
                    f"（cut={cut_f:.0%} tone={tone_f:.1f}）"
                )
            else:
                bases.append(Base(filled, _LAST_REFILL_TAG, False, None))

    best = choose(src, bases, log=log)
    if best.recipe is not None:
        unit = _unit_image(apply_recipe(native, best.recipe), ctx)
    else:
        unit = restore(Image.fromarray(best.arr, mode="RGB"), ctx)
    return unit, f"前景 {ratio:.0%}｜{best.label}"
