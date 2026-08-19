"""跑無縫之前的分流：成品外框與產品封面圖直接跳過。

判斷一律看畫面內容，不看檔名。四方連續花布是空間平穩的——任一象限
的顏色分佈都該差不多；產品封面圖（尺寸表、行銷字、半張空白）不是。
非平穩還不夠：水彩暈染、拼布格、大色階漸層同樣非平穩，但那就是設計。
所以要再加一道證據（半張空白、或高對比行銷文字排版）才判定跳過。
寧可放過不可錯殺。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from app.color_utils import analysis_rgb

VERDICT_TILEABLE = "tileable"
VERDICT_FRAMED = "framed"
VERDICT_NOT_PATTERN = "not_pattern"

_ANALYZE_SIDE = 256

# 非平穩：象限直方圖卡方 + 區塊平均色離散。門檻取自 30 張封面 / 180 張
# 花布的分離曲線（封面 p05 quad=0.099，花布 p95=0.056）。
_QUAD_MIN = 0.08
_HET_MIN = 20.0

# 連邊地色：半張空白用。花布地色也會連邊，不能單憑面積當封面。
_BG_COLOR_DIST = 22.0

# 半張空白：一邊幾乎是地色、對邊有內容。稀疏點綴不會這麼極端。
_LOPSIDED_HIGH = 0.80
_LOPSIDED_LOW = 0.35

# 行銷字：字母狀小連通域排成水平列。密花也會觸發，故不能單獨當跳過條件。
_TEXT_ROW_MIN = 6
# 數位拼貼封面（多塊花布 + 橫幅字）：非平穩且文字列很多。
# 硬花布的 text_rows 雖高，quad 通常 < 0.12；門檻取自校準集空隙。
_TEXT_LAYOUT_ROWS = 16
_TEXT_LAYOUT_QUAD = 0.32

# 單色外框：四邊同色近定值帶，內部與框色明顯不同。
_FRAME_MIN_PX = 4
_FRAME_MAX_FRAC = 0.18
_FRAME_LINE_STD = 10.0
_FRAME_COLOR_DIST = 16.0
_FRAME_WIDTH_TOL = 0.40
_FRAME_WIDTH_ABS = 6
_FRAME_INTERIOR_DIST = 28.0
_FRAME_INTERIOR_MATCH_MAX = 0.38


@dataclass(frozen=True)
class Triage:
    """分流結果。"""

    verdict: str
    reasons: list[str]
    signals: dict[str, float]

    def describe(self) -> str:
        if self.verdict == VERDICT_FRAMED:
            detail = "／".join(self.reasons) or "四邊有框"
            return f"跳過：成品外框（{detail}）"
        if self.verdict == VERDICT_NOT_PATTERN:
            detail = "／".join(self.reasons) or "產品封面圖"
            return f"跳過：產品封面圖（{detail}）"
        return "可四方連續"


def _luminance(rgb: np.ndarray) -> np.ndarray:
    a = rgb.astype(np.float32)
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def _resize_rgb(image: Image.Image, side: int) -> np.ndarray:
    rgb = analysis_rgb(image)
    small = rgb.resize((side, side), Image.Resampling.BILINEAR)
    return np.asarray(small, dtype=np.uint8)


def _chi_square(a: np.ndarray, b: np.ndarray) -> float:
    return float(0.5 * np.sum((a - b) ** 2 / (a + b + 1e-9)))


def _color_hist(arr: np.ndarray) -> np.ndarray:
    hist = cv2.calcHist(
        [arr], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]
    ).ravel()
    s = float(hist.sum())
    return hist / (s + 1e-9)


def _stationarity(arr: np.ndarray) -> dict[str, float]:
    """象限／半邊顏色分佈差，以及 8×8 區塊平均色的空間離散。"""
    h, w = arr.shape[:2]
    hy, hx = h // 2, w // 2
    quads = (
        arr[:hy, :hx],
        arr[:hy, hx:],
        arr[hy:, :hx],
        arr[hy:, hx:],
    )
    hs = [_color_hist(q) for q in quads]
    quad = max(
        _chi_square(hs[i], hs[j]) for i in range(4) for j in range(i + 1, 4)
    )
    halves = max(
        _chi_square(_color_hist(arr[:hy]), _color_hist(arr[hy:])),
        _chi_square(_color_hist(arr[:, :hx]), _color_hist(arr[:, hx:])),
    )
    grid = 8
    bh, bw = h // grid, w // grid
    means = np.empty((grid, grid, 3), dtype=np.float64)
    for i in range(grid):
        for j in range(grid):
            block = arr[i * bh : (i + 1) * bh, j * bw : (j + 1) * bw]
            means[i, j] = block.reshape(-1, 3).mean(axis=0)
    het = float(np.linalg.norm(means.reshape(-1, 3).std(axis=0)))
    lum = _luminance(arr)
    small = cv2.resize(lum, (16, 16), interpolation=cv2.INTER_AREA)
    lows = float(small.std()) / (float(lum.std()) + 1e-6)
    return {
        "quad": float(quad),
        "halves": float(halves),
        "het": het,
        "lows": lows,
    }


def _edge_connected_bg(arr: np.ndarray) -> np.ndarray:
    """與邊界相連、顏色接近邊緣中位色的區域。"""
    h, w = arr.shape[:2]
    edge = np.concatenate(
        [arr[0], arr[-1], arr[:, 0], arr[:, -1]], axis=0
    ).astype(np.float32)
    med = np.median(edge, axis=0)
    dist = np.sqrt(((arr.astype(np.float32) - med) ** 2).sum(axis=2))
    mask = (dist < _BG_COLOR_DIST).astype(np.uint8)
    _n, labels = cv2.connectedComponents(mask, connectivity=4)
    border_ids = np.unique(
        np.concatenate(
            [labels[0], labels[-1], labels[:, 0], labels[:, w - 1]]
        )
    )
    keep = set(int(i) for i in border_ids if i != 0)
    if not keep:
        return np.zeros((h, w), dtype=bool)
    lut = np.zeros(int(labels.max()) + 1, dtype=bool)
    for i in keep:
        lut[i] = True
    return lut[labels]


def _lopsided_blank(bg: np.ndarray) -> dict[str, float]:
    """半張是地色、對邊有內容：封面裁切殘片或沒鋪滿的產品圖。"""
    h, w = bg.shape
    hy, hx = h // 2, w // 2
    parts = (
        float(bg[:hy].mean()),
        float(bg[hy:].mean()),
        float(bg[:, :hx].mean()),
        float(bg[:, hx:].mean()),
    )
    hi = max(parts)
    lo = min(parts)
    return {
        "lopsided_high": hi,
        "lopsided_low": lo,
        "lopsided": 1.0 if hi >= _LOPSIDED_HIGH and lo <= _LOPSIDED_LOW else 0.0,
    }


def _text_density(arr: np.ndarray) -> dict[str, float]:
    """字母狀小連通域排成水平列的數量。"""
    g = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    g = cv2.resize(g, (512, 512), interpolation=cv2.INTER_AREA)
    bw = cv2.adaptiveThreshold(
        g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 12
    )
    n, _labels, stats, centroids = cv2.connectedComponentsWithStats(
        bw, connectivity=8
    )
    letters: list[tuple[float, float]] = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        bw_i = int(stats[i, cv2.CC_STAT_WIDTH])
        bh_i = int(stats[i, cv2.CC_STAT_HEIGHT])
        if not (8 <= area <= 180 and 4 <= bh_i <= 26 and 2 <= bw_i <= 28):
            continue
        aspect = bw_i / max(bh_i, 1)
        if not (0.15 <= aspect <= 3.2):
            continue
        letters.append((float(centroids[i][1]), float(centroids[i][0])))
    if len(letters) < _TEXT_ROW_MIN:
        return {"letters": float(len(letters)), "text_rows": 0.0}
    letters.sort()
    rows = 0
    used = [False] * len(letters)
    for i, (cy, _cx) in enumerate(letters):
        if used[i]:
            continue
        members = 1
        used[i] = True
        for j in range(i + 1, len(letters)):
            if used[j]:
                continue
            if abs(letters[j][0] - cy) <= 7.0:
                used[j] = True
                members += 1
        if members >= _TEXT_ROW_MIN:
            rows += 1
    return {"letters": float(len(letters)), "text_rows": float(rows)}


def _side_uniform_run(
    arr: np.ndarray,
    axis: str,
    reverse: bool,
    max_scan: int,
) -> tuple[int, np.ndarray]:
    """從一側量近定值、近同色的連續帶寬，回傳 (寬度, 帶的中位色)。"""
    h, w = arr.shape[:2]
    if axis == "row":
        order = range(h - 1, h - 1 - max_scan, -1) if reverse else range(max_scan)
        ref = arr[-1] if reverse else arr[0]
    else:
        order = range(w - 1, w - 1 - max_scan, -1) if reverse else range(max_scan)
        ref = arr[:, -1] if reverse else arr[:, 0]
    ref_c = np.median(ref.reshape(-1, 3).astype(np.float32), axis=0)
    width = 0
    for idx in order:
        line = arr[idx] if axis == "row" else arr[:, idx]
        pix = line.reshape(-1, 3).astype(np.float32)
        if float(pix.std(axis=0).max()) > _FRAME_LINE_STD:
            break
        if float(np.sqrt(((pix.mean(axis=0) - ref_c) ** 2).sum())) > _FRAME_COLOR_DIST:
            break
        width += 1
    return width, ref_c


def uniform_frame(image: Image.Image) -> dict[str, float]:
    """
    四邊同色近定值外框。黑／白／灰／品牌色都認；滿版素色不算框。
    """
    rgb = analysis_rgb(image)
    arr = np.asarray(rgb, dtype=np.uint8)
    h, w = arr.shape[:2]
    if min(h, w) < 64:
        return {"frame": 0.0, "frame_w": 0.0}
    max_scan = max(
        _FRAME_MIN_PX + 1,
        min(int(min(h, w) * _FRAME_MAX_FRAC), h // 5, w // 5, 160),
    )
    specs = (
        ("row", False),
        ("row", True),
        ("col", False),
        ("col", True),
    )
    widths: list[int] = []
    colors: list[np.ndarray] = []
    for axis, rev in specs:
        bw, col = _side_uniform_run(arr, axis, rev, max_scan)
        widths.append(bw)
        colors.append(col)
    if min(widths) < _FRAME_MIN_PX:
        return {"frame": 0.0, "frame_w": float(min(widths))}
    ref_w = float(np.median(widths))
    for bw in widths:
        if abs(bw - ref_w) > max(_FRAME_WIDTH_ABS, ref_w * _FRAME_WIDTH_TOL):
            return {"frame": 0.0, "frame_w": float(min(widths))}
    cols = np.stack(colors)
    if float(cols.std(axis=0).max()) > 18.0:
        return {"frame": 0.0, "frame_w": ref_w}
    t, b, l, rgt = widths
    interior = arr[t : h - b, l : w - rgt]
    if interior.size < 100 or min(interior.shape[:2]) < 16:
        return {"frame": 0.0, "frame_w": ref_w}
    frame_c = cols.mean(axis=0)
    interior_c = interior.reshape(-1, 3).astype(np.float32).mean(axis=0)
    dist = float(np.sqrt(((interior_c - frame_c) ** 2).sum()))
    match = float(
        np.mean(
            np.sqrt(
                ((interior.astype(np.float32) - frame_c) ** 2).sum(axis=2)
            )
            < _FRAME_COLOR_DIST + 8.0
        )
    )
    # 花布白底常在四邊留下一圈地色；內部仍大量同色，不是框。
    if dist < _FRAME_INTERIOR_DIST or match > _FRAME_INTERIOR_MATCH_MAX:
        return {"frame": 0.0, "frame_w": ref_w, "frame_dist": dist, "frame_match": match}
    return {
        "frame": 1.0,
        "frame_w": ref_w,
        "frame_dist": dist,
        "frame_match": match,
    }


def kuotu_black_white_frame(image: Image.Image) -> bool:
    """kuotu 成品雙框：外黑內白、四邊同寬。"""
    from app.color_utils import _legacy_kuotu_frame

    return _legacy_kuotu_frame(image)


def is_framed(image: Image.Image) -> bool:
    """成品外框：kuotu 黑白雙框，或任一單色均勻外框。"""
    if kuotu_black_white_frame(image):
        return True
    return uniform_frame(image).get("frame", 0.0) >= 1.0


def triage(image: Image.Image) -> Triage:
    """把圖分成可拼接／有框／封面圖。"""
    reasons: list[str] = []
    signals: dict[str, float] = {}

    kuotu = kuotu_black_white_frame(image)
    frame = uniform_frame(image)
    signals.update(frame)
    signals["kuotu_frame"] = 1.0 if kuotu else 0.0
    if kuotu or frame.get("frame", 0.0) >= 1.0:
        if kuotu:
            reasons.append("kuotu黑白雙框")
        if frame.get("frame", 0.0) >= 1.0:
            reasons.append(f"單色外框{frame.get('frame_w', 0):.0f}px")
        return Triage(VERDICT_FRAMED, reasons, signals)

    arr = _resize_rgb(image, _ANALYZE_SIDE)
    stat = _stationarity(arr)
    bg = _edge_connected_bg(arr)
    lop = _lopsided_blank(bg)
    text = _text_density(arr)
    signals.update(stat)
    signals.update(lop)
    signals.update(text)

    nonstat = stat["quad"] > _QUAD_MIN and stat["het"] > _HET_MIN
    cover = False
    if lop["lopsided"] >= 1.0:
        cover = True
        reasons.append("半張空白")
    if (
        nonstat
        and text["text_rows"] >= _TEXT_LAYOUT_ROWS
        and stat["quad"] >= _TEXT_LAYOUT_QUAD
    ):
        cover = True
        reasons.append(f"行銷文字排版{text['text_rows']:.0f}列")

    if nonstat and cover:
        return Triage(VERDICT_NOT_PATTERN, reasons, signals)
    return Triage(VERDICT_TILEABLE, [], signals)


def skip_message(result: Triage) -> str:
    return result.describe()
