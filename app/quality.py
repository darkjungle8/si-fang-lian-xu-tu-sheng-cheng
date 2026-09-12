"""客觀品質指標：量測目視會看到的破壞。

## 為什麼要用「超出量」而不是絕對值

`_tile_seam_scores` 只量最外圈對邊色差（wrap 線）。半幅滾動是無損的
`np.roll`，它把原稿的外緣接縫搬到單元中央，wrap 線因此變得完美，使用者
仍會在畫面正中央看到同一條縫。所以必須把單元當成環形，掃描**每一條**線。

但線差的絕對值由圖案內容主導：條紋壁紙相鄰兩列本來就差 112，滿版素色
只差 2。同一個門檻不可能同時適用。因此所有接縫指標都改成
**超出量 = 該線差 − 該圖自身的典型線差（中位數）**，這是無量綱的、可以
跨圖比較的量。

## 兩道閘門

1. `wrap_excess`：單元四邊自己接自己的超出量。這是使用者第一眼會看到
   的那條縫，必須壓到接近 0。
2. `internal_excess`：環形單元內部最差線的超出量。半幅搬家、最小誤差切
   斷裂都會在這裡現形。判定時與原稿比較，因為條紋壁紙的硬邊在原稿裡
   本來就存在，不該算到我們頭上。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import median_filter

_AXIS_ANALYSIS_SIDE = 900
# 軸向結構低於此比例時，幾何保真無從比較（有機圖案沒有橫豎線可扳斜）。
# 實測：拼布格 0.24、條紋 0.57、點綴陣列 0.20–0.28、有機花卉 0.13–0.18。
_AXIS_STRUCTURE_MIN = 0.22
# 梯度方向偏離水平／垂直超過 atan(1/8)≈7° 即視為斜向。
# 放寬到 1/4（14°）會漏掉常見的 9°～10° 錯切扳斜。
_AXIS_ANISOTROPY = 8.0
# 逐列掃描時每塊的目標元素數，控制暫時記憶體用量
_SCAN_BLOCK_ELEMS = 4_000_000


def _downscale(arr: np.ndarray, max_side: int) -> np.ndarray:
    """通道數無關的縮圖（CMYK 四通道也能用）。"""
    h, w = arr.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return arr
    s = m / float(max_side)
    return cv2.resize(
        arr,
        (max(32, int(round(w / s))), max(32, int(round(h / s)))),
        interpolation=cv2.INTER_AREA,
    )


def _line_signal(arr: np.ndarray, axis: int) -> np.ndarray:
    """
    環形相鄰線色差序列，索引 0 為 wrap 線。

    axis=1 掃垂直線（逐列比較），axis=0 掃水平線（逐行比較）。

    一律在原解析度掃描：縮圖會把單像素寬的接縫和鄰居平均掉，讓搬家後的
    縫看起來只有一半強度，半幅就又能靠縮圖誤差得分。改用分塊掃描控制
    記憶體，不改精度。
    """
    a = arr if axis == 1 else np.swapaxes(arr, 0, 1)
    h, w = a.shape[:2]
    c = a.shape[2] if a.ndim == 3 else 1
    sig = np.empty(w, dtype=np.float64)
    sig[0] = float(
        np.abs(a[:, 0].astype(np.int16) - a[:, -1].astype(np.int16)).mean()
    )
    step = max(2, _SCAN_BLOCK_ELEMS // max(h * c, 1))
    for x0 in range(0, w - 1, step):
        x1 = min(w - 1, x0 + step)
        block = a[:, x0 : x1 + 1].astype(np.int16)
        d = np.abs(np.diff(block, axis=1)).mean(axis=(0, 2))
        sig[1 + x0 : 1 + x1] = d
    return sig


@dataclass(frozen=True)
class LineDefect:
    """環形單元中最差的一條線。"""

    diff: float
    """該線與相鄰線的平均色差。"""

    at: float
    """相對位置 0–1。0 表示落在 wrap 邊界，0.5 附近是典型的半幅搬家縫。"""

    baseline: float
    """圖案自身的典型線差（中位數）。格紋、條紋天生較高。"""

    @property
    def excess(self) -> float:
        """超出圖案典型線差的量。用於跨圖比較時的絕對門檻。"""
        return max(0.0, self.diff - self.baseline)

    @property
    def on_wrap(self) -> bool:
        return self.at < 0.02 or self.at > 0.98

    def describe(self) -> str:
        where = "wrap" if self.on_wrap else f"{self.at:.2f}"
        return f"{self.diff:.0f}@{where}(+{self.excess:.0f})"


def _worst_line(sig: np.ndarray) -> tuple[int, float, float]:
    baseline = float(np.median(sig))
    idx = int(np.argmax(sig))
    return idx, float(sig[idx]), baseline


def worst_lines(arr: np.ndarray) -> tuple[LineDefect, LineDefect]:
    """
    回傳 (垂直方向最差線, 水平方向最差線)。

    把單元視為環形，wrap 線與所有內部線一起評比。半幅／錯位補白把縫
    搬到內部時，最差線的值不會變好，只有位置改變。
    """
    out: list[LineDefect] = []
    for axis in (1, 0):
        sig = _line_signal(arr, axis)
        idx, diff, baseline = _worst_line(sig)
        out.append(
            LineDefect(diff=diff, at=idx / float(sig.size), baseline=baseline)
        )
    return out[0], out[1]


def worst_line_score(arr: np.ndarray) -> float:
    """兩軸最差線之和。舊介面，保留給既有呼叫端。"""
    v, h = worst_lines(arr)
    return v.diff + h.diff


# 局部基準的取樣半徑（線數）。太小會被單一硬邊帶偏，太大就退化成全圖中位數。
_LOCAL_HALF = 8
# 圖案邊緣的中央線差，相對兩肩通常在兩倍出頭以內。
_EDGE_SHOULDER = 2.2


def _line_excess(sig: np.ndarray) -> np.ndarray:
    """
    每條線超出「該處應有的線差」的量。

    基準取三者的較大值，缺一不可：

    - **全圖中位數**：圖案自身的紋理起伏，低於它一定看不見。
    - **局部中位數**：繁忙區域本來就會遮蔽瑕疵，這與人眼一致。
    - **兩肩線差的倍數**：真正的分辨依據。圖案的一道邊是有寬度的，中央
      線差高、兩側也跟著抬高；人造接縫則是孤立尖峰，兩肩仍是正常內容。

    第三項不能省。最小誤差切之後的對邊其實是原圖裡相鄰的兩欄，本來就
    連續，但若剛好落在圖案的一道邊上（實測線差 14.2，兩肩 6.0／6.2，
    而全圖中位數只有 2.1），只看中位數就會把一個完美的結果判成有縫。
    真接縫則相反——某張圖的 wrap 線差 84.7，兩肩只有 5。
    """
    ref = median_filter(sig, size=2 * _LOCAL_HALF + 1, mode="wrap")
    np.maximum(ref, float(np.median(sig)), out=ref)
    shoulder = np.maximum(np.roll(sig, 1), np.roll(sig, -1)) * _EDGE_SHOULDER
    np.maximum(ref, shoulder, out=ref)
    return np.maximum(0.0, sig - ref)


@dataclass(frozen=True)
class SeamReport:
    """單元圖的接縫體檢表。所有 excess 皆為「超出該處應有線差」的量。"""

    wrap_v: float
    wrap_h: float
    baseline_v: float
    baseline_h: float
    excess_v: float
    excess_h: float
    internal_v: float
    internal_h: float
    internal_at_v: float
    internal_at_h: float

    @property
    def wrap_excess(self) -> float:
        """兩軸 wrap 超出量的較大者。使用者第一眼看到的那條縫。"""
        return max(self.excess_v, self.excess_h)

    @property
    def internal_excess(self) -> float:
        """兩軸內部最差線超出量的較大者。抓半幅搬家與切線斷裂。"""
        return max(self.internal_v, self.internal_h)

    @property
    def wrap_raw(self) -> float:
        return self.wrap_v + self.wrap_h

    def describe(self) -> str:
        return (
            f"wrap {self.wrap_v:.1f}+{self.wrap_h:.1f}"
            f"(超出 {self.wrap_excess:.1f})"
            f" 內部 {self.internal_excess:.1f}"
            f"@({self.internal_at_v:.2f},{self.internal_at_h:.2f})"
        )


def seam_report(arr: np.ndarray) -> SeamReport:
    """一次掃完兩軸，同時取得 wrap 線與內部最差線的超出量。"""
    vals: list[tuple[float, float, float, float, float]] = []
    for axis in (1, 0):
        sig = _line_signal(arr, axis)
        exc = _line_excess(sig)
        if sig.size > 1:
            idx = int(np.argmax(exc[1:])) + 1
            internal = float(exc[idx])
            at = idx / float(sig.size)
        else:
            internal = float(exc[0])
            at = 0.0
        vals.append(
            (float(sig[0]), float(np.median(sig)), float(exc[0]), internal, at)
        )
    (wv, bv, ev, iv, av), (wh, bh, eh, ih, ah) = vals
    return SeamReport(
        wrap_v=wv,
        wrap_h=wh,
        baseline_v=bv,
        baseline_h=bh,
        excess_v=ev,
        excess_h=eh,
        internal_v=iv,
        internal_h=ih,
        internal_at_v=av,
        internal_at_h=ah,
    )


def wrap_excess(arr: np.ndarray) -> float:
    """單元自己接自己時，接縫超出圖案典型線差多少。0 表示看不見。"""
    return seam_report(arr).wrap_excess


def _luminance(arr: np.ndarray) -> np.ndarray:
    a = arr.astype(np.float32, copy=False)
    if a.ndim == 2:
        return a
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def ink_mask(arr: np.ndarray, *, delta: float = 18.0) -> np.ndarray:
    """相對畫面中位亮度的『有圖案』遮罩。不依賴背景色，深底淺花也能用。"""
    lum = _luminance(arr)
    return np.abs(lum - float(np.median(lum))) > delta


def ink_frac(arr: np.ndarray) -> float:
    return float(ink_mask(arr).mean())


def edge_void_ratio(arr: np.ndarray, *, band_frac: float = 0.05) -> float:
    """
    邊緣帶相對內部少了多少圖案。

    清邊補花若把四邊點綴清掉卻補不回密度，wrap 色差是 0（白接白），
    2×2 正中央卻出現十字空洞。這是色差接縫量不到的。
    """
    ink = ink_mask(arr)
    h, w = ink.shape
    b = max(8, int(round(min(h, w) * band_frac)))
    b = min(b, h // 4, w // 4)
    edge = np.zeros((h, w), dtype=bool)
    edge[:b, :] = True
    edge[h - b :, :] = True
    edge[:, :b] = True
    edge[:, w - b :] = True
    interior = ~edge
    e = float(ink[edge].mean()) if edge.any() else 0.0
    i = float(ink[interior].mean()) if interior.any() else 0.0
    if i < 0.04:
        return 0.0
    return float(max(0.0, 1.0 - e / i))


def _strip_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return d.mean(axis=-1) if d.ndim == 3 else d


def wrap_hotspot_axes(arr: np.ndarray) -> tuple[float, float]:
    """
    wrap 線上局部最熱的色差，回傳 (左右縫, 上下縫)。

    `wrap_excess` 是整條縫的平均超出量。狐狸頭、漿果被剖開時，九成接縫
    是地色對地色（平均 ≈ 0），只有圖章那一段炸掉。用 90 分位減 50 分位
    抓這種長尾；再跟內部相鄰線的 90 分位比，避免把繁忙紋理本身當成縫。
    """
    h, w = arr.shape[:2]
    if h < 8 or w < 8:
        return 0.0, 0.0

    def axis_hot(wrap: np.ndarray, interiors: list[np.ndarray]) -> float:
        p90 = float(np.percentile(wrap, 90))
        p50 = float(np.percentile(wrap, 50))
        tail = max(0.0, p90 - p50)
        base = float(np.median([np.percentile(s, 90) for s in interiors]))
        vs_int = max(0.0, p90 - base)
        return max(tail, vs_int)

    interiors_v = [
        _strip_diff(arr[:, x], arr[:, x + 1])
        for x in (1, max(2, w // 4), w // 2, min(w - 2, 3 * w // 4))
    ]
    interiors_h = [
        _strip_diff(arr[y], arr[y + 1])
        for y in (1, max(2, h // 4), h // 2, min(h - 2, 3 * h // 4))
    ]
    return (
        axis_hot(_strip_diff(arr[:, 0], arr[:, -1]), interiors_v),
        axis_hot(_strip_diff(arr[0], arr[-1]), interiors_h),
    )


def wrap_hotspot(arr: np.ndarray) -> float:
    """兩軸 wrap 熱點的較大者。"""
    hot_v, hot_h = wrap_hotspot_axes(arr)
    return max(hot_v, hot_h)


def _longest_true_run(mask: np.ndarray) -> int:
    if mask.size == 0 or not np.any(mask):
        return 0
    m = np.empty(mask.size + 2, dtype=np.uint8)
    m[0] = 0
    m[-1] = 0
    m[1:-1] = np.asarray(mask, dtype=np.uint8)
    d = np.diff(m.view(np.int8))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    if starts.size == 0:
        return 0
    return int((ends - starts).max())


def wrap_orphan_run(arr: np.ndarray) -> int:
    """
    wrap 兩側「一側有厚圖案、對邊幾乎沒有」的最長連續段（像素）。

    只看最外 3px 會被對邊另一顆圖章的邊緣墨點騙過：半隻刺蝟對邊只要
    有一點抗鋸齒，XOR 就斷掉。改看從邊緣往內連續有墨的深度，半截圖章
    （一側深入、對邊幾乎是地色）才會連成一段。
    """
    ink = ink_mask(arr)
    h, w = ink.shape
    if h < 8 or w < 8:
        return 0
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
    return max(_longest_true_run(v), _longest_true_run(hz))


def wrap_both_thick_run(arr: np.ndarray) -> int:
    """wrap 兩側都有厚圖案的最長連續段。兩隻半截對上時很長，清邊後的空邊是 0。"""
    ink = ink_mask(arr)
    h, w = ink.shape
    if h < 8 or w < 8:
        return 0
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
    thick = 8
    v = (left >= thick) & (right >= thick)
    hz = (top >= thick) & (bot >= thick)
    return max(_longest_true_run(v), _longest_true_run(hz))


def _union_find_parent(n: int) -> np.ndarray:
    return np.arange(n, dtype=np.int32)


def _uf_find(parent: np.ndarray, a: int) -> int:
    while parent[a] != a:
        parent[a] = parent[parent[a]]
        a = int(parent[a])
    return a


def _uf_union(parent: np.ndarray, a: int, b: int) -> None:
    ra, rb = _uf_find(parent, a), _uf_find(parent, b)
    if ra != rb:
        parent[rb] = ra


def _unwrap_torus_mask(mask: np.ndarray) -> np.ndarray:
    """把跨縫的連通塊滾到不切開的位置，才能量凸包實心度。"""
    m = mask
    cols = m.any(axis=0)
    if cols.size and bool(cols[0]) and bool(cols[-1]):
        gap = np.flatnonzero(~cols)
        if gap.size:
            m = np.roll(m, -int(gap[0]), axis=1)
    rows = m.any(axis=1)
    if rows.size and bool(rows[0]) and bool(rows[-1]):
        gap = np.flatnonzero(~rows)
        if gap.size:
            m = np.roll(m, -int(gap[0]), axis=0)
    return m


def _component_solidity(mask: np.ndarray) -> float | None:
    roi = mask.astype(np.uint8)
    cnts, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    area = float(cv2.contourArea(cnt))
    if area < 200.0:
        return None
    hull = cv2.convexHull(cnt)
    hull_area = float(cv2.contourArea(hull))
    if hull_area < 1.0:
        return None
    return area / hull_area


def motif_fragment_ratio(arr: np.ndarray) -> float:
    """
    尺寸相近的圖章裡，實心度明顯低於同伴的比例。

    雪花／枝葉大家都凹，沒有離群值；雪人缺一塊、刺蝟被掏空則會低於同伴。
    跨縫被切開的同一朵花先在環面上合併再比，避免把正確的四方連續判成殘缺。
    """
    ink = ink_mask(arr).astype(np.uint8)
    h, w = ink.shape
    if min(h, w) < 32:
        return 0.0
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    if n <= 2:
        return 0.0
    parent = _union_find_parent(n)
    for y in range(h):
        a, b = int(labels[y, 0]), int(labels[y, -1])
        if a and b:
            _uf_union(parent, a, b)
    for x in range(w):
        a, b = int(labels[0, x]), int(labels[-1, x])
        if a and b:
            _uf_union(parent, a, b)

    groups: dict[int, list[int]] = {}
    for i in range(1, n):
        groups.setdefault(_uf_find(parent, i), []).append(i)

    recs: list[tuple[int, float]] = []
    for ids in groups.values():
        area = int(sum(int(stats[i, cv2.CC_STAT_AREA]) for i in ids))
        if area < 200:
            continue
        member = np.isin(labels, np.asarray(ids, dtype=labels.dtype))
        solid = _component_solidity(_unwrap_torus_mask(member))
        if solid is None:
            continue
        recs.append((area, solid))
    if len(recs) < 4:
        return 0.0

    areas = np.array([a for a, _ in recs], dtype=np.float64)
    sols = np.array([s for _, s in recs], dtype=np.float64)
    flagged = 0
    for i, (area, solid) in enumerate(recs):
        peer = (areas >= area / 1.55) & (areas <= area * 1.55)
        if int(peer.sum()) < 4:
            continue
        med = float(np.median(sols[peer]))
        med_a = float(np.median(areas[peer]))
        # 缺一塊的圖章通常面積也比較小；同面積但帽子造型不同的雪人不要算殘缺。
        if solid < med - 0.14 and solid < 0.78 and area < med_a * 0.90:
            flagged += 1
    return float(flagged) / float(len(recs))


def _typical_stamp_area(areas: list[int]) -> float:
    """內部圖章的代表面積。碎點很多時改看主圖章那一檔。"""
    if not areas:
        return 0.0
    vals = np.asarray(areas, dtype=np.float64)
    if vals.size < 4:
        return float(np.median(vals))
    med = float(np.median(vals))
    p90 = float(np.percentile(vals, 90))
    # 與補花模板同一套：雪點把中位數拉下去時改看大圖章。
    if p90 > med * 6.0:
        upper = vals[vals >= p90 * 0.45]
        return float(np.median(upper if upper.size else vals))
    cut = float(np.percentile(vals, 70))
    big = vals[vals >= cut]
    return float(np.median(big if big.size else vals))


def _ids_span_frame(stats: np.ndarray, ids: list[int], h: int, w: int) -> bool:
    """外接框幾乎拉滿一條邊：沿邊底紋，不是單顆圖章。"""
    max_w = max(int(stats[i, cv2.CC_STAT_WIDTH]) for i in ids)
    max_h = max(int(stats[i, cv2.CC_STAT_HEIGHT]) for i in ids)
    return max_w >= int(w * 0.80) or max_h >= int(h * 0.80)


def _wrap_group_color_mismatch(
    arr: np.ndarray, labels: np.ndarray, ids: list[int]
) -> bool:
    """
    環面已併成「一隻圖章」，但對邊墨點顏色對不上。

    面積加起來像真跨縫時，glued／two_wholes 都不會開火；雪花這種同色
    不同形會在 wrap 上留下長尾色差，2×2 仍是兩朵假接。只給 leftover
    清掉重貼，不把 wrap_cut 比值拉高（真跨縫細線也曾被比值誤殺）。
    """
    if arr.ndim != 3 or not ids:
        return False
    nlab = int(labels.max()) + 1
    lut = np.zeros(nlab, dtype=bool)
    for i in ids:
        ii = int(i)
        if 0 < ii < nlab:
            lut[ii] = True
    mem = lut[labels]
    h, w = labels.shape[:2]

    def _hot(both: np.ndarray, delta: np.ndarray) -> bool:
        n = int(np.count_nonzero(both))
        if n < 6:
            return False
        vals = delta[both]
        p75 = float(np.percentile(vals, 75))
        p90 = float(np.percentile(vals, 90))
        return p75 >= 28.0 or (n >= 12 and p90 >= 45.0)

    dv = np.abs(arr[:, 0].astype(np.int16) - arr[:, -1].astype(np.int16)).mean(
        axis=-1
    )
    dh = np.abs(arr[0].astype(np.int16) - arr[-1].astype(np.int16)).mean(axis=-1)
    return _hot(mem[:, 0] & mem[:, -1], dv) or _hot(mem[0] & mem[-1], dh)


@dataclass
class WrapCutRepair:
    """wrap_cut 量測，以及清邊補花第二輪該清掉的連通域。"""

    ratio: float
    typical: float
    labels: np.ndarray
    stats: np.ndarray
    centroids: np.ndarray
    kill_ids: list[int]


def _wrap_group_close_k(stats: np.ndarray, n: int) -> int:
    """與圖章分組相近的閉合核：把掌墊＋趾合成一枚，但不把格點黏成一片。"""
    areas = [
        int(stats[i, cv2.CC_STAT_AREA])
        for i in range(1, n)
        if int(stats[i, cv2.CC_STAT_AREA]) >= 200
    ]
    if len(areas) < 2:
        return 0
    hero = float(np.percentile(np.asarray(areas, dtype=np.float64), 85))
    k = int(round(0.28 * np.sqrt(max(hero, 1.0))))
    k = max(3, k)
    if k % 2 == 0:
        k += 1
    return k


def _merge_near_wrap_groups(
    parent: np.ndarray,
    ink: np.ndarray,
    labels: np.ndarray,
    stats: np.ndarray,
    n: int,
) -> None:
    """
    環面閉合：只合併「碰到畫框」的鄰近連通塊。

    腳印跨縫時掌在右邊、趾在左邊，逐塊 union-find 對不上同一列，
    會被算成切圖。3×3 閉合後它們是同一枚；內部格點不碰框，不合併。
    """
    h, w = ink.shape
    if n <= 2 or h * w > 4_000_000:
        return
    k = _wrap_group_close_k(stats, n)
    if k < 3:
        return
    # 2×2 平鋪再閉合：跨縫的掌＋趾會連成同一標籤；只 pad 一圈的話，
    # 對邊本體仍在原圖裡各算各的。
    tiled = np.tile(ink.astype(np.uint8), (2, 2))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    closed = cv2.morphologyEx(tiled, cv2.MORPH_CLOSE, ker)
    _nc, lab = cv2.connectedComponents(closed, connectivity=8)
    seam_labs = {
        int(v)
        for sl in (lab[:, w - 1], lab[:, w], lab[h - 1, :], lab[h, :])
        for v in np.unique(sl)
        if int(v) > 0
    }
    if not seam_labs:
        return
    buckets: dict[int, list[int]] = {}
    seen: dict[int, set[int]] = {}
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < 80:
            continue
        x0 = int(stats[i, cv2.CC_STAT_LEFT])
        y0 = int(stats[i, cv2.CC_STAT_TOP])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        roi = labels[y0 : y0 + bh, x0 : x0 + bw] == i
        keys: set[int] = set()
        for dy, dx in ((0, 0), (0, w), (h, 0), (h, w)):
            votes = lab[y0 + dy : y0 + dy + bh, x0 + dx : x0 + dx + bw][roi]
            if votes.size:
                keys.add(int(np.bincount(votes.ravel()).argmax()))
        for key in keys:
            if key not in seam_labs:
                continue
            if i in seen.setdefault(key, set()):
                continue
            seen[key].add(i)
            buckets.setdefault(key, []).append(i)
    for ids in buckets.values():
        if len(ids) < 2:
            continue
        a0 = ids[0]
        for b in ids[1:]:
            _uf_union(parent, a0, b)


def wrap_cut_repair(arr: np.ndarray) -> WrapCutRepair:
    """
    碰邊卻沒在環面上接到對邊的圖章，相對內部典型圖章有多大。

    半隻刺蝟停在左緣、右緣是地色時，同伴離群比不到它（面積只有一半，
    進不了 1.55× 分桶），但這一項會接近 0.5–1。真正跨縫對接的圖章
    會在對邊合併，不會進分子。

    對邊同列各有一顆完整圖章時，舊邏輯會把它們環面合併、當成接好。
    合併後面積明顯大於內部典型、或兩邊都幾乎是整顆，仍算切圖。
    對邊各一截不同圖章、顏色對不上，也當成假接。

    帶寬上限 8px：15px（短邊/80）會把離框 10px 的完整小圖章當成切圖。
    """
    empty = WrapCutRepair(
        0.0,
        0.0,
        np.zeros((0, 0), dtype=np.int32),
        np.zeros((0, 5), dtype=np.int32),
        np.zeros((0, 2), dtype=np.float64),
        [],
    )
    ink = ink_mask(arr).astype(np.uint8)
    h, w = ink.shape
    if min(h, w) < 32:
        return empty
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(
        ink, connectivity=8
    )
    if n <= 2:
        return empty
    parent = _union_find_parent(n)
    for y in range(h):
        a, b = int(labels[y, 0]), int(labels[y, -1])
        if a and b:
            _uf_union(parent, a, b)
    for x in range(w):
        a, b = int(labels[0, x]), int(labels[-1, x])
        if a and b:
            _uf_union(parent, a, b)
    _merge_near_wrap_groups(parent, ink, labels, stats, n)

    band = max(2, min(8, min(h, w) // 80))
    left = np.unique(labels[:, :band])
    right = np.unique(labels[:, w - band :])
    top = np.unique(labels[:band, :])
    bot = np.unique(labels[h - band :, :])
    edge_ids = {
        int(i)
        for i in np.concatenate([left, right, top, bot])
        if int(i) > 0
    }
    if not edge_ids:
        return WrapCutRepair(0.0, 0.0, labels, stats, centroids, [])

    merged_roots: set[int] = set()
    for a in left:
        if a == 0:
            continue
        ra = _uf_find(parent, int(a))
        if any(_uf_find(parent, int(b)) == ra for b in right if b):
            merged_roots.add(ra)
    for a in top:
        if a == 0:
            continue
        ra = _uf_find(parent, int(a))
        if any(_uf_find(parent, int(b)) == ra for b in bot if b):
            merged_roots.add(ra)

    groups: dict[int, list[int]] = {}
    for i in range(1, n):
        groups.setdefault(_uf_find(parent, i), []).append(i)

    for root, ids in groups.items():
        left_hit = any(int(stats[i, cv2.CC_STAT_LEFT]) <= band for i in ids)
        right_hit = any(
            int(stats[i, cv2.CC_STAT_LEFT]) + int(stats[i, cv2.CC_STAT_WIDTH])
            >= w - band
            for i in ids
        )
        top_hit = any(int(stats[i, cv2.CC_STAT_TOP]) <= band for i in ids)
        bot_hit = any(
            int(stats[i, cv2.CC_STAT_TOP]) + int(stats[i, cv2.CC_STAT_HEIGHT])
            >= h - band
            for i in ids
        )
        if (left_hit and right_hit) or (top_hit and bot_hit):
            merged_roots.add(root)

    canvas = float(h * w)
    field = canvas * 0.10
    interior_areas: list[int] = []
    wrap_groups: list[tuple[int, list[int]]] = []
    unmatched_groups: list[tuple[int, list[int]]] = []
    unmatched_areas: list[int] = []
    for root, ids in groups.items():
        area = int(sum(int(stats[i, cv2.CC_STAT_AREA]) for i in ids))
        if area < 200:
            continue
        # 滿版連通底紋／沿邊長條不是圖章：不當切圖，也不拉低 typical。
        if area >= field or _ids_span_frame(stats, ids, h, w):
            continue
        on_edge = any(i in edge_ids for i in ids)
        if not on_edge:
            interior_areas.append(area)
            continue
        if root in merged_roots:
            wrap_groups.append((area, ids))
            continue
        unmatched_areas.append(area)
        unmatched_groups.append((area, ids))

    typical = _typical_stamp_area(interior_areas)
    if typical < 80.0:
        typical = _typical_stamp_area([a for a, _ in wrap_groups])
    if typical < 80.0:
        typical = float(max(unmatched_areas, default=0) or 0)
    if typical < 80.0:
        return WrapCutRepair(0.0, 0.0, labels, stats, centroids, [])

    fake_groups: list[tuple[int, list[int]]] = []
    for area, ids in wrap_groups:
        parts = [
            int(stats[i, cv2.CC_STAT_AREA])
            for i in ids
            if int(stats[i, cv2.CC_STAT_AREA]) >= 200
        ]
        glued = area > typical * 1.80
        two_wholes = (
            len(parts) >= 2
            and min(parts) > typical * 0.62
            and area > typical * 1.70
        )
        # 對邊各一條杏仁殘片：環面併起來面積仍遠小於完整圖章，舊邏輯
        # 當成「接好」。波點錯相位就是這樣綠燈的。
        sliver_join = (
            len(parts) >= 2
            and area < typical * 0.75
            and sum(1 for i in ids if _cc_is_edge_sliver(stats, i, typical)) >= 2
        )
        color_fake = _wrap_group_color_mismatch(arr, labels, ids)
        if glued or two_wholes:
            unmatched_areas.append(area)
            fake_groups.append((area, ids))
        elif sliver_join:
            # 用真實合併面積。舊的 0.55×typical 地板會讓補花後剩下的
            # 碎點永遠卡在 WRAP_CUT_MAX 之上，雪人／毛衣已經跨縫也過不了。
            unmatched_areas.append(area)
            fake_groups.append((area, ids))
        elif color_fake:
            fake_groups.append((area, ids))

    kill_ids: list[int] = []
    seen: set[int] = set()
    for _, ids in unmatched_groups + fake_groups:
        for i in ids:
            ii = int(i)
            if ii in seen or int(stats[ii, cv2.CC_STAT_AREA]) < 80:
                continue
            seen.add(ii)
            kill_ids.append(ii)

    if not unmatched_areas:
        return WrapCutRepair(0.0, typical, labels, stats, centroids, kill_ids)
    return WrapCutRepair(
        float(min(max(unmatched_areas) / typical, 8.0)),
        typical,
        labels,
        stats,
        centroids,
        kill_ids,
    )


def wrap_cut_ratio(arr: np.ndarray) -> float:
    return wrap_cut_repair(arr).ratio


def wrap_density_ratio(arr: np.ndarray) -> float:
    """
    2×2 十字接縫帶前景 / 四個象限內部前景。
    清邊補花若把左右兩列都留在縫上，正中央會明顯更密，比值 > 1。
    只看單元外緣會誤判：殘片刪掉後畫框變疏，2×2 卻仍擠成雙行。
    """
    ink = ink_mask(arr)
    h, w = ink.shape
    if min(h, w) < 32:
        return 1.0
    tiled = np.tile(ink.astype(bool), (2, 2))
    band = max(16, min(h, w) // 10)
    cross = np.zeros(tiled.shape, dtype=bool)
    cross[:, w - band : w + band] = True
    cross[h - band : h + band, :] = True
    inner = np.zeros(tiled.shape, dtype=bool)
    for y0 in (0, h):
        for x0 in (0, w):
            inner[y0 + band : y0 + h - band, x0 + band : x0 + w - band] = True
    inner_mean = float(np.mean(tiled[inner])) if inner.any() else 0.0
    if inner_mean < 1e-4:
        return 1.0
    return float(np.mean(tiled[cross])) / inner_mean


def _cc_is_edge_sliver(stats: np.ndarray, i: int, typical: float) -> bool:
    """碰邊殘片：面積遠小於同伴，外接框又細長。"""
    area = int(stats[i, cv2.CC_STAT_AREA])
    bw = int(stats[i, cv2.CC_STAT_WIDTH])
    bh = int(stats[i, cv2.CC_STAT_HEIGHT])
    if area < 200 or typical < 80.0 or area >= typical * 0.42:
        return False
    diam = max(8.0, float(np.sqrt(4.0 * typical / np.pi)))
    thin_v = bw <= max(8, int(diam * 0.38)) and bh >= max(bw * 1.7, diam * 0.45)
    thin_h = bh <= max(8, int(diam * 0.38)) and bw >= max(bh * 1.7, diam * 0.45)
    return bool(thin_v or thin_h)


def _interior_bg_runs(occupied: np.ndarray) -> np.ndarray:
    """一維佔用列的內部地色空檔長度（不含兩端接到 wrap 的溝）。"""
    m = np.asarray(occupied, dtype=np.uint8)
    if m.size < 4 or not m.any():
        return np.zeros(0, dtype=np.int32)
    padded = np.empty(m.size + 2, dtype=np.uint8)
    padded[0] = 1
    padded[-1] = 1
    padded[1:-1] = m
    d = np.diff(padded.view(np.int8))
    starts = np.flatnonzero(d == -1)
    ends = np.flatnonzero(d == 1)
    if starts.size == 0:
        return np.zeros(0, dtype=np.int32)
    return (ends - starts).astype(np.int32)


def wrap_gutter_error(arr: np.ndarray) -> float:
    """
    接縫落在地色溝時，溝寬相對內部圖章間距差多少。

    wrap 色差可以是 0：週期裁切把縫滾到兩列波點中間的底色，左右都是綠
    接綠。但單元寬不是 X 週期的整數倍時，縫上的溝會比內部密得多（兩顆
    圓幾乎碰上，2×2 看起來像杏仁殘片）或疏得多。圖章真的跨縫（最外 8px
    有墨）時這項為 0，改由 wrap_cut 負責。
    """
    ink = ink_mask(arr)
    h, w = ink.shape
    if min(h, w) < 32:
        return 0.0
    band = max(2, min(8, min(h, w) // 80))
    err = 0.0
    if not bool(ink[:, :band].any()) and not bool(ink[:, w - band :].any()):
        err = max(err, _axis_gutter_error(ink.any(axis=0)))
    if not bool(ink[:band, :].any()) and not bool(ink[h - band :, :].any()):
        err = max(err, _axis_gutter_error(ink.any(axis=1)))
    return float(err)


def _axis_gutter_error(occupied: np.ndarray) -> float:
    occ = np.asarray(occupied, dtype=bool)
    idx = np.flatnonzero(occ)
    if idx.size < 4:
        return 0.0
    wrap_gap = int(idx[0] + (occ.size - 1 - idx[-1]))
    gaps = _interior_bg_runs(occ)
    if gaps.size == 0:
        return 0.0
    stamp = _interior_bg_runs(~occ)
    typical_stamp = (
        float(np.median(stamp[stamp >= 12])) if np.any(stamp >= 12) else 0.0
    )
    min_gap = max(16, int(round(typical_stamp * 0.15))) if typical_stamp else 16
    large = gaps[gaps >= min_gap]
    if large.size < 3:
        return 0.0
    typical_gap = float(np.median(large))
    if typical_gap < 16.0:
        return 0.0
    mad = float(np.median(np.abs(large.astype(np.float64) - typical_gap)))
    # 散點間距本身就亂，沒有「該有的溝寬」可比較。
    if mad > typical_gap * 0.35:
        return 0.0
    return abs(float(wrap_gap) - typical_gap) / typical_gap


def wrap_period_remainder(arr: np.ndarray) -> float:
    """
    單元邊長對強週期的最短餘數比例。

    滿幅裁切把高度做成 3×525、寬度仍是 2048 時，Y 整除、X 餘 140px。
    接縫色差可以是 0（地色溝對上），波點格子卻對不齊。只在兩軸都偵測到
    強週期時才看該軸；單軸條紋沒有另一軸週期，餘數視為 0。
    """
    from app.processor import (
        _len_period_rem,
        _luma_square_grid_pitch,
        _luminance_map,
        _strong_axis_periods,
    )

    gray = _luminance_map(arr)
    h, w = gray.shape[:2]
    fine = _luma_square_grid_pitch(arr)
    if fine is not None:
        return max(_len_period_rem(w, fine[0]), _len_period_rem(h, fine[1]))
    xs = _strong_axis_periods(gray, 0)[:1]
    ys = _strong_axis_periods(gray, 1)[:1]

    def _frac(length: int, periods: list[int]) -> float:
        if not periods:
            return 0.0
        return _len_period_rem(length, int(periods[0]))

    return max(_frac(w, xs), _frac(h, ys))


@dataclass(frozen=True)
class ColorShift:
    """兩張同尺寸圖之間的色偏。單位為 0–255 階。"""

    mean: float
    p99: float
    peak: float
    lowfreq: float
    """低頻分量的最大值。整體色偏／暈影會落在這裡，是肉眼最敏感的部分。"""

    @property
    def visible(self) -> bool:
        """印刷品實務上約 2 階以內看不出來，低頻色塊則更敏感。"""
        return self.mean > 1.2 or self.p99 > 6.0 or self.lowfreq > 3.0

    def describe(self) -> str:
        return (
            f"色偏 mean {self.mean:.2f} p99 {self.p99:.1f} "
            f"低頻 {self.lowfreq:.1f}"
        )


_NO_SHIFT = ColorShift(mean=0.0, p99=0.0, peak=0.0, lowfreq=0.0)


def color_shift(before: np.ndarray, after: np.ndarray) -> ColorShift:
    """
    量測一步處理造成的色偏。兩張必須同尺寸。

    低頻分量另外算：梯度域週期化會疊上一層極平滑的修正場，逐像素差看
    起來很小，整片色調卻可能被拉走，那才是使用者說的「色差」。
    """
    if before.shape != after.shape:
        raise ValueError(f"色偏需同尺寸：{before.shape} vs {after.shape}")
    if before is after:
        return _NO_SHIFT
    d = np.abs(after.astype(np.int16) - before.astype(np.int16))
    if not d.any():
        return _NO_SHIFT
    small_b = _downscale(before, 128).astype(np.float32)
    small_a = _downscale(after, 128).astype(np.float32)
    low = float(np.abs(small_a - small_b).max())
    return ColorShift(
        mean=float(d.mean()),
        p99=float(np.percentile(d, 99)),
        peak=float(d.max()),
        lowfreq=low,
    )


def tone_shift(src: np.ndarray, out: np.ndarray, *, edge_frac: float = 0.0) -> float:
    """
    整體色調偏移。

    只在同尺寸時比通道均值：那才是週期化／色彩轉換把整片拉亮、拉青的情況。
    最小誤差切與週期裁切會改尺寸，像素仍來自原稿；拿切掉的邊去跟整張原稿
    比均值，等於把「少了一條邊」判成偏色。週期化疊在改尺寸之後的色偏，
    由 `color_mean`／`color_low`／截斷閘門負責。

    清邊補花只改畫框附近：edge_frac>0 時只比內部，避免跨縫貼上的圖章
    把整張均值拉走。
    """
    if src.shape != out.shape:
        return 0.0
    a = src
    b = out
    if edge_frac > 0.0:
        h, w = src.shape[:2]
        band = int(round(min(h, w) * float(edge_frac)))
        band = min(max(band, 0), h // 4, w // 4)
        if band > 0:
            a = src[band : h - band, band : w - band]
            b = out[band : h - band, band : w - band]
    ch = src.shape[2] if src.ndim == 3 else 1
    am = a.reshape(-1, ch).mean(axis=0).astype(np.float64)
    bm = b.reshape(-1, ch).mean(axis=0).astype(np.float64)
    return float(np.abs(am - bm).max())


def design_error(
    src: np.ndarray,
    unit: np.ndarray,
    *,
    max_side: int = 256,
) -> float:
    """
    把單元平鋪回原尺寸、做相位對齊後與原圖比對的平均色差。

    這是判斷「設計有沒有被改壞」最直接的量：
    - 抓到真週期並裁切 → 平鋪回去等於原圖，值接近 0，即使單元只有原圖的
      百分之一大也不該被扣分。
    - 假週期把花距拉成兩倍、或把圖案切一半 → 平鋪回去對不上，值很大。
    - 最小誤差切少掉一條帶子 → 值小幅上升，符合它確實動了版面的事實。

    單純比面積或比像素改動量都會誤判：前者罰了正確的週期裁切，後者放過了
    「保留原圖但根本沒接上」。
    """
    s = src if src.ndim == 3 else src[:, :, None]
    u = unit if unit.ndim == 3 else unit[:, :, None]
    scale = min(1.0, max_side / float(max(s.shape[0], s.shape[1])))
    sh = max(16, int(round(s.shape[0] * scale)))
    sw = max(16, int(round(s.shape[1] * scale)))
    uh = max(4, int(round(u.shape[0] * scale)))
    uw = max(4, int(round(u.shape[1] * scale)))
    small_s = cv2.resize(
        s.mean(axis=2).astype(np.float32), (sw, sh), interpolation=cv2.INTER_AREA
    )
    small_u = cv2.resize(
        u.mean(axis=2).astype(np.float32), (uw, uh), interpolation=cv2.INTER_AREA
    )

    reps_y = int(np.ceil((sh + uh) / uh))
    reps_x = int(np.ceil((sw + uw) / uw))
    canvas = np.tile(small_u, (reps_y, reps_x))[: sh + uh - 1, : sw + uw - 1]
    if canvas.shape[0] < sh or canvas.shape[1] < sw:
        return float(np.abs(small_s - float(small_s.mean())).mean())

    res = cv2.matchTemplate(canvas, small_s, cv2.TM_SQDIFF)
    best = float(res.min())
    return float(np.sqrt(max(best, 0.0) / float(sh * sw)))


def axis_line_energy(
    arr: np.ndarray,
    *,
    max_side: int = _AXIS_ANALYSIS_SIDE,
) -> float:
    """
    軸向（水平／垂直）邊緣能量占全部邊緣能量的比例。

    拼布格線、條紋接近 1；有機花卉、手繪圖案偏低。
    錯切／旋轉會讓這個值下降。
    """
    small = _downscale(arr, max_side)
    g = small.astype(np.float32).mean(axis=2)
    gy, gx = np.gradient(g)
    mag = np.hypot(gx, gy)
    total = float(mag.sum())
    if total < 1e-6:
        return 0.0
    ax = np.abs(gx)
    ay = np.abs(gy)
    axis_aligned = np.maximum(ax, ay) >= np.minimum(ax, ay) * _AXIS_ANISOTROPY
    return float(mag[axis_aligned].sum() / total)


def geometry_fidelity(src: np.ndarray, out: np.ndarray) -> float:
    """
    輸出保留了多少原圖的軸向結構。1.0 為完整保留，越低表示越被扳斜／扭曲。

    原圖本身缺少軸向結構時回傳 1.0（此指標不適用，不應據此扣分）。

    同尺寸且過半像素沒動時也回傳 1.0：清邊補花、邊緣均值這類局部改寫
    會讓軸向能量比晃動，但並沒有把整張圖重映射，不應判成扳斜。
    錯切／旋轉／半幅滾動會讓幾乎所有像素換位，仍走能量比。
    """
    s = axis_line_energy(src)
    if s < _AXIS_STRUCTURE_MIN:
        return 1.0
    if src.shape == out.shape:
        unchanged = float(np.mean(np.all(src == out, axis=2)))
        if unchanged >= 0.5:
            return 1.0
    o = axis_line_energy(out)
    return float(min(1.0, o / max(s, 1e-6)))
