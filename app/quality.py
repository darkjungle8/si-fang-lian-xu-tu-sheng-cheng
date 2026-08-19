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


def tone_shift(src: np.ndarray, out: np.ndarray) -> float:
    """
    整體色調偏移。

    只在同尺寸時比通道均值：那才是週期化／色彩轉換把整片拉亮、拉青的情況。
    最小誤差切與週期裁切會改尺寸，像素仍來自原稿；拿切掉的邊去跟整張原稿
    比均值，等於把「少了一條邊」判成偏色。週期化疊在改尺寸之後的色偏，
    由 `color_mean`／`color_low`／截斷閘門負責。
    """
    if src.shape != out.shape:
        return 0.0
    ch = src.shape[2] if src.ndim == 3 else 1
    a = src.reshape(-1, ch).mean(axis=0).astype(np.float64)
    b = out.reshape(-1, ch).mean(axis=0).astype(np.float64)
    return float(np.abs(a - b).max())


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
