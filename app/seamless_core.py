"""保證四方連續的基本算子。

這裡的每個算子都**由構造保證**輸出可以四方連續拼接，不是「試試看再量」。
上層只負責挑選與把關，不必再為每種圖案寫禁令。

三個算子，依對原稿的破壞程度由小到大：

1. `wrap_mincut`：在邊緣重疊帶裡找一條最小誤差切線，把首尾縫合。
   全部像素都直接來自原稿，**色值零改動**，代價是單元尺寸縮小、
   接縫附近有少量內容重複。切線會繞開頭尾對不上的區域，避免把同一個
   圖案切成兩個來源（詳見 `_seam_cost`）。
2. `periodize`：Moisan periodic + smooth 分解。把不連續量表示成一個
   解 Poisson 方程的極平滑場並減掉，數學上保證完全週期。細節完全不糊，
   代價是疊上一層平滑的明暗修正。
3. 兩者串接：先用 1 把結構對齊，殘差就小到讓 2 的修正場幾乎為零。
   這是難圖的主力組合。

順序很要緊。單獨用 2 去修一條 80 階的大縫，修正場本身就會變成一大片
色偏（實測平均 18/255）；先用 1 把縫壓到 10 階左右再用 2，色偏平均只
剩 0.4/255。

所有算子都與通道數無關，可以直接吃 CMYK 四通道，不必先轉 RGB。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# 超過此像素數就改用降取樣解 Poisson。修正場本身極平滑，低解析度解完
# 再放大幾乎無損；116MP 的圖若走全解析度 FFT 會吃掉數 GB。
_POISSON_FULLRES_PIXELS = 24_000_000
_POISSON_COARSE_SIDE = 1024
# 邊界落差剖面的低通寬度（占剖面長度比例）。修正場只該處理整體光照落差。
_PROFILE_LOWPASS_FRAC = 0.05

# 重疊帶寬度候選（占該軸邊長比例）。窄帶保留較多內容，寬帶讓切線有更多
# 迴避空間；實際採用哪個由切線成本決定。
_BAND_FRACTIONS = (0.04, 0.08, 0.14, 0.22, 0.30, 0.36, 0.46)
_BAND_MIN_PX = 12
# 除固定比例外，另外掃出幾個「頭尾本來就吻合」的帶寬（見 `_shift_mismatch`）。
_BAND_SCAN_KEEP = 8
# 掃描時沿線方向的分塊數。帶寬只決定另一軸的相位，線方向可以大幅降取樣。
_BAND_SCAN_BLOCKS = 48

# 重疊帶裡「頭尾對不上」的區域，也就是有東西只存在於單側。切線從中穿過
# 時，同一個圖案會一半取自頭端、一半取自尾端，長出半隻大象、孤立的長頸鹿
# 犄角這種殘肢。這種殘肢的對邊色差是零（切線兩側在原稿裡本來就相鄰），
# 接縫指標一律量不到，只能在下刀前就避開。
#
# 反過來說，頭尾在該處長得一樣就可以放心切——兩側顯示的是同一個東西。
# 所以判準是「兩側是否一致」，不是「這裡有沒有圖案」，也就不需要背景色。
_MOTIF_DIFF_MIN = 8.0
# 對不上區域的中位數倍率。整體有色偏或條紋相位略錯的圖，滿帶都有小落差，
# 固定門檻會把整條帶子標成不可穿越，反而讓迴避機制失效。
_MOTIF_DIFF_MEDIAN = 2.5
# 遮罩外擴比例（占短邊）。貼著輪廓走會削掉抗鋸齒邊緣，留一點餘裕。
_MOTIF_DILATE_FRAC = 0.002
# 穿越一列圖案的代價。要大到讓切線寧可繞完整條帶子也不穿過去，但保持有限
# 值——密花圖整條帶子都對不上時無路可繞，那時就該退回比基礎色差。
_MOTIF_PENALTY = 1.0e7


def _as_3d(arr: np.ndarray) -> np.ndarray:
    return arr if arr.ndim == 3 else arr[:, :, None]


# --------------------------------------------------------------------------
# 1. 最小誤差切
# --------------------------------------------------------------------------


def _min_cost_path(cost: np.ndarray) -> np.ndarray:
    """在 (h, b) 成本圖裡找一條由上到下、每列橫移不超過 1 的最小成本路徑。"""
    h, b = cost.shape
    acc = np.empty((h, b), dtype=np.float64)
    back = np.empty((h, b), dtype=np.int32)
    acc[0] = cost[0]
    idx = np.arange(b)
    for y in range(1, h):
        prev = acc[y - 1]
        left = np.empty(b, dtype=np.float64)
        left[0] = np.inf
        left[1:] = prev[:-1]
        right = np.empty(b, dtype=np.float64)
        right[-1] = np.inf
        right[:-1] = prev[1:]
        stack = np.stack((left, prev, right))
        k = np.argmin(stack, axis=0)
        back[y] = idx + k - 1
        acc[y] = cost[y] + stack[k, idx]
    path = np.empty(h, dtype=np.int32)
    path[-1] = int(np.argmin(acc[-1]))
    for y in range(h - 1, 0, -1):
        path[y - 1] = back[y, path[y]]
    return path


def _seam_cost(arr: np.ndarray, band: int) -> tuple[np.ndarray, np.ndarray]:
    """
    重疊帶內每個位置的接合誤差（平方色距），與不可穿越遮罩。

    遮罩標出頭尾對不上的區域。切線只要不與某個對不上的連通塊相交，該塊就
    整塊落在切線的同一側，於是整塊取自頭端或整塊被尾端取代，不會被拼成
    兩個來源的混合體。切線每列只移動一格（八連通），四連通的區塊跨不過去，
    這個保證是純幾何的。
    """
    w = arr.shape[1]
    head = arr[:, :band].astype(np.float32)
    tail = arr[:, w - band :].astype(np.float32)
    diff = tail - head
    cost = (diff**2).sum(axis=2)
    mag = np.abs(diff).mean(axis=2)
    thr = max(_MOTIF_DIFF_MIN, float(np.median(mag)) * _MOTIF_DIFF_MEDIAN)
    mask = (mag > thr).astype(np.uint8)
    k = int(round(min(arr.shape[0], w) * _MOTIF_DILATE_FRAC)) | 1
    if k >= 3:
        mask = cv2.dilate(mask, np.ones((k, k), np.uint8))
    return cost, mask


def _cyclic_path(cost: np.ndarray) -> np.ndarray:
    """
    首尾必須落在同一位置的最小成本路徑。

    第二刀非做不可的約束。合成時是「這一列取自尾端、那一列取自首端」，
    若切線在第一列與最後一列停在不同位置，另一軸剛縫好的連續性就會在
    這條帶子裡被打斷——實測 12042×9678 那張的垂直對邊差因此從 1.9 彈到
    14.8。先跑一次自由路徑決定釘點，再跑一次把兩端釘住。

    阻擋值一定要用 `inf`。用「最大成本 × 帶寬」這種有限大數，在列數遠多
    於帶寬時（該圖是 10356 列對 774 帶寬）會被累積成本蓋過去，釘點形同
    虛設。

    釘點取兩端成本之和最低者。自由路徑的兩端是首選，但它只保證自己那一端
    便宜；釘點同時是首列與末列的落點，若那個位置在另一端剛好壓在圖案上，
    整條帶子就白繞了。
    """
    free = _min_cost_path(cost)
    ends = cost[0] + cost[-1]
    pin = min(
        (int(free[0]), int(free[-1]), int(np.argmin(ends))),
        key=lambda c: float(ends[c]),
    )
    pinned = cost.copy()
    for row in (0, -1):
        keep = pinned[row, pin]
        pinned[row] = np.inf
        pinned[row, pin] = keep
    return _min_cost_path(pinned)


def _splice_axis(arr: np.ndarray, band: int, path: np.ndarray) -> np.ndarray:
    """依既定切線把首尾縫起來。純像素搬移，適用於任何通道數。"""
    w = arr.shape[1]
    from_tail = np.arange(band)[None, :] < path[:, None]
    out = arr[:, : w - band].copy()
    out[:, :band] = np.where(
        from_tail[..., None], arr[:, w - band :], arr[:, :band]
    )
    return out


def _solve_axis(
    arr: np.ndarray, band: int, *, cyclic: bool
) -> tuple[np.ndarray, float, float, float]:
    """
    求一條切線，回傳 (切線, 平均色差, 穿過圖案的比例, 帶內圖案占比)。

    懲罰壓倒基礎色差，所以解出來的就是「穿過圖案最少」的路線。換句話說
    回傳的 cut 是該帶寬下的下限：cut 大於零代表這個帶寬繞不過去，不是
    演算法沒盡力。帶內圖案占比則說明繞不過去有多合理——整條帶子都是圖案
    時當然無路可繞。
    """
    h = arr.shape[0]
    cost, mask = _seam_cost(arr, band)
    blocked = cost + mask * _MOTIF_PENALTY
    # 切線不能落在第 0 欄，否則 wrap 邊界會原封不動保留舊縫
    blocked[:, 0] = np.inf
    path = _cyclic_path(blocked) if cyclic else _min_cost_path(blocked)
    rows = np.arange(h)
    err = float(np.sqrt(cost[rows, path] / arr.shape[2]).mean())
    cut = float(mask[rows, path].mean())
    return path, err, cut, float(mask.mean())


def _shift_mismatch(arr: np.ndarray, max_band: int) -> np.ndarray:
    """
    每個帶寬（1..max_band）的頭尾平均落差。

    帶寬決定單元寬度 w−band，而單元能無縫拼接，等價於「原稿平移 w−band 後
    與自己重合」。所以帶寬不是可以隨便挑的自由參數：圖案週期的整數倍剛好
    落在 w−band 時，頭尾帶會逐像素吻合，切線無處可切也不必切，接縫與圖案
    韻律同時保住；差幾十個像素，整條帶子就全是對不上的區域，切線無論怎麼
    走都得剖開圖案。

    固定的幾個比例命中這種帶寬純屬碰運氣。與其去估週期（要對付亞像素、多重
    週期、只有單軸有週期），不如直接把每個帶寬的落差掃出來排序。

    沿線方向先做分塊平均：帶寬只影響另一軸的相位，線方向的細節對排序沒有
    貢獻，降下來才掃得動四位數的帶寬範圍。
    """
    h, w = arr.shape[:2]
    blocks = min(_BAND_SCAN_BLOCKS, h)
    edges = np.linspace(0, h, blocks + 1).astype(np.int64)
    sig = np.stack(
        [
            arr[a:b].astype(np.float32).mean(axis=0)
            for a, b in zip(edges[:-1], edges[1:])
        ]
    ).reshape(blocks, w, -1)
    out = np.full(max_band + 1, np.inf, dtype=np.float64)
    for b in range(1, max_band + 1):
        out[b] = float(np.abs(sig[:, :b] - sig[:, w - b :]).mean())
    return out


def _band_candidates(arr: np.ndarray, max_band: int) -> list[int]:
    """值得試的重疊帶寬度：固定比例打底，再加上頭尾本來就吻合的幾個。"""
    w = arr.shape[1]
    lo = min(_BAND_MIN_PX, max_band)
    hi = max(lo, max_band)
    bands = {max(lo, min(int(round(w * f)), hi)) for f in _BAND_FRACTIONS}

    if hi >= lo + 8:
        mism = _shift_mismatch(arr, hi)
        # 谷底附近的帶寬彼此差不多，隔開來取才會拿到不同的谷
        apart = max(8, hi // 20)
        picked: list[int] = []
        for b in np.argsort(mism[: hi + 1]):
            band = int(b)
            if band < lo or any(abs(band - c) <= apart for c in picked):
                continue
            picked.append(band)
            if len(picked) >= _BAND_SCAN_KEEP:
                break
        bands.update(picked)
    return sorted(b for b in bands if b >= 2)


# 一條切線只要穿過圖案就會留下殘肢，代價遠高於多吃掉一點邊長。權重要大到
# 讓「乾淨的寬帶」永遠勝過「有殘肢的窄帶」。
_CUT_PENALTY = 1000.0


def _cut_axis(
    arr: np.ndarray, max_band: int, *, cyclic: bool = False
) -> tuple[np.ndarray, int, np.ndarray, float, float, float, float]:
    """
    挑帶寬並沿垂直方向縫合首尾。

    回傳 (輸出, 帶寬, 切線, 平均色差, 穿過圖案比例, 帶內圖案占比, 取自尾端比例)。
    輸出寬度為 w − 帶寬：尾端該幾欄被吃掉，與首端幾欄依切線合成。

    帶寬與切線一起挑：能讓切線完全繞開圖案的帶寬優先，都繞不開時才比平均
    色差，同分取窄的少犧牲內容。
    """
    w = arr.shape[1]
    max_band = min(max_band, w // 2)
    best: tuple[float, int, np.ndarray, float, float, float] | None = None
    for band in _band_candidates(arr, max_band):
        path, err, cut, dense = _solve_axis(arr, band, cyclic=cyclic)
        # 每多吃 1% 邊長，容許誤差放寬一點點，避免為了 0.1 階去砍掉四分之一畫面
        score = cut * _CUT_PENALTY + err + (band / float(w)) * 3.0
        if best is None or score < best[0]:
            best = (score, band, path, err, cut, dense)
    assert best is not None
    _, band, path, err, cut, dense = best
    dup = float((np.arange(band)[None, :] < path[:, None]).mean())
    return _splice_axis(arr, band, path), band, path, err, cut, dense, dup


@dataclass(frozen=True)
class MincutInfo:
    """切線的完整記錄，足以在另一份像素資料上原樣重放。

    重放能力是 CMYK 保真的關鍵：判斷在 sRGB 做，實際的像素搬移要直接作用
    在原生通道上。CMYK→sRGB→CMYK 來回轉換實測會造成平均 4–8 階的視覺
    色偏，比接縫修復本身大一個數量級，絕不能走那條路。
    """

    band_v: int
    band_h: int
    err_v: float
    err_h: float
    dup_v: float
    dup_h: float
    cut_v: float = 0.0
    cut_h: float = 0.0
    """切線穿過圖案的比例。大於零表示有圖案被切成兩個來源，會留下殘肢。"""

    dense_v: float = 0.0
    dense_h: float = 0.0
    """重疊帶裡屬於圖案的面積比。用來判斷「繞不過去」有多合理。"""

    path_v: np.ndarray | None = None
    path_h: np.ndarray | None = None
    h_first: bool = False
    """為真時先切水平再切垂直；重放必須同一順序，否則內部會再裂開。"""

    @property
    def applied(self) -> bool:
        return self.band_v > 0 or self.band_h > 0

    @property
    def motif_cut(self) -> float:
        return max(self.cut_v, self.cut_h)

    @property
    def motif_dense(self) -> float:
        """切得最兇那一軸的帶內圖案占比。"""
        return self.dense_v if self.cut_v >= self.cut_h else self.dense_h

    def describe(self) -> str:
        parts = []
        if self.band_v:
            parts.append(f"V帶{self.band_v}px誤差{self.err_v:.1f}")
        if self.band_h:
            parts.append(f"H帶{self.band_h}px誤差{self.err_h:.1f}")
        if self.motif_cut > 0:
            parts.append(
                f"切到圖案{self.motif_cut:.1%}（帶內圖案{self.motif_dense:.0%}）"
            )
        return "最小誤差切(" + "，".join(parts) + ")" if parts else "未切"


def replay_mincut(arr: np.ndarray, info: MincutInfo) -> np.ndarray:
    """在另一份像素資料上重放同一組切線。"""
    a = _as_3d(arr)

    def _h(x: np.ndarray) -> np.ndarray:
        t = np.ascontiguousarray(np.swapaxes(x, 0, 1))
        t = _splice_axis(t, info.band_h, info.path_h)
        return np.ascontiguousarray(np.swapaxes(t, 0, 1))

    if info.h_first:
        if info.path_h is not None:
            a = _h(a)
        if info.path_v is not None:
            a = _splice_axis(a, info.band_v, info.path_v)
    else:
        if info.path_v is not None:
            a = _splice_axis(a, info.band_v, info.path_v)
        if info.path_h is not None:
            a = _h(a)
    return a[:, :, 0] if arr.ndim == 2 else a


def wrap_mincut(
    arr: np.ndarray,
    *,
    do_v: bool = True,
    do_h: bool = True,
    max_band_frac: float = 0.25,
    h_first: bool = False,
) -> tuple[np.ndarray, MincutInfo]:
    """
    以最小誤差切線縫合首尾，讓單元真正四方連續。

    `do_v` / `do_h` 由呼叫端依該軸是否本來就連續決定。對已經連續的軸下刀
    只會無中生有一條縫——實測 `1 (4).png` 水平原本是 0，硬切之後變成 9.9。
    """
    a = _as_3d(arr)
    h, w = a.shape[:2]
    band_v = band_h = 0
    err_v = err_h = 0.0
    dup_v = dup_h = 0.0
    cut_v = cut_h = 0.0
    dense_v = dense_h = 0.0
    path_v = path_h = None
    out = a

    def _v(x: np.ndarray, *, cyclic: bool) -> tuple:
        return _cut_axis(
            x, min(int(x.shape[1] * max_band_frac), x.shape[1] // 2), cyclic=cyclic
        )

    def _h(x: np.ndarray, *, cyclic: bool) -> tuple:
        t = np.ascontiguousarray(np.swapaxes(x, 0, 1))
        t, b, p, e, c, d, u = _cut_axis(
            t, min(int(t.shape[1] * max_band_frac), t.shape[1] // 2), cyclic=cyclic
        )
        return np.ascontiguousarray(np.swapaxes(t, 0, 1)), b, p, e, c, d, u

    # 後刀一律環形：按欄位在「取上緣／取下緣」切換時，若首欄與末欄切在不同
    # 列，先縫好的那一軸會在帶子裡被打斷。
    if h_first:
        if do_h and h >= 4 * _BAND_MIN_PX:
            out, band_h, path_h, err_h, cut_h, dense_h, dup_h = _h(out, cyclic=False)
        if do_v and out.shape[1] >= 4 * _BAND_MIN_PX:
            out, band_v, path_v, err_v, cut_v, dense_v, dup_v = _v(out, cyclic=True)
    else:
        if do_v and w >= 4 * _BAND_MIN_PX:
            out, band_v, path_v, err_v, cut_v, dense_v, dup_v = _v(out, cyclic=False)
        if do_h and out.shape[0] >= 4 * _BAND_MIN_PX:
            out, band_h, path_h, err_h, cut_h, dense_h, dup_h = _h(out, cyclic=True)

    if arr.ndim == 2:
        out = out[:, :, 0]
    return out, MincutInfo(
        band_v,
        band_h,
        err_v,
        err_h,
        dup_v,
        dup_h,
        cut_v,
        cut_h,
        dense_v,
        dense_h,
        path_v,
        path_h,
        h_first,
    )


# --------------------------------------------------------------------------
# 2. 梯度域週期化
# --------------------------------------------------------------------------


def _poisson_smooth(tb: np.ndarray, lr: np.ndarray, h: int, w: int) -> np.ndarray:
    """
    由邊界落差解出平滑場 s，使 u - s 完全週期。

    tb: (w, c) 上下邊落差 u[-1] - u[0]；lr: (h, c) 左右邊落差 u[:,-1] - u[:,0]。
    """
    chan = tb.shape[1]
    v = np.zeros((h, w, chan), dtype=np.float32)
    v[0, :] = tb
    v[-1, :] = -tb
    v[:, 0] += lr
    v[:, -1] -= lr

    q = np.arange(h, dtype=np.float32).reshape(-1, 1)
    r = np.arange(w // 2 + 1, dtype=np.float32).reshape(1, -1)
    den = (
        2.0 * np.cos(2.0 * np.pi * q / h)
        + 2.0 * np.cos(2.0 * np.pi * r / w)
        - 4.0
    )
    den[0, 0] = 1.0

    s = np.empty((h, w, chan), dtype=np.float32)
    for c in range(chan):
        fv = np.fft.rfft2(v[:, :, c])
        fv /= den
        fv[0, 0] = 0.0
        s[:, :, c] = np.fft.irfft2(fv, s=(h, w))
    return s


def _resample_profile(p: np.ndarray, n: int) -> np.ndarray:
    """把長度 m 的邊界落差剖面重取樣成長度 n。"""
    m = p.shape[0]
    if m == n:
        return p.astype(np.float32)
    src = np.linspace(0.0, 1.0, m, dtype=np.float64)
    dst = np.linspace(0.0, 1.0, n, dtype=np.float64)
    return np.stack(
        [np.interp(dst, src, p[:, c].astype(np.float64)) for c in range(p.shape[1])],
        axis=1,
    ).astype(np.float32)


def _lowpass_profile(p: np.ndarray, frac: float) -> np.ndarray:
    """
    只保留邊界落差的低頻部分。

    修正場的職責是抹掉整體光照／色溫落差。落差剖面裡的高頻尖刺來自圖案
    本身沒對齊，Poisson 無法真的接上它，只會在附近糊出一塊肉眼可見的
    局部色偏。把剖面低通之後，修正場才是真正意義上的平滑場。
    """
    n = p.shape[0]
    k = max(3, int(round(n * frac)) | 1)
    if k >= n:
        return np.repeat(p.mean(axis=0, keepdims=True), n, axis=0)
    pad = k // 2
    # 環形補邊：剖面本身是繞一圈的，用 wrap 才不會在頭尾生出假的落差
    padded = np.concatenate([p[-pad:], p, p[:pad]], axis=0)
    kernel = cv2.getGaussianKernel(k, k / 5.0).ravel().astype(np.float32)
    out = np.empty_like(p, dtype=np.float32)
    for c in range(p.shape[1]):
        out[:, c] = np.convolve(padded[:, c], kernel, mode="valid")
    return out


@dataclass(frozen=True)
class PeriodizeInfo:
    smooth_peak: float
    """修正場的最大絕對值。越大表示原稿本身首尾差越多。"""

    shift_mean: float
    shift_p99: float
    clipped: float
    """被截斷到 0/255 的像素比例。截斷會破壞週期性，過高就不該採用。"""

    def describe(self) -> str:
        return (
            f"週期化(修正場 {self.smooth_peak:.0f}，"
            f"色偏 {self.shift_mean:.2f}/{self.shift_p99:.0f}"
            f"，截斷 {self.clipped:.2%})"
        )


def periodize(
    arr: np.ndarray,
    *,
    profile_lowpass: float = _PROFILE_LOWPASS_FRAC,
) -> tuple[np.ndarray, PeriodizeInfo]:
    """
    Moisan periodic + smooth 分解，取週期分量。

    u = p + s，其中 s 是解 Poisson 方程得到的平滑場，p 在數學上完全週期。
    細節（高頻）原封不動，所以不會糊、不會鬼影；改變的只有極低頻的明暗。

    大圖改在低解析度解 s 再放大。s 本來就只有低頻，這樣做誤差可以忽略，
    但省下數 GB 記憶體。
    """
    a = _as_3d(arr)
    h, w = a.shape[:2]
    top = a[0].astype(np.float32)
    bottom = a[-1].astype(np.float32)
    left = a[:, 0].astype(np.float32)
    right = a[:, -1].astype(np.float32)
    tb = _lowpass_profile(bottom - top, profile_lowpass)
    lr = _lowpass_profile(right - left, profile_lowpass)

    if h * w <= _POISSON_FULLRES_PIXELS:
        s = _poisson_smooth(tb, lr, h, w)
    else:
        scale = _POISSON_COARSE_SIDE / float(max(h, w))
        nh = max(64, int(round(h * scale)))
        nw = max(64, int(round(w * scale)))
        s_small = _poisson_smooth(
            _resample_profile(tb, nw), _resample_profile(lr, nh), nh, nw
        )
        s = cv2.resize(s_small, (w, h), interpolation=cv2.INTER_CUBIC)
        s = _as_3d(s).astype(np.float32)

    p = a.astype(np.float32) - s
    clipped = float(np.mean((p < -0.5) | (p > 255.5)))
    out = np.clip(np.rint(p), 0, 255).astype(np.uint8)

    d = np.abs(out.astype(np.int16) - a.astype(np.int16))
    info = PeriodizeInfo(
        smooth_peak=float(np.abs(s).max()),
        shift_mean=float(d.mean()),
        shift_p99=float(np.percentile(d, 99)),
        clipped=clipped,
    )
    if arr.ndim == 2:
        out = out[:, :, 0]
    return out, info


# --------------------------------------------------------------------------
# 3. 輔助：判斷某軸是否已經連續
# --------------------------------------------------------------------------


# 粗定位的縮圖邊長。規則圖案在低解析度下到處都長得一樣，320 太粗會讓
# 前幾名全落在週期倍數上而錯過真正的位置。
_RECOVER_COARSE_SIDE = 768
_RECOVER_PROBE = 192
_RECOVER_TOP_K = 12


def torus_crop(arr: np.ndarray, y0: int, x0: int, ch: int, cw: int) -> np.ndarray:
    """
    環面裁切：以 (y0, x0) 為左上角，超出邊界的部分繞回來。

    不用 `np.roll`。滾動會複製整張陣列，在 116MP 的稿件上光是取一塊
    64×64 的探針就要搬 900MB。改用模數索引，成本只跟取出的大小有關。
    """
    h, w = arr.shape[:2]
    y0 %= h
    x0 %= w
    if y0 + ch <= h and x0 + cw <= w:
        return arr[y0 : y0 + ch, x0 : x0 + cw]
    ys = (y0 + np.arange(ch)) % h
    xs = (x0 + np.arange(cw)) % w
    return arr[np.ix_(ys, xs)]


def _gray(arr: np.ndarray) -> np.ndarray:
    return arr.mean(axis=2).astype(np.float32)


def recover_torus_crop(
    src: np.ndarray, out: np.ndarray
) -> tuple[int, int] | None:
    """
    找出 `out` 是 `src` 環面上哪個位置切下來的，找不到回 None。

    週期裁切與點綴晶格產出的都是原稿的環面裁切，但它們只回傳陣列不回傳
    座標。與其去改那兩個上千行的搜尋器，不如在這裡把座標找回來——找到了
    就能把同一刀原封不動下在原生 CMYK 資料上，輸出與原稿逐位元相同。

    兩段定位：先在縮圖上用樣板比對抓大概位置，再在該位置附近取一小塊原
    解析度區域做第二次樣板比對得到精確座標。最後一定做逐位元驗證，驗不
    過就回 None，呼叫端會退回別的做法，絕不會悄悄輸出錯的圖。
    """
    sh, sw = src.shape[:2]
    oh, ow = out.shape[:2]
    if oh > sh or ow > sw or src.shape[2:] != out.shape[2:]:
        return None
    if (oh, ow) == (sh, sw) and np.array_equal(src, out):
        return (0, 0)

    scale = min(1.0, _RECOVER_COARSE_SIDE / float(max(sh, sw)))
    th = max(8, int(round(oh * scale)))
    tw = max(8, int(round(ow * scale)))
    small_src = cv2.resize(
        _gray(src),
        (max(16, int(round(sw * scale))), max(16, int(round(sh * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    small_out = cv2.resize(_gray(out), (tw, th), interpolation=cv2.INTER_AREA)
    canvas = np.tile(small_src, (2, 2))[
        : small_src.shape[0] + th, : small_src.shape[1] + tw
    ]
    if canvas.shape[0] < th or canvas.shape[1] < tw:
        return None
    res = cv2.matchTemplate(canvas, small_out, cv2.TM_SQDIFF)

    # 縮圖上一格對應原圖 1/scale 個像素，再留一格餘裕
    rad = max(4, int(round(2.0 / max(scale, 1e-9))))
    ph = min(oh, _RECOVER_PROBE)
    pw = min(ow, _RECOVER_PROBE)
    probe = _gray(out[:ph, :pw])

    # 試前幾名而非只試第一名：重複圖案在縮圖上有一堆長得一樣的位置，
    # 粗定位很容易挑到「像但不是」的那個，只試一次就會白白放棄。
    flat = res.ravel()
    k = int(min(_RECOVER_TOP_K, flat.size))
    order = np.argpartition(flat, k - 1)[:k]
    for idx in order[np.argsort(flat[order])]:
        my, mx = np.unravel_index(int(idx), res.shape)
        gy = int(round(my / max(scale, 1e-9))) - rad
        gx = int(round(mx / max(scale, 1e-9))) - rad
        region = _gray(torus_crop(src, gy, gx, ph + 2 * rad, pw + 2 * rad))
        fine = cv2.matchTemplate(region, probe, cv2.TM_SQDIFF)
        fy, fx = np.unravel_index(int(np.argmin(fine)), fine.shape)
        y0 = (gy + int(fy)) % sh
        x0 = (gx + int(fx)) % sw
        if np.array_equal(torus_crop(src, y0, x0, oh, ow), out):
            return (y0, x0)
    return None


def axis_seam_excess(arr: np.ndarray) -> tuple[float, float]:
    """
    回傳 (垂直軸, 水平軸) 的 wrap 超出量，用來決定哪一軸需要下刀。

    超出量 = 對邊差 − 該圖內部相鄰線差的中位數。用中位數當基準，條紋這種
    天生線差大的圖才不會被誤判成有縫。
    """
    from app.quality import seam_report

    rep = seam_report(_as_3d(arr))
    return rep.excess_v, rep.excess_h
