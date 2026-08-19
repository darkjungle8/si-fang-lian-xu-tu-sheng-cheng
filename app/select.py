"""候選產生、無縫閘門、保真度擇優。

## 為什麼要換掉舊的決策方式

舊流程是「試各種變換 → 量一下 → 覺得不夠好就退回原圖」。每次有圖出包
就加一條禁令（禁半幅、禁錯切、禁 soft、密花禁這個、點綴禁那個），禁到
最後所有變換都被擋住，退路變成什麼都不做。實測 52 個案例裡 45 個輸出
與原圖完全相同、34 個仍留著肉眼可見的縫，其中三張的對邊色差高達 85、
82、40。而回歸測試的斷言是「不得比原稿更差」，原圖直出永遠通過，所以
它還一路顯示綠燈。

## 現在的契約

反過來：**只有能證明自己無縫的候選才有資格出線**。

1. 產生基底候選（原圖、週期裁切、點綴晶格、清邊補花）。這些都是無損的
   裁切／滾動，是保真度最高的來源。
2. 每個基底若還有縫，就用 `seamless_core` 的最小誤差切與梯度域週期化
   把它變成真的無縫。這兩個算子由構造保證結果可拼接，不是碰運氣。
3. 閘門：接縫超出量、內部有無新斷裂、幾何有無被扳斜、色調有無跑掉。
4. 通過閘門的候選才比成本，成本以「平鋪回去還原不還原得了原設計」為主。

好處是不必再為圖種寫禁令。半幅滾動之所以要禁，是因為它只把縫搬到中央；
現在 `internal_excess` 直接量得到那條搬過去的縫，它自然過不了閘門。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from app.quality import (
    SeamReport,
    axis_line_energy,
    color_shift,
    design_error,
    edge_void_ratio,
    ink_frac,
    seam_report,
    tone_shift,
    wrap_hotspot,
)
from app.seamless_core import (
    periodize,
    replay_mincut,
    torus_crop,
    wrap_mincut,
)

# 接縫超出量的硬門檻，也就是「這條縫絕對不能接受」的界線。
#
# 用整批稿件實際比對過：超出量 8.1 那張藍底能看到一條垂直邊、雛菊被切成
# 兩半，12.5 的格紋明顯錯位。5 以下多數看不出來，但平坦大色塊例外——某張
# 波點稿超出量只有 4.0，1:1 檢視仍看得到一顆白點在接縫處被切出台階。
#
# 所以門檻只負責擋掉一定不行的，灰帶交給成本函數權衡。
SEAM_OK = 5.0

# 原稿好到這個程度就直接採用，不再搜尋。任何加工都只會降低保真度，而且
# 這批稿件有 129/199 落在這裡，省下的週期搜尋佔了大半執行時間。
SEAM_PERFECT = 2.0
# 判斷某一軸要不要下刀。比 SEAM_OK 鬆，寧可多切也不要留縫。
AXIS_CUT_MIN = 1.0
# 內部斷裂容許量：與原稿比。條紋壁紙的硬邊原稿就有，不算我們造成的。
INTERNAL_FLOOR = 6.0
INTERNAL_SLACK = 1.15
INTERNAL_MARGIN = 2.0
TONE_MAX = 4.0
# 逐像素平均色偏。印刷實務上 2 階以內看不出來。
COLOR_MEAN_MAX = 2.0
# 低頻色塊：整片偏亮／偏暗，人眼對這種大面積漸變最敏感。
COLOR_LOW_MAX = 12.0
# 被壓到 0/255 的比例。截斷會把層次壓成死白死黑，還會破壞週期性。
CLIP_MAX = 0.02
# 平鋪回去與原設計的差距，只用來把關裁切類候選。
#
# 這個量對「裁錯週期」很靈敏：抓到真週期的裁切平鋪回去幾乎完全還原，
# 假週期則對不上。但它不適合拿來評判最小誤差切——切掉一條帶子並不改變
# 任何局部結構，只是讓整體相位對不回原圖，數值自然就高。實測兩張目視
# 完全無縫、圖案完整的稿件（散點馴鹿、漿果碎花）分別是 54 與 45，用同
# 一把尺就會把正確結果判成失敗。
DESIGN_MAX_CROP = 35.0
# 非裁切類只擋災難級的破壞，細緻的判斷交給接縫與色偏門檻。
DESIGN_MAX = 90.0
# 切線穿過圖案的容許比例。切線每列只移動一格，穿過 1% 的長度就足以把一
# 個圖案剖半——實測那張條紋大象只有 8% 的切線壓在象身上，成品卻是一隻
# 沒有身體的象頭。留一點餘裕給抗鋸齒與 JPEG 雜訊誤標的零星像素。
# 切線穿過圖案的容許比例。1% 就把可辨認圖章剖半；給抗鋸齒／JPEG 一點餘裕。
MOTIF_CUT_MAX = 0.012
# 密花圖放寬：容許量隨重疊帶裡的圖案占比成長。
#
# 殘肢要醒目，得先有素底讓人看出「這東西缺了一半」。滿版碎花整條帶子都是
# 花，切線根本無路可繞，切到也認不出是哪一朵被剖開——實測兩張密花稿切到
# 2.2%／2.3%，1:1 檢視完全看不出來；而稀疏的條紋大象只切 8% 就是一隻沒有
# 身體的象頭。用帶內圖案占比當尺，正好分開這兩種情況。
MOTIF_CUT_DENSE = 0.12
# 帶內圖案占比的放寬只適用於整張也是密花的情況。白底漿果帶內可以 62% 都
# 是花，但畫面仍是稀疏圖章——剖開一顆就看得見。整張墨量不夠時維持 1%。
MOTIF_CUT_DENSE_INK = 0.40

# wrap 平均為 0 時，局部 90 分位仍可能很高（圖章被剖、幾何條帶錯相位）。
# 校準：已接上的週期裁切鴨子約 12，狐狸／部落紋／玫瑰 22–55。
HOTSPOT_OK = 14.0
CUT_ERR_OK = 18.0
# 邊緣相對內部少掉的圖案比例。清邊補花掏成十字空洞時約 0.28–0.33；
# 真無縫單元即使邊緣略疏也通常 < 0.22。還要跟原稿比，避免誤殺本身就疏邊的圖。
VOID_MAX = 0.22
VOID_DELTA = 0.12

# 這裡沒有幾何門檻是刻意的。它原本是為了擋錯切對齊把拼布格扳斜，但錯切
# 機制已經整個移除，現存的算子全是像素搬移或逐通道平移，不可能扳斜。留著
# 只會冤枉正確的裁切——條紋圖裁掉一段，軸向能量比自然變化，`1 (84).jpg`
# 的最佳候選就是這樣被判成「扳斜 0.87」而出局的。設計有沒有被改壞，改由
# `design_error`（平鋪回去比對原圖）判定，那才是直接的量。

LogFn = Callable[[str], None] | None


@dataclass
class Candidate:
    arr: np.ndarray
    label: str
    lossless: bool
    """像素值是否完全來自原稿（裁切、滾動、最小誤差切都算）。"""

    recipe: list[tuple[str, object]] | None = None
    """
    重現此候選所需的操作序列，用來在原生色彩通道上重放。

    None 表示無法重放（例如清邊補花是在 sRGB 空間逐像素改寫的），
    此時只能退回 ICC 轉換，保真度較差，成本會因此被加重。
    """

    color_mean: float = 0.0
    """相對其基底的平均色偏。只有梯度域週期化會讓這個值不為零。"""

    color_low: float = 0.0
    clipped: float = 0.0
    """被壓到 0/255 的像素比例。截斷會把層次壓成死白／死黑。"""

    dup: float = 0.0
    """最小誤差切造成的內容重複比例。"""

    motif_cut: float = 0.0
    """
    最小誤差切線穿過圖案的比例。

    切線穿過一個只存在於單側的圖案時，該圖案會一半取自頭端、一半取自尾端，
    留下半隻大象、孤立的長頸鹿犄角這種殘肢。殘肢兩側在原稿裡本來就相鄰，
    對邊色差是零，`seam_report` 看不到它——這是唯一能抓到的量。
    """

    motif_dense: float = 0.0
    """重疊帶裡屬於圖案的面積比。密花圖切線無路可繞，判定時據此放寬。"""

    cut_err: float = 0.0
    """最小誤差切路徑的平均色差。wrap 被構造保證為 0 之後，這仍能分辨錯相位。"""

    hotspot: float = 0.0
    """wrap 線局部熱點，見 `wrap_hotspot`。"""

    derr: float = 0.0

    rep: SeamReport | None = None
    errors: list[str] = field(default_factory=list)
    cost: float = 0.0

    def describe(self) -> str:
        return self.label


def apply_recipe(arr: np.ndarray, recipe: list[tuple[str, object]]) -> np.ndarray:
    """在另一份像素資料上重放候選的操作序列。"""
    out = arr
    for kind, param in recipe:
        if kind == "crop":
            y0, x0, ch, cw = param  # type: ignore[misc]
            out = torus_crop(out, y0, x0, ch, cw)
        elif kind == "inset":
            y0, x0, ch, cw = param  # type: ignore[misc]
            out = out[y0 : y0 + ch, x0 : x0 + cw]
        elif kind == "mincut":
            out = replay_mincut(out, param)  # type: ignore[arg-type]
        elif kind == "roll":
            oy, ox = param  # type: ignore[misc]
            out = np.roll(np.roll(out, -int(oy), 0), -int(ox), 1)
        elif kind == "periodize":
            out, _ = periodize(out)
        else:
            raise ValueError(f"未知的操作：{kind}")
    return out


@dataclass
class SourceFacts:
    """對原稿量一次就好的東西，後面所有候選共用。"""

    arr: np.ndarray
    rep: SeamReport
    axis_energy: float
    needs_native: bool = False
    """
    原稿的色彩空間是否非 sRGB 所能無損表達（印刷 CMYK 就是）。

    為真時，無法重放到原生通道的候選一律出局。那條路要走
    CMYK→sRGB→CMYK 來回轉換，實測視覺色偏平均 4–8 階、最大 37 階，
    比接縫修復本身大一個數量級——用它換無縫等於拆東牆補西牆。
    """
    ink: float = 0.0
    edge_void: float = 0.0

    @property
    def internal_allow(self) -> float:
        return (
            max(self.rep.internal_excess, INTERNAL_FLOOR) * INTERNAL_SLACK
            + INTERNAL_MARGIN
        )


def measure(src: SourceFacts, cand: Candidate) -> Candidate:
    """填上候選的接縫體檢、閘門違規與成本。"""
    rep = seam_report(cand.arr)
    cand.rep = rep
    derr = design_error(src.arr, cand.arr)
    tone = tone_shift(src.arr, cand.arr)
    errs: list[str] = []

    if rep.wrap_excess > SEAM_OK:
        errs.append(f"接縫未消:{rep.wrap_excess:.1f}")
    if rep.internal_excess > src.internal_allow:
        errs.append(f"內部新增斷裂:{rep.internal_excess:.1f}")

    # 色偏三道：逐像素平均、低頻色塊、截斷。
    # `tone_shift` 只比通道均值，擋不住「一半變亮一半變暗」這種抵銷掉的
    # 大偏移——某張圖的梯度域週期化色偏平均高達 18/255、截斷 19%，通道均
    # 值卻只動了 3.4，就這樣混過去成為當選者。
    if cand.color_mean > COLOR_MEAN_MAX:
        errs.append(f"色偏過大:{cand.color_mean:.1f}")
    if cand.color_low > COLOR_LOW_MAX:
        errs.append(f"低頻色塊:{cand.color_low:.0f}")
    if cand.clipped > CLIP_MAX:
        errs.append(f"截斷過多:{cand.clipped:.0%}")
    if tone > TONE_MAX and not (not cand.lossless and cand.recipe is None):
        errs.append(f"色調偏移:{tone:.1f}")
    has_crop = bool(cand.recipe) and any(k == "crop" for k, _ in cand.recipe)
    if derr > (DESIGN_MAX_CROP if has_crop else DESIGN_MAX):
        errs.append(f"設計被改壞:{derr:.0f}")
    if src.ink >= MOTIF_CUT_DENSE_INK:
        motif_allow = max(MOTIF_CUT_MAX, cand.motif_dense * MOTIF_CUT_DENSE)
    else:
        motif_allow = MOTIF_CUT_MAX
    if cand.motif_cut > motif_allow:
        errs.append(f"切線剖開圖案:{cand.motif_cut:.1%}")
    hot = wrap_hotspot(cand.arr)
    cand.hotspot = hot
    cand.derr = derr
    # 真週期裁切平鋪回去對得上時，wrap 線本來就是圖案自身的邊緣分佈，
    # 90 分位熱點會偏高，不能當成結構縫。
    if (not has_crop) and hot > HOTSPOT_OK:
        errs.append(f"結構接縫:{hot:.0f}")
    if cand.cut_err > CUT_ERR_OK:
        errs.append(f"切線色差:{cand.cut_err:.0f}")
    void = edge_void_ratio(cand.arr)
    if void > VOID_MAX and void > src.edge_void + VOID_DELTA:
        errs.append(f"邊緣掏空:{void:.0%}")
    if src.needs_native and cand.recipe is None:
        errs.append("無法保色")

    cand.errors = errs
    cand.cost = (
        derr * 4.0
        + cand.color_mean * 8.0
        + cand.color_low * 0.6
        + cand.dup * 20.0
        # 殘肢比殘縫更醒目：縫是一條線，殘肢是一個認得出來、卻缺了一半的
        # 圖案。權重要壓得過「還原度」，否則寧可留著殘肢也不肯換一刀。
        + cand.motif_cut * 400.0
        + cand.cut_err * 4.0
        + hot * hot * 0.12
        # 接縫代價超線性：殘縫是這個工具唯一不能妥協的東西，愈接近門檻
        # 就愈值得付代價去修。線性權重會讓「超出 4.0 的波點稿」寧可留著
        # 那顆被切台階的白點，也不肯接受一次乾淨的週期裁切。
        + rep.wrap_excess**2 * 3.0
        + max(0.0, rep.internal_excess - src.rep.internal_excess) * 1.0
        + (0.0 if cand.lossless else 3.0)
        # 無法重放到原生通道就得走 ICC 來回轉換，實測平均 4–8 階視覺色偏
        + (0.0 if cand.recipe is not None else 30.0)
    )
    cand.label = f"{cand.label}｜還原 {derr:.1f} 縫 {rep.wrap_excess:.1f}"
    return cand


def make_seamless_variants(
    base: np.ndarray,
    label: str,
    *,
    lossless: bool = True,
    recipe: list[tuple[str, object]] | None = None,
) -> list[Candidate]:
    """
    把一個基底候選加工成真的無縫，回傳幾種加工強度供比較。

    順序很要緊。單獨用梯度域週期化去修一條 80 階的大縫，修正場本身就會
    變成一大片色偏（實測平均 18/255）；先用最小誤差切把結構對上，殘差
    小了再週期化，色偏平均只剩 0.2。所以「先切再週期化」是主力，
    「只週期化」留給那種縫其實只是整體光照落差的圖。
    """

    def _r(*extra: tuple[str, object]) -> list[tuple[str, object]] | None:
        return None if recipe is None else [*recipe, *extra]

    out: list[Candidate] = [Candidate(base, label, lossless, recipe)]
    rep = seam_report(base)
    # 用嚴格門檻決定要不要展開加工版本，寬鬆門檻只用來判定「絕對不行」。
    # 灰帶案例也要把完整選單擺出來，才輪得到成本函數權衡。
    if rep.wrap_excess <= SEAM_PERFECT:
        return out

    do_v = rep.excess_v > AXIS_CUT_MIN
    do_h = rep.excess_h > AXIS_CUT_MIN
    if do_v or do_h:
        cut, mi = wrap_mincut(base, do_v=do_v, do_h=do_h)
        dup = mi.dup_v * mi.band_v / max(base.shape[1], 1) + (
            mi.dup_h * mi.band_h / max(base.shape[0], 1)
        )
        out.append(
            Candidate(
                cut,
                f"{label}＋{mi.describe()}",
                lossless,
                _r(("mincut", mi)),
                dup=dup,
                motif_cut=mi.motif_cut,
                motif_dense=mi.motif_dense,
                cut_err=max(mi.err_v, mi.err_h),
            )
        )
        cut_rep = seam_report(cut)
        if cut_rep.wrap_excess > 0.5:
            per, pi = periodize(cut)
            out.append(
                Candidate(
                    per,
                    f"{label}＋{mi.describe()}＋{pi.describe()}",
                    False,
                    _r(("mincut", mi), ("periodize", None)),
                    color_mean=pi.shift_mean,
                    color_low=color_shift(cut, per).lowfreq,
                    clipped=pi.clipped,
                    dup=dup,
                    motif_cut=mi.motif_cut,
                    motif_dense=mi.motif_dense,
                    cut_err=max(mi.err_v, mi.err_h),
                )
            )

        def _add_mincut(cut_arr: np.ndarray, info, tag: str) -> None:
            dup_i = info.dup_v * info.band_v / max(base.shape[1], 1) + (
                info.dup_h * info.band_h / max(base.shape[0], 1)
            )
            out.append(
                Candidate(
                    cut_arr,
                    f"{label}＋{tag}{info.describe()}",
                    lossless,
                    _r(("mincut", info)),
                    dup=dup_i,
                    motif_cut=info.motif_cut,
                    motif_dense=info.motif_dense,
                    cut_err=max(info.err_v, info.err_h),
                )
            )
            if seam_report(cut_arr).wrap_excess > 0.5:
                peri, pinfo = periodize(cut_arr)
                out.append(
                    Candidate(
                        peri,
                        f"{label}＋{tag}{info.describe()}＋{pinfo.describe()}",
                        False,
                        _r(("mincut", info), ("periodize", None)),
                        color_mean=pinfo.shift_mean,
                        color_low=color_shift(cut_arr, peri).lowfreq,
                        clipped=pinfo.clipped,
                        dup=dup_i,
                        motif_cut=info.motif_cut,
                        motif_dense=info.motif_dense,
                        cut_err=max(info.err_v, info.err_h),
                    )
                )

        # 第一刀仍剖開圖案、結構熱點高、或切線色差大時，加寬重疊帶再切。
        # wrap 平均可以是 0（地色接地上），所以不能只看 wrap_excess／殘肢比例。
        # 不放寬閘門，只是多給構造上仍無縫的候選。
        allow = max(MOTIF_CUT_MAX, mi.motif_dense * MOTIF_CUT_DENSE)
        internal_allow = (
            max(rep.internal_excess, INTERNAL_FLOOR) * INTERNAL_SLACK
            + INTERNAL_MARGIN
        )
        internal_worse = cut_rep.internal_excess > internal_allow
        cut_hot = wrap_hotspot(cut)
        struct_fail = (
            mi.motif_cut > MOTIF_CUT_MAX
            or max(mi.err_v, mi.err_h) > CUT_ERR_OK
            or cut_hot > HOTSPOT_OK
        )
        if (
            mi.motif_cut > allow
            or cut_rep.wrap_excess > SEAM_OK
            or internal_worse
            or struct_fail
        ):
            for frac, tag in ((0.42, "寬帶"), (0.50, "更寬帶")):
                cut_w, mi_w = wrap_mincut(
                    base, do_v=do_v, do_h=do_h, max_band_frac=frac
                )
                _add_mincut(cut_w, mi_w, tag)

        if do_v and do_h and (
            mi.motif_cut > allow or internal_worse or struct_fail
        ):
            for dv, dh, tag in ((True, False, "只V"), (False, True, "只H")):
                for frac, ftag in (
                    (0.25, tag),
                    (0.42, f"{tag}寬帶"),
                    (0.50, f"{tag}更寬帶"),
                ):
                    c1, m1 = wrap_mincut(
                        base, do_v=dv, do_h=dh, max_band_frac=frac
                    )
                    _add_mincut(c1, m1, ftag)

        if do_v and do_h and (internal_worse or struct_fail):
            for frac, tag in ((0.25, "先H"), (0.42, "先H寬帶"), (0.50, "先H更寬帶")):
                ch, mh = wrap_mincut(
                    base,
                    do_v=True,
                    do_h=True,
                    max_band_frac=frac,
                    h_first=True,
                )
                _add_mincut(ch, mh, tag)

        # 把縫移到較空的相位再切：稀疏圖章被釘在預設邊緣時，平移後切線才繞得開。
        if struct_fail and recipe is not None:
            h0, w0 = base.shape[:2]
            for oy, ox, tag in (
                (0, w0 // 4, "平移H"),
                (h0 // 4, 0, "平移V"),
            ):
                rolled = np.roll(np.roll(base, -oy, 0), -ox, 1)
                cut_r, mi_r = wrap_mincut(
                    rolled, do_v=do_v, do_h=do_h, max_band_frac=0.42
                )
                dup_r = mi_r.dup_v * mi_r.band_v / max(rolled.shape[1], 1) + (
                    mi_r.dup_h * mi_r.band_h / max(rolled.shape[0], 1)
                )
                out.append(
                    Candidate(
                        cut_r,
                        f"{label}＋{tag}＋{mi_r.describe()}",
                        lossless,
                        _r(("roll", (oy, ox)), ("mincut", mi_r)),
                        dup=dup_r,
                        motif_cut=mi_r.motif_cut,
                        motif_dense=mi_r.motif_dense,
                        cut_err=max(mi_r.err_v, mi_r.err_h),
                    )
                )

    # 縫純粹來自整體光照／色溫落差時，不必動結構
    per0, pi0 = periodize(base)
    out.append(
        Candidate(
            per0,
            f"{label}＋{pi0.describe()}",
            False,
            _r(("periodize", None)),
            color_mean=pi0.shift_mean,
            color_low=color_shift(base, per0).lowfreq,
            clipped=pi0.clipped,
        )
    )
    return out


@dataclass
class Base:
    """一個基底候選：保真度最高的無損來源。"""

    arr: np.ndarray
    label: str
    lossless: bool = True
    recipe: list[tuple[str, object]] | None = None


def choose(
    src: SourceFacts,
    bases: list[Base],
    log: LogFn = None,
) -> Candidate:
    """
    對每個基底展開無縫變體，過閘門後取成本最低者。

    閘門與成本都是圖種無關的量（接縫超出、殘肢、色偏、還原），不對檔名
    寫特例；新稿件走同一條路。全部過不了閘門仍標「未達標」，不當沉默退回原圖。

    全部都過不了閘門時，退而求其次取「接縫超出量最小」的那個，並在說明
    字串標上「未達標」，讓掃描報告能把它撈出來——沉默地退回原圖正是舊
    流程的病灶。
    """

    def _lg(msg: str) -> None:
        if log is not None:
            log(msg)

    cands: list[Candidate] = []
    for base in bases:
        for c in make_seamless_variants(
            base.arr, base.label, lossless=base.lossless, recipe=base.recipe
        ):
            cands.append(measure(src, c))

    for c in cands:
        _lg(
            f"     候選 成本 {c.cost:7.1f} "
            f"{c.arr.shape[1]}×{c.arr.shape[0]} "
            f"{'／'.join(c.errors) if c.errors else 'OK':22s} {c.label}"
        )

    ok = [c for c in cands if not c.errors]
    if ok:
        # 接縫已經壓到看不見時，不准再讓「還原 0 的原圖」憑成本贏回去。
        # 灰帶（超出 2～5）原圖直出的 design_error 永遠是 0，最小誤差切
        # 切掉帶子後還原分數天生偏高，成本一比就退回原圖，1:1 卻仍切到輪廓。
        tight = [
            c
            for c in ok
            if c.rep is not None and c.rep.wrap_excess <= SEAM_PERFECT
        ]
        if tight:
            best = min(tight, key=lambda c: c.cost)
            _lg(
                f"  → 採用（{len(ok)}/{len(cands)} 過關，"
                f"{len(tight)} 個接縫≤{SEAM_PERFECT:.0f}，"
                f"不讓灰帶靠還原贏）：{best.label}"
            )
            return best
        best = min(
            ok,
            key=lambda c: (
                c.rep.wrap_excess if c.rep is not None else 1e9,
                c.cost,
            ),
        )
        _lg(
            f"  → 採用（{len(ok)}/{len(cands)} 個候選過關，"
            f"全在灰帶取接縫最低）：{best.label}"
        )
        return best

    def _fail_key(c: Candidate) -> tuple[float, float]:
        wrap = c.rep.wrap_excess if c.rep else 1e9
        internal_pen = 0.0
        if c.rep is not None:
            internal_pen = max(0.0, c.rep.internal_excess - src.internal_allow)
        if src.ink >= MOTIF_CUT_DENSE_INK:
            motif_allow = max(MOTIF_CUT_MAX, c.motif_dense * MOTIF_CUT_DENSE)
        else:
            motif_allow = MOTIF_CUT_MAX
        motif_pen = 400.0 * c.motif_cut if c.motif_cut > motif_allow else 0.0
        color_pen = 0.0
        if c.color_mean > COLOR_MEAN_MAX:
            color_pen += 80.0
        if c.color_low > COLOR_LOW_MAX:
            color_pen += 80.0
        if c.clipped > CLIP_MAX:
            color_pen += 80.0
        derr_pen = 80.0 if c.derr > DESIGN_MAX else 0.0
        return (
            wrap
            + c.hotspot * 0.5
            + c.cut_err * 0.35
            + internal_pen
            + motif_pen
            + color_pen
            + derr_pen,
            c.cost,
        )

    best = min(cands, key=_fail_key)
    reasons = "／".join(best.errors)
    _lg(f"  → 全部未達標，取最接近者：{best.label}（{reasons}）")
    best.label = f"未達標［{reasons}］{best.label}"
    return best


def source_facts(arr: np.ndarray, *, needs_native: bool = False) -> SourceFacts:
    return SourceFacts(
        arr=arr,
        rep=seam_report(arr),
        axis_energy=axis_line_energy(arr),
        needs_native=needs_native,
        ink=ink_frac(arr),
        edge_void=edge_void_ratio(arr),
    )


def timed(label: str, fn, log: LogFn = None):
    """跑一個候選產生器並記時；它自己爆掉不該拖垮整批。"""
    t0 = time.perf_counter()
    try:
        result = fn()
    except Exception as exc:  # noqa: BLE001 — 候選失敗只是少一個選項
        if log is not None:
            log(f"  → {label} 失敗（{exc}），跳過")
        return None
    if log is not None:
        log(f"  → {label} 完成（{time.perf_counter() - t0:.1f}s）")
    return result
