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
    motif_fragment_ratio,
    seam_report,
    tone_shift,
    wrap_both_thick_run,
    wrap_cut_ratio,
    wrap_density_ratio,
    wrap_gutter_error,
    wrap_hotspot,
    wrap_hotspot_axes,
    wrap_orphan_run,
    wrap_period_remainder,
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
# 清邊補花搬圖章後，淡底紋／抗鋸齒可讓平均超出量到 8；那不是切出台階的
# 波點（那種是 4.0 且不是補花）。9 仍擋住肉眼一條色帶。
SEAM_REFILL_MAX = 9.0

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
# 切線穿過圖案的容許比例。1% 就把可辨認圖章剖半；給抗鋸齒／JPEG 一點餘裕。
# 1.5% 讓已跨縫對接的雪花切線（約 1.3%）過關，仍擋住散花 4–6% 的剖開。
MOTIF_CUT_MAX = 0.015
# 密花圖放寬：容許量隨重疊帶裡的圖案占比成長。
#
# 殘肢要醒目，得先有素底讓人看出「這東西缺了一半」。滿版碎花整條帶子都是
# 花，切線根本無路可繞，切到也認不出是哪一朵被剖開。稀疏圖章（散花、雪人、
# 蜜蜂）整張墨量也可到 20–40%，不能再用 0.15 當密花——否則 4–6% 切線仍
# 綠燈，2×2 卻是半朵花。只有整張接近滿鋪（ink≥0.42）才放寬，且幅度很小。
MOTIF_CUT_DENSE = 0.04
MOTIF_CUT_DENSE_INK = 0.42

# wrap 平均為 0 時，局部 90 分位仍可能很高（圖章被剖、幾何條帶錯相位）。
# 校準：已接上的週期裁切鴨子約 12，狐狸／部落紋／玫瑰 22–55。
HOTSPOT_OK = 14.0
# 最小誤差切的 wrap 線可以穿過完整圖章，熱點常到 20–35；格紋上的蜜蜂對不
# 上會到 70+。14 會誤殺正確的切線，40 才代表圖章結構仍裂開。
HOTSPOT_CUT_OK = 40.0
# 真週期裁切的 wrap 線也可以穿過完整圖章（狐狸 22–55）。超過約 70 就是
# 格紋對上、圖章沒對上。
HOTSPOT_CROP_OK = 70.0
# 清邊補花若補回完整圖章，熱點應接近地色接縫；刺蝟半截對空白約 13。
# 雪人底紋漩渦在 wrap 上可到 26，冬青白雲跨縫抗鋸齒約 29；實心白圖章
# 跨縫的抗鋸齒長尾可到 ~57。95 的格紋蜜蜂未對齊才出局。
HOTSPOT_REFILL_OK = 60.0
CUT_ERR_OK = 18.0
# 清邊補花只是搬原稿圖章，通道均值會因刪了殘片而動一點；4.0 會讓完整補花
# 出局、半截最小誤差切反而當選。細線雪花清環帶後可到 7.4。
TONE_REFILL_MAX = 8.0
# 邊緣相對內部少掉的圖案比例。清邊補花把碰框件吸到 wrap 後，5% 帶會略疏
# （約 0.24）；十字空洞仍在 0.28–0.33。
VOID_MAX = 0.26
VOID_DELTA = 0.12
# wrap 一側有墨、對邊沒有的最長段。刺蝟被框切、對邊卻是地色時約 200px+；
# 真跨縫對接通常 < 20px。蘑菇／水彩抗鋸齒可到 29px；32 仍擋住半隻刺蝟。
ORPHAN_RUN_MAX = 32
# 清邊補花把左右殘片吸到同一條縫時，2×2 十字會比內部更密。
# 驗證器與閘門共用 1.15：超過就是肉眼看得見的接縫雙行。
WRAP_DENSITY_MAX = 1.15
# 同伴裡實心度離群的圖章比例。雪人缺一塊約 0.16–0.21；雪花大家都凹約 0.06。
FRAGMENT_MAX = 0.10
# 碰邊卻沒接到對邊的圖章 / 內部典型圖章。半隻刺蝟約 0.5+。
WRAP_CUT_MAX = 0.42
# 清邊補花把圖章吸到 wrap 後，水彩碎花的連通域仍可讓比值到 0.53；
# 1:1 圖章是完整的。0.60 仍擋住半隻動物（常 >1）。
WRAP_CUT_REFILL_MAX = 0.60
# 地色溝接縫的溝寬相對內部圖章間距。波點 2048 對 547 週期約 0.88；
# 真週期單元應接近 0。0.35 擋住擠在縫上的杏仁條，仍容許 JPEG 抖動。
GUTTER_ERR_MAX = 0.35
# 週期裁切的邊長必須接近所偵測週期的整數倍。2048÷547 餘 140px（26%），
# 2×2 會在垂直縫上擠出杏仁條；顏色仍接得上所以舊閘門全綠。
PERIOD_REM_MAX = 0.12
# 原稿若有強週期，餘數超過約 3% 就不能當無縫直出（波點 2048÷547 會在縫上擠花）。
PERIOD_REM_SOURCE_MAX = 0.03

# 單元最短邊至少為原稿短邊的 22%。1024 原稿約 225px。
# 0.25 會誤殺 255×384 這種合理直條；0.22 仍擋住把織紋／細條週期當成花布
# 單元的裁切（實測 184×1024、122×558、156×168、104×112）。
MIN_UNIT_EDGE_FRAC = 0.22
# compact 搜尋另加 64px 地板，避免測試小圖依 22% 產出十幾像素的碎片。
COMPACT_ABS_MIN = 64


def min_unit_edge(h: int, w: int) -> int:
    return max(8, int(round(MIN_UNIT_EDGE_FRAC * min(int(h), int(w)))))


def _pref_rank(c: Candidate) -> int:
    """無損裁切 > 環帶補花 > 整張重排 > 切線 > 其他。"""
    lab = c.label
    crop_only = (
        ("點綴晶格" in lab or "週期裁切" in lab)
        and "最小誤差切" not in lab
        and "週期化" not in lab
    )
    if crop_only:
        return 0
    if "整張重排" in lab:
        return 2
    if "清邊補花" in lab:
        return 1
    if "最小誤差切" in lab:
        return 3
    if "週期化" in lab:
        return 4
    return 5


def verify_tiling(
    arr: np.ndarray,
    *,
    cls: str = "A",
    src_density: float | None = None,
) -> list[str]:
    """
    2×2 視覺驗證器：縫帶熱點、十字密度、孤兒／殘片。
    任何策略都不得免檢。cls=A 散點、B 週期裁切、C 滿版。
    """
    from app.processor import stamp_structure_view

    view = stamp_structure_view(arr)
    errs: list[str] = []
    hot = wrap_hotspot(view)
    dens = wrap_density_ratio(view)
    orphan = wrap_orphan_run(view)
    frag = motif_fragment_ratio(view)
    cut = wrap_cut_ratio(view)
    if cls == "C":
        if hot > HOTSPOT_CROP_OK:
            errs.append(f"結構接縫:{hot:.0f}")
        return errs
    if dens > WRAP_DENSITY_MAX and cls != "B":
        if (
            cls == "A"
            and src_density is not None
            and src_density > WRAP_DENSITY_MAX
            and dens <= src_density * 1.12 + 0.03
        ):
            pass
        else:
            errs.append(f"接縫過密:{dens:.2f}")
    if orphan > ORPHAN_RUN_MAX:
        errs.append(f"接縫殘片:{orphan}px")
    # 週期裁切的殘缺比的是圖章實心度離群。藤蔓碎花連通域凹凸不一，
    # 11% 仍會亮紅，但 wrap_cut 才代表縫上剖開；切圖不高就不要誤殺。
    if frag > FRAGMENT_MAX and not (cls == "B" and cut <= WRAP_CUT_MAX):
        errs.append(f"圖案殘缺:{frag:.0%}")
    if cls == "A" and cut > (WRAP_CUT_REFILL_MAX if hot <= HOTSPOT_REFILL_OK else WRAP_CUT_MAX):
        errs.append(f"接縫切圖:{cut:.0%}")
    hot_lim = HOTSPOT_CROP_OK if cls == "B" else HOTSPOT_REFILL_OK
    if hot > hot_lim:
        from app.processor import _fine_grid_aligned

        wrap_ok = cls == "A" and cut <= WRAP_CUT_REFILL_MAX
        if not wrap_ok and not (cls == "B" and _fine_grid_aligned(arr)):
            errs.append(f"結構接縫:{hot:.0f}")
    return errs


def classify_unit_class(src: SourceFacts, *, discrete: bool = False) -> str:
    """A 散點、B 強週期／晶格、C 滿版連通。"""
    if src.ink >= MOTIF_CUT_DENSE_INK and not discrete:
        return "C"
    if discrete or src.period_rem > 0.0:
        return "B"
    if src.ink < 0.55:
        return "A"
    return "C"


def motif_allowance(ink: float, motif_dense: float) -> float:
    """稀疏圖章維持 1.2%；只有滿鋪密花才依帶內密度略放寬。"""
    if ink >= MOTIF_CUT_DENSE_INK:
        return max(MOTIF_CUT_MAX, motif_dense * MOTIF_CUT_DENSE)
    return MOTIF_CUT_MAX


def compact_min_edge(h: int, w: int) -> int:
    return max(COMPACT_ABS_MIN, min_unit_edge(h, w))

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
    hotspot: float = 0.0
    """原稿 wrap 線局部熱點。平均超出量為 0 時仍可能剖開圖章。"""

    orphan: int = 0
    """原稿 wrap 殘片最長段。"""

    fragment: float = 0.0
    """原稿裡相對同伴缺一塊的圖章比例。"""

    wrap_cut: float = 0.0
    """原稿碰邊卻沒跨縫接上的圖章相對大小。"""

    gutter: float = 0.0
    """原稿地色溝接縫相對內部間距的誤差。"""

    period_rem: float = 0.0
    """原稿邊長對強週期的餘數比例。"""

    wrap_density: float = 1.0
    """原稿 2×2 十字相對內部的前景密度。波點卡在畫框時 > 1。"""

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
    has_crop = (
        (bool(cand.recipe) and any(k == "crop" for k, _ in cand.recipe))
        or "週期裁切" in cand.label
        or "點綴晶格" in cand.label
    )
    has_mincut = (
        (bool(cand.recipe) and any(k == "mincut" for k, _ in cand.recipe))
        or ("最小誤差切" in cand.label)
    )
    has_refill = "清邊補花" in cand.label
    tone = tone_shift(
        src.arr, cand.arr, edge_frac=0.20 if has_refill else 0.0
    )
    errs: list[str] = []

    seam_lim = SEAM_REFILL_MAX if has_refill and (not has_mincut) else SEAM_OK
    if rep.wrap_excess > seam_lim:
        errs.append(f"接縫未消:{rep.wrap_excess:.1f}")

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
    tone_ok = TONE_REFILL_MAX if has_refill else TONE_MAX
    if tone > tone_ok:
        errs.append(f"色調偏移:{tone:.1f}")
    if derr > (DESIGN_MAX_CROP if has_crop else DESIGN_MAX):
        errs.append(f"設計被改壞:{derr:.0f}")
    motif_allow = motif_allowance(src.ink, cand.motif_dense)
    if cand.motif_cut > motif_allow:
        errs.append(f"切線剖開圖案:{cand.motif_cut:.1%}")
    from app.processor import stamp_structure_view

    view = stamp_structure_view(cand.arr)
    hot = wrap_hotspot(view)
    cand.hotspot = hot
    cand.derr = derr
    # 滿鋪幾何真週期：色差／熱點都是 0，但連通域會把整片菱形當切圖，
    # 相位一滾 internal 也會過線。2×2 對得上就不要判死。
    crop_clean = (
        has_crop
        and (not has_refill)
        and hot <= HOTSPOT_OK
        and rep.wrap_excess <= SEAM_PERFECT
    )
    if rep.internal_excess > src.internal_allow and not crop_clean:
        errs.append(f"內部新增斷裂:{rep.internal_excess:.1f}")
    # 真週期裁切／最小誤差切之後，wrap 線本來就是圖案自身的邊緣分佈，
    # 90 分位熱點會偏高，不能當成結構縫。殘縫改看 wrap_excess、切線色差、
    # 殘肢比例。但切線仍剖開圖章且熱點極高（格紋蜜蜂）時，2×2 不是完整元素。
    if has_refill and (not has_mincut) and hot > HOTSPOT_REFILL_OK:
        # 細線雪花跨縫時 wrap 線穿過花瓣，熱點會到 100+，但切圖與色差已過。
        # 對邊顏色對不上（紅塊對藍塊）仍要當結構縫，不能只看切圖比值。
        cut_ok = wrap_cut_ratio(view) <= WRAP_CUT_REFILL_MAX
        seam_ok = rep.wrap_excess <= seam_lim
        if not (cut_ok and seam_ok):
            errs.append(f"結構接縫:{hot:.0f}")
    elif has_crop and hot > HOTSPOT_CROP_OK:
        from app.processor import _fine_grid_aligned

        if not _fine_grid_aligned(cand.arr):
            errs.append(f"結構接縫:{hot:.0f}")
    elif (
        has_mincut
        and (not has_crop)
        and (
            hot > HOTSPOT_CROP_OK
            or (hot > HOTSPOT_CUT_OK and cand.motif_cut > 0)
        )
    ):
        errs.append(f"結構接縫:{hot:.0f}")
    elif (not has_crop) and (not has_mincut) and (not has_refill):
        # 滿版佩斯利／碎花 wrap=0、切圖=0 時，wrap 線穿過圖案本身，熱點 50–65
        # 不是結構縫。稀疏圖章對不上仍走 14。
        hot_lim = HOTSPOT_OK
        if (
            wrap_cut_ratio(view) <= WRAP_CUT_MAX
            and wrap_orphan_run(view) <= ORPHAN_RUN_MAX
        ):
            hot_lim = HOTSPOT_CROP_OK
        if hot > hot_lim:
            errs.append(f"結構接縫:{hot:.0f}")
    if cand.cut_err > CUT_ERR_OK:
        # 最小誤差切後 wrap／熱點／切圖都過關：切線色差是路徑上的殘差，
        # 不是看得見的縫（馬蒂斯有機色塊常 24～30）。
        cut_clean = (
            has_mincut
            and rep.wrap_excess <= SEAM_PERFECT
            and hot <= HOTSPOT_CUT_OK
            and wrap_cut_ratio(view) <= WRAP_CUT_MAX
        )
        if not cut_clean:
            errs.append(f"切線色差:{cand.cut_err:.0f}")
    void = edge_void_ratio(cand.arr)
    skip_void = has_crop and (not has_refill) and derr <= DESIGN_MAX_CROP
    if has_refill and (not has_mincut) and wrap_cut_ratio(view) <= WRAP_CUT_REFILL_MAX:
        # 跨縫件稀疏時 5% 邊帶仍像掏空；真沒補的是切圖／熱點，不是這個。
        skip_void = True
    if (not skip_void) and void > VOID_MAX and void > src.edge_void + VOID_DELTA:
        errs.append(f"邊緣掏空:{void:.0%}")
    orphan = wrap_orphan_run(view)
    if orphan > ORPHAN_RUN_MAX:
        errs.append(f"接縫殘片:{orphan}px")
    frag = motif_fragment_ratio(view)
    cut_stamp = wrap_cut_ratio(view)
    # 真週期裁切／晶格：縫上切圖已另查。藤蔓碎花的實心度離群不是殘肢。
    skip_frag = has_crop and (not has_refill) and cut_stamp <= WRAP_CUT_MAX
    if frag > FRAGMENT_MAX and not skip_frag:
        errs.append(f"圖案殘缺:{frag:.0%}")
    # 滿版連通底紋的 wrap_cut 沒有「圖章」語意，改由熱點把關。
    full_bleed = src.ink >= MOTIF_CUT_DENSE_INK and (not has_refill)
    cut_lim = WRAP_CUT_MAX
    if has_refill and (not has_mincut) and rep.wrap_excess <= SEAM_OK:
        cut_lim = WRAP_CUT_REFILL_MAX
    if cut_stamp > cut_lim and not full_bleed and not crop_clean and not has_mincut:
        errs.append(f"接縫切圖:{cut_stamp:.0%}")
    if not has_mincut:
        crowd = wrap_density_ratio(view)
        if crowd > WRAP_DENSITY_MAX and (has_refill or not has_crop):
            if has_refill and src.wrap_density > WRAP_DENSITY_MAX:
                if crowd > src.wrap_density * 1.12 + 0.03:
                    errs.append(f"接縫過密:{crowd:.2f}")
            else:
                errs.append(f"接縫過密:{crowd:.2f}")
    if (
        src.wrap_cut > WRAP_CUT_MAX
        and (has_mincut or has_crop)
        and (not has_refill)
        and (not full_bleed)
        and hot <= 8.0
        and cand.motif_cut <= MOTIF_CUT_MAX
        and wrap_both_thick_run(view) > ORPHAN_RUN_MAX
    ):
        errs.append("接縫切圖:對邊假接")
    gut = wrap_gutter_error(view)
    if gut > GUTTER_ERR_MAX:
        errs.append(f"接縫錯格:{gut:.0%}")
    elif has_crop and (not has_refill) and not crop_clean:
        prem = wrap_period_remainder(cand.arr)
        if prem > PERIOD_REM_MAX:
            errs.append(f"接縫錯格:{prem:.0%}")
        gut = max(gut, prem)
    if src.needs_native and cand.recipe is None:
        errs.append("無法保色")
    uh, uw = cand.arr.shape[:2]
    need = min_unit_edge(*src.arr.shape[:2])
    if min(uh, uw) < need:
        errs.append(f"單元過小:{min(uh, uw)}<{need}")
    if has_mincut:
        vcls = "C"
    elif has_crop:
        vcls = "B"
    elif has_refill:
        vcls = "A"
    else:
        vcls = classify_unit_class(src)
    for e in verify_tiling(
        cand.arr, cls=vcls, src_density=src.wrap_density
    ):
        if e not in errs:
            errs.append(e)

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
        + orphan * 0.35
        + frag * 120.0
        + min(cut_stamp, 3.0) * 50.0
        + gut * 80.0
        # 接縫代價超線性：殘縫是這個工具唯一不能妥協的東西，愈接近門檻
        # 就愈值得付代價去修。線性權重會讓「超出 4.0 的波點稿」寧可留著
        # 那顆被切台階的白點，也不肯接受一次乾淨的週期裁切。
        + rep.wrap_excess**2 * 3.0
        + max(0.0, rep.internal_excess - src.rep.internal_excess) * 1.0
        + (0.0 if cand.lossless else 3.0)
        # 無法重放到原生通道就得走 ICC 來回轉換，實測平均 4–8 階視覺色偏。
        # 清邊補花本來就在 sRGB 搬圖章，30 分會讓剖開圖章的最小誤差切贏過
        # 已經跨縫貼好的補花。
        + (
            0.0
            if cand.recipe is not None
            else (4.0 if has_refill else 30.0)
        )
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
    from app.processor import stamp_structure_view

    view = stamp_structure_view(base)
    hot_v, hot_h = wrap_hotspot_axes(base)
    view_hot = wrap_hotspot(view)
    # 用嚴格門檻決定要不要展開加工版本，寬鬆門檻只用來判定「絕對不行」。
    # 灰帶案例也要把完整選單擺出來，才輪得到成本函數權衡。
    if source_looks_seamless(
        rep,
        view_hot,
        orphan=wrap_orphan_run(view),
        fragment=motif_fragment_ratio(view),
        wrap_cut=wrap_cut_ratio(view),
        gutter=wrap_gutter_error(view),
        period_rem=wrap_period_remainder(base),
        wrap_density=wrap_density_ratio(view),
    ):
        return out

    # 清邊補花已經把完整圖章跨縫貼好。再最小誤差切會把剛補上的圖章剖開。
    if "清邊補花" in label:
        if rep.wrap_excess > 0.5:
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

    view_cut = wrap_cut_ratio(view)

    # 週期裁切不要再最小誤差切：碎花 wrap_cut 是連通域誤報，再切會把
    # 1152 單元切成 669 還引出內部斷裂。假週期本身也不該靠切線硬修。
    if "週期裁切" in label or "點綴晶格" in label:
        return out

    # 去邊／原圖上圖章還停在畫框時，最小誤差切只會剖開；留給清邊補花。
    # wrap_cut 超過約 100% 是滿版碎花的連通域誤報，不是「半朵卡在畫框」——
    # 那種圖再 skip 下刀，就只剩毀圖的補花可選。
    skip_cut = False
    if WRAP_CUT_MAX < view_cut <= 1.0:
        if label.startswith("去"):
            skip_cut = True
        elif label == "原圖":
            from app.processor import _has_separated_stamps, _looks_like_discrete_motifs

            bg_est = tuple(
                int(v)
                for v in np.median(
                    base[:: max(1, base.shape[0] // 8), :: max(1, base.shape[1] // 8)].reshape(
                        -1, 3
                    ),
                    axis=0,
                )
            )
            skip_cut = _looks_like_discrete_motifs(
                base, bg_est, 40.0
            ) or _has_separated_stamps(base, bg_est, 40.0)
    if skip_cut:
        if rep.wrap_excess > 0.5:
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

    from app.processor import _luma_square_grid_pitch

    # 細格紋切線會把格子剪錯位，只保留裁切／原圖。
    if _luma_square_grid_pitch(base) is not None:
        return out

    # wrap 平均可以是 0（地色接地色），圖章仍對不上。那種軸也必須下刀，
    # 否則只會對「去邊之後才出現平均縫」的 inset 切，邊緣被掏成十字空洞。
    # wrap_cut 高但色差／熱點都冷時（斑點底紋、切在 8px 帶裡的圖章）也要切。
    view_hot_v, view_hot_h = wrap_hotspot_axes(view)
    do_v = (
        rep.excess_v > AXIS_CUT_MIN
        or hot_v > HOTSPOT_OK
        or view_hot_v > HOTSPOT_OK
        or view_cut > WRAP_CUT_MAX
    )
    do_h = (
        rep.excess_h > AXIS_CUT_MIN
        or hot_h > HOTSPOT_OK
        or view_hot_h > HOTSPOT_OK
        or view_cut > WRAP_CUT_MAX
    )
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
        allow = motif_allowance(ink_frac(base), mi.motif_dense)
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
        # 細線幾何在 42% 帶仍會剖到 2%，50% 帶才繞得過（石墨冰裂 1.1%）。
        if struct_fail and recipe is not None:
            h0, w0 = base.shape[:2]
            for oy, ox, tag in (
                (0, w0 // 4, "平移H"),
                (h0 // 4, 0, "平移V"),
            ):
                rolled = np.roll(np.roll(base, -oy, 0), -ox, 1)
                for frac, suffix in ((0.42, ""), (0.50, "更寬帶")):
                    cut_r, mi_r = wrap_mincut(
                        rolled, do_v=do_v, do_h=do_h, max_band_frac=frac
                    )
                    dup_r = mi_r.dup_v * mi_r.band_v / max(rolled.shape[1], 1) + (
                        mi_r.dup_h * mi_r.band_h / max(rolled.shape[0], 1)
                    )
                    out.append(
                        Candidate(
                            cut_r,
                            f"{label}＋{tag}{suffix}＋{mi_r.describe()}",
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
    # 水彩／散點圖章補花過關後，最小誤差切仍常以 wrap=0 進「完美縫」集合，
    # 切線卻貼著淺色描邊走，2×2 看起來像多一圈白邊。補花已把完整圖章跨縫
    # 貼好，就不要再切。
    if any(
        "清邊補花" in c.label and "最小誤差切" not in c.label for c in ok
    ):
        ok = [c for c in ok if "最小誤差切" not in c.label]
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
            best = min(
                tight,
                key=lambda c: (_pref_rank(c), c.derr, c.hotspot, c.cost),
            )
            _lg(
                f"  → 採用（{len(ok)}/{len(cands)} 過關，"
                f"{len(tight)} 個接縫≤{SEAM_PERFECT:.0f}，"
                f"不讓灰帶靠還原贏）：{best.label}"
            )
            return best
        best = min(
            ok,
            key=lambda c: (
                _pref_rank(c),
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
        motif_allow = motif_allowance(src.ink, c.motif_dense)
        motif_pen = 400.0 * c.motif_cut if c.motif_cut > motif_allow else 0.0
        color_pen = 0.0
        if c.color_mean > COLOR_MEAN_MAX:
            color_pen += 80.0
        if c.color_low > COLOR_LOW_MAX:
            color_pen += 80.0
        if c.clipped > CLIP_MAX:
            color_pen += 80.0
        cropish = (
            ("週期裁切" in c.label or "點綴晶格" in c.label)
            and "最小誤差切" not in c.label
        )
        derr_lim = DESIGN_MAX_CROP if cropish else DESIGN_MAX
        derr_pen = 80.0 if c.derr > derr_lim else 0.0
        void_pen = (
            80.0 if any(e.startswith("邊緣掏空") for e in c.errors) else 0.0
        )
        size_pen = (
            80.0 if any(e.startswith("單元過小") for e in c.errors) else 0.0
        )
        orphan_pen = (
            80.0 if any(e.startswith("接縫殘片") for e in c.errors) else 0.0
        )
        frag_pen = (
            80.0 if any(e.startswith("圖案殘缺") for e in c.errors) else 0.0
        )
        wrap_cut_pen = (
            80.0 if any(e.startswith("接縫切圖") for e in c.errors) else 0.0
        )
        fake_join_pen = (
            160.0 if any("對邊假接" in e for e in c.errors) else 0.0
        )
        gutter_pen = (
            80.0 if any(e.startswith("接縫錯格") for e in c.errors) else 0.0
        )
        wrap_cut_mag = 0.0
        for e in c.errors:
            if not e.startswith("接縫切圖:"):
                continue
            rest = e.split(":", 1)[1]
            if rest.endswith("%"):
                try:
                    wrap_cut_mag = float(rest[:-1])
                except ValueError:
                    wrap_cut_mag = 40.0
            else:
                wrap_cut_mag = 40.0
            wrap_cut_mag = min(wrap_cut_mag, 80.0)
            break
        return (
            wrap
            + c.hotspot * 0.5
            + c.cut_err * 0.35
            + internal_pen
            + motif_pen
            + color_pen
            + derr_pen
            + void_pen
            + size_pen
            + orphan_pen
            + frag_pen
            + wrap_cut_pen
            + wrap_cut_mag
            + fake_join_pen
            + gutter_pen,
            c.cost,
        )

    best = min(cands, key=_fail_key)
    reasons = "／".join(best.errors)
    _lg(f"  → 全部未達標，取最接近者：{best.label}（{reasons}）")
    best.label = f"未達標［{reasons}］{best.label}"
    return best


def source_looks_seamless(
    rep: SeamReport,
    hotspot: float,
    *,
    orphan: int = 0,
    fragment: float = 0.0,
    wrap_cut: float = 0.0,
    gutter: float = 0.0,
    period_rem: float = 0.0,
    wrap_density: float = 1.0,
) -> bool:
    """平均超出量為 0 仍可能剖開圖章或留下半截圖案。"""
    hot_lim = HOTSPOT_OK
    if wrap_cut <= WRAP_CUT_MAX and orphan <= ORPHAN_RUN_MAX and gutter <= GUTTER_ERR_MAX:
        # 滿版碎花已對上時，wrap 線穿過圖案，熱點可到 60；70 仍擋住錯相位。
        hot_lim = HOTSPOT_CROP_OK
    return (
        rep.wrap_excess <= SEAM_PERFECT
        and hotspot <= hot_lim
        and orphan <= ORPHAN_RUN_MAX
        and fragment <= FRAGMENT_MAX
        and wrap_cut <= WRAP_CUT_MAX
        and gutter <= GUTTER_ERR_MAX
        and period_rem <= PERIOD_REM_SOURCE_MAX
        and wrap_density <= WRAP_DENSITY_MAX
    )


def source_facts(arr: np.ndarray, *, needs_native: bool = False) -> SourceFacts:
    from app.processor import stamp_structure_view

    view = stamp_structure_view(arr)
    return SourceFacts(
        arr=arr,
        rep=seam_report(arr),
        axis_energy=axis_line_energy(arr),
        needs_native=needs_native,
        ink=ink_frac(arr),
        edge_void=edge_void_ratio(arr),
        hotspot=wrap_hotspot(view),
        orphan=wrap_orphan_run(view),
        fragment=motif_fragment_ratio(view),
        wrap_cut=wrap_cut_ratio(view),
        gutter=wrap_gutter_error(view),
        period_rem=wrap_period_remainder(arr),
        wrap_density=wrap_density_ratio(view),
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
