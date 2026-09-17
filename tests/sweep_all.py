# -*- coding: utf-8 -*-
"""全量掃描：把整批圖跑過一次，產生可目視審查的對照圖與量測報告。

回歸集只跑抽樣，抽樣看不出「整體策略退化成什麼都不做」這種問題。這支
腳本跑完整批，並且**用純網格 2×2 拼接**來驗證——實際擴圖
（`kuotu.image_pipeline.tile_to_canvas`）就是 `canvas.paste` 逐格貼，
`processor.tile_2x2_multi` 那種多圖錯位預覽會把接縫遮起來，不能拿來驗收。

用法：
    python tests/sweep_all.py --inventory             # 清點例圖與成品框
    python tests/sweep_all.py                    # samples/F 全部
    python tests/sweep_all.py --src samples/F --limit 20
    python tests/sweep_all.py --only "18994288478"
    python tests/sweep_all.py --workers 2
    python tests/sweep_all.py --failed-from tests/sweep_out/report_full.json --tag failretry
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import re
import sys
import time
import traceback
from pathlib import Path

# 必須在 numpy / cv2 載入前：每個行程若再各開一套 BLAS 執行緒，
# 3 個行程 × 16 執行緒就會把機器卡死。
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("CV_THREADS", "1")

import numpy as np
from PIL import Image, ImageDraw, ImageFont

Image.MAX_IMAGE_PIXELS = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _configure_stdio() -> None:
    """Windows 主控台常是 cp950，路徑裡的簡體字會直接炸掉。"""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def _lower_priority() -> None:
    """掃描讓出 CPU，避免把桌面卡死。"""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), 0x00004000)
    except Exception:
        pass

SRC = ROOT / "samples" / "F"
OUT = ROOT / "tests" / "sweep_out"
EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}

# 超過此像素數的圖另開小池跑，避免 16 個行程同時吃下 116MP 把記憶體撐爆
HUGE_PIXELS = 30_000_000

# 目視審查用的版面尺寸
_THUMB = 260
_TILE_VIEW = 420
_ZOOM = 420


def _font(size: int) -> ImageFont.ImageFont:
    for name in ("msyh.ttc", "simhei.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(f"C:/Windows/Fonts/{name}", size)
        except OSError:
            continue
    return ImageFont.load_default()


def _fit(img: Image.Image, side: int) -> Image.Image:
    out = img.copy()
    out.thumbnail((side, side), Image.Resampling.LANCZOS)
    return out


def _to_view(arr: np.ndarray) -> Image.Image:
    """任意通道數的陣列轉成可顯示的 RGB。"""
    if arr.ndim == 2:
        return Image.fromarray(arr, "L").convert("RGB")
    c = arr.shape[2]
    if c == 3:
        return Image.fromarray(arr, "RGB")
    if c == 4:
        return Image.fromarray(arr, "CMYK").convert("RGB")
    return Image.fromarray(arr[:, :, :3], "RGB")


def plain_tile_2x2(arr: np.ndarray) -> np.ndarray:
    """純網格 2×2，與實際擴圖一致。接縫落在正中央十字。"""
    return np.tile(arr, (2, 2) + (1,) * (arr.ndim - 2))


def _ticks(
    d: ImageDraw.ImageDraw, box: tuple[int, int, int, int], sx: int, sy: int
) -> None:
    """在面板外緣畫指向接縫的短刻度。不畫在畫面上，才不會遮住要看的東西。"""
    x0, y0, w, h = box
    red = (220, 40, 40)
    for dx, dy, ex, ey in (
        (sx, -7, sx, -1),
        (sx, h + 1, sx, h + 7),
        (-7, sy, -1, sy),
        (w + 1, sy, w + 7, sy),
    ):
        d.line([x0 + dx, y0 + dy, x0 + ex, y0 + ey], fill=red, width=2)


def _contact_sheet(
    src_arr: np.ndarray,
    out_arr: np.ndarray,
    row: dict,
) -> Image.Image:
    """原圖 / 純 2×2 / 接縫 1:1 特寫，橫排成一張審查用對照圖。"""
    src_v = _fit(_to_view(src_arr), _THUMB)
    tiled = plain_tile_2x2(out_arr)
    tiled_v = _fit(_to_view(tiled), _TILE_VIEW)

    # 接縫十字的 1:1 特寫。單元小於視窗時就整段取出，並記下縫的實際位置。
    h, w = out_arr.shape[:2]
    th, tw = tiled.shape[:2]
    zw = min(_ZOOM, tw)
    zh = min(_ZOOM, th)
    x0 = max(0, min(w - zw // 2, tw - zw))
    y0 = max(0, min(h - zh // 2, th - zh))
    zoom_v = _to_view(tiled[y0 : y0 + zh, x0 : x0 + zw])
    zoom_sx, zoom_sy = w - x0, h - y0

    pad = 10
    cap_h = 56
    panels = [
        (src_v, "原圖", None),
        (tiled_v, "純 2×2（接縫在正中十字）", (tiled_v.width // 2, tiled_v.height // 2)),
        (zoom_v, "接縫 1:1 特寫", (zoom_sx, zoom_sy)),
    ]
    width = pad * (len(panels) + 1) + sum(p[0].width for p in panels)
    body_h = max(p[0].height for p in panels) + 20
    height = pad * 2 + body_h + cap_h
    sheet = Image.new("RGB", (width, height), (250, 250, 250))
    d = ImageDraw.Draw(sheet)

    x = pad
    for img, label, seam in panels:
        sheet.paste(img, (x, pad))
        d.rectangle(
            [x - 1, pad - 1, x + img.width, pad + img.height],
            outline=(190, 190, 190),
        )
        if seam is not None:
            _ticks(d, (x, pad, img.width, img.height), seam[0], seam[1])
        d.text(
            (x, pad + img.height + 4), label, fill=(90, 90, 90), font=_font(13)
        )
        x += img.width + pad

    ok = not row["errors"]
    d.rectangle(
        [0, height - cap_h, width, height],
        fill=(232, 245, 233) if ok else (253, 231, 231),
    )
    head = (
        f"{'PASS' if ok else 'FAIL'}  {row['folder']}/{row['name']}  "
        f"{row['src_size'][0]}×{row['src_size'][1]} → "
        f"{row['size'][0]}×{row['size'][1]}  {row['elapsed_s']}s"
        f"{'  [原圖直出]' if row.get('unchanged') else ''}"
    )
    body = (
        f"接縫超出 {row['src_wrap_excess']:.1f} → {row['wrap_excess']:.1f}   "
        f"內部 {row['src_internal_excess']:.1f} → {row['internal_excess']:.1f}   "
        f"熱點 {row.get('src_hotspot', 0):.0f}→{row.get('hotspot', 0):.0f}   "
        f"掏空 {row.get('src_void', 0):.0%}→{row.get('void', 0):.0%}   "
        f"殘片 {row.get('src_orphan', 0)}→{row.get('orphan', 0)}   "
        f"殘缺 {row.get('src_fragment', 0):.0%}→{row.get('fragment', 0):.0%}   "
        f"切圖 {row.get('src_wrap_cut', 0):.0%}→{row.get('wrap_cut', 0):.0%}   "
        f"錯格 {row.get('src_gutter', 0):.0%}→{row.get('gutter', 0):.0%}   "
        f"週期餘 {row.get('src_period_rem', 0):.0%}→{row.get('period_rem', 0):.0%}   "
        f"還原 {row['design_error']:.0f}   色調 {row['tone_shift']:.1f}   "
        f"{row['src_mode']}→{row['out_mode']}"
        f"{'+ICC' if row['keeps_icc'] else ''}   "
        f"{'／'.join(row['errors']) if row['errors'] else row['mode'][:66]}"
    )
    d.text((pad, height - cap_h + 6), head, fill=(20, 20, 20), font=_font(15))
    d.text((pad, height - cap_h + 30), body, fill=(60, 60, 60), font=_font(13))
    return sheet


def check(row: dict) -> list[str]:
    """驗收條件。比生成閘門更緊，專門抓「指標綠、肉眼仍有縫」的假 PASS。"""
    from app.select import (
        DESIGN_MAX_REFILL_FIX,
        DESIGN_VISUAL_MAX,
        EDGE_COLOR_BREAK_MAX,
        FRAGMENT_MAX,
        FRAGMENT_REFILL_MAX,
        GUTTER_ERR_MAX,
        GUTTER_VISUAL_MAX,
        HOTSPOT_CROP_OK,
        HOTSPOT_CUT_OK,
        HOTSPOT_OK,
        HOTSPOT_REFILL_OK,
        ORPHAN_RUN_MAX,
        PERIOD_REM_MAX,
        SEAM_OK,
        SEAM_REFILL_MAX,
        TONE_MAX,
        TONE_REFILL_MAX,
        VOID_DELTA,
        VOID_MAX,
        WRAP_CUT_MAX,
        WRAP_CUT_REFILL_MAX,
        WRAP_CUT_VISUAL_MAX,
        WRAP_DENSITY_MAX,
        WRAP_DENSITY_MIN,
        WRAP_DENSITY_SPARSE_MIN,
        WRAP_DENSITY_VISUAL_MAX,
        motif_allowance,
    )

    errs: list[str] = []
    mode = row.get("mode") or ""
    has_crop = "週期裁切" in mode or "點綴晶格" in mode
    has_mincut = "最小誤差切" in mode
    has_refill = "清邊補花" in mode
    hot = float(row.get("hotspot") or 0.0)
    wrap_cut_early = float(row.get("wrap_cut") or 0.0)
    dens_early = float(row.get("wrap_density") or 0.0)
    src_cut = float(row.get("src_wrap_cut") or 0.0)
    crop_clean = (
        has_crop
        and "清邊補花" not in mode
        and hot <= 14.0
        and float(row.get("wrap_excess") or 0.0) <= 2.0
    )

    # 1. 這是全部的重點：輸出自己接自己時不能看得出縫。
    seam_lim = SEAM_OK
    if has_refill and "最小誤差切" not in mode:
        seam_lim = SEAM_REFILL_MAX
    if row["wrap_excess"] > seam_lim:
        errs.append(f"接縫未消:{row['wrap_excess']:.1f}")

    allow = max(row["src_internal_excess"], 6.0) * 1.15 + 2.0
    wrap_ok_ie = (
        has_refill
        and wrap_cut_early <= WRAP_CUT_VISUAL_MAX
        and WRAP_DENSITY_MIN <= dens_early <= WRAP_DENSITY_MAX
        and float(row.get("wrap_excess") or 0.0) <= seam_lim
    )
    if wrap_ok_ie:
        allow += 4.0
    if row["internal_excess"] > allow and not crop_clean:
        errs.append(f"內部新增斷裂:{row['internal_excess']:.1f}>{allow:.1f}")

    tone_ok = TONE_REFILL_MAX if has_refill else TONE_MAX
    # 散點圖章清邊後邊緣帶色占比會變；與 measure 一致放寬到 40。
    if has_refill and float(row.get("ink") or 0.0) < 0.42:
        tone_ok = max(tone_ok, 40.0)
    if row["tone_shift"] > tone_ok:
        errs.append(f"色調偏移:{row['tone_shift']:.1f}")

    # 加工件用更緊的毀圖門檻；原圖直出仍用災難級 90。
    # 清邊補花重排圖章修掉原稿切圖：derr 高是方法本身，不是打散連續底紋。
    derr = float(row.get("design_error") or 0.0)
    derr_lim = DESIGN_VISUAL_MAX
    if has_refill and "最小誤差切" not in mode and src_cut > WRAP_CUT_MAX:
        if (
            wrap_cut_early <= WRAP_CUT_REFILL_MAX
            and WRAP_DENSITY_MIN <= dens_early <= WRAP_DENSITY_MAX
        ):
            derr_lim = DESIGN_MAX_REFILL_FIX
    if has_crop or has_refill:
        if derr > derr_lim:
            errs.append(f"設計被改壞:{derr:.0f}")
    elif derr > 90.0:
        errs.append(f"設計被改壞:{derr:.0f}")

    if row["src_mode"] == "CMYK" and row["out_mode"] != "CMYK":
        errs.append(f"色彩空間被降級:{row['out_mode']}")

    if has_refill and "最小誤差切" not in mode and hot > HOTSPOT_REFILL_OK:
        cut_ok = wrap_cut_early <= WRAP_CUT_REFILL_MAX
        seam_ok = float(row.get("wrap_excess") or 0.0) <= SEAM_REFILL_MAX
        if not (cut_ok and seam_ok):
            errs.append(f"結構接縫:{hot:.0f}")
    elif has_crop and hot > HOTSPOT_CROP_OK:
        if not bool(row.get("fine_aligned")):
            errs.append(f"結構接縫:{hot:.0f}")
    elif (not has_crop) and (not has_mincut) and not has_refill:
        hot_lim = HOTSPOT_OK
        if wrap_cut_early <= WRAP_CUT_MAX and int(row.get("orphan") or 0) <= ORPHAN_RUN_MAX:
            hot_lim = HOTSPOT_CROP_OK
        if hot > hot_lim:
            errs.append(f"結構接縫:{hot:.0f}")

    cut_m = re.search(r"切到圖案([0-9.]+)%", mode)
    dense_m = re.search(r"帶內圖案([0-9.]+)%", mode)
    ink_m = re.search(r"前景 ([0-9.]+)%", mode)
    motif_cut = float(cut_m.group(1)) / 100.0 if cut_m else 0.0
    motif_dense = float(dense_m.group(1)) / 100.0 if dense_m else 0.0
    ink = float(ink_m.group(1)) / 100.0 if ink_m else float(row.get("ink") or 0.0)
    if motif_cut > motif_allowance(ink, motif_dense):
        errs.append(f"切線剖開圖案:{motif_cut:.1%}")
    if has_mincut and (not has_crop) and hot > HOTSPOT_CUT_OK and motif_cut > 0:
        errs.append(f"結構接縫:{hot:.0f}")
    elif has_mincut and (not has_crop) and hot > HOTSPOT_CROP_OK:
        errs.append(f"結構接縫:{hot:.0f}")

    void = float(row.get("void") or 0.0)
    src_void = float(row.get("src_void") or 0.0)
    skip_void = has_crop and not has_refill and derr <= 35.0
    if (
        has_refill
        and "最小誤差切" not in mode
        and wrap_cut_early <= WRAP_CUT_REFILL_MAX
        and float(row.get("wrap_density") or 0.0) >= 0.82
    ):
        skip_void = True
    if (not skip_void) and void > VOID_MAX and void > src_void + VOID_DELTA:
        errs.append(f"邊緣掏空:{void:.0%}")

    orphan = int(row.get("orphan") or 0)
    if orphan > ORPHAN_RUN_MAX:
        errs.append(f"接縫殘片:{orphan}px")
    frag = float(row.get("fragment") or 0.0)
    frag_lim = FRAGMENT_MAX
    if (
        has_refill
        and "最小誤差切" not in mode
        and wrap_cut_early <= WRAP_CUT_VISUAL_MAX
        and float(row.get("wrap_excess") or 0.0) <= SEAM_OK
    ):
        frag_lim = FRAGMENT_REFILL_MAX
    if frag > frag_lim and not (
        has_crop and not has_refill and wrap_cut_early <= WRAP_CUT_MAX
    ):
        errs.append(f"圖案殘缺:{frag:.0%}")

    wrap_cut = float(row.get("wrap_cut") or 0.0)
    full_bleed = ink >= 0.42 and not has_refill
    cut_lim = WRAP_CUT_MAX
    if has_refill and "最小誤差切" not in mode and float(row.get("wrap_excess") or 0) <= 5.0:
        cut_lim = (
            WRAP_CUT_VISUAL_MAX if ink < 0.42 else WRAP_CUT_REFILL_MAX
        )
    if wrap_cut > cut_lim and not full_bleed and not has_mincut:
        if (not crop_clean) or wrap_cut >= 0.85:
            errs.append(f"接縫切圖:{wrap_cut:.0%}")
    # 稀疏圖章：上面 cut_lim 已對齊 VISUAL；此段僅防漏。
    if (
        not full_bleed
        and not has_mincut
        and ink < 0.42
        and wrap_cut > WRAP_CUT_VISUAL_MAX
        and f"接縫切圖:{wrap_cut:.0%}" not in errs
    ):
        errs.append(f"接縫切圖:{wrap_cut:.0%}")

    dens = float(row.get("wrap_density") or 0.0)
    if dens > WRAP_DENSITY_MAX and (not has_mincut) and not full_bleed:
        src_d = float(row.get("src_wrap_density") or 0.0)
        if has_refill and src_d > WRAP_DENSITY_MAX:
            if dens > src_d * 1.12 + 0.03:
                errs.append(f"接縫過密:{dens:.2f}")
        else:
            errs.append(f"接縫過密:{dens:.2f}")
    elif dens > WRAP_DENSITY_VISUAL_MAX and (not has_mincut) and not full_bleed:
        errs.append(f"接縫過密:{dens:.2f}")
    if dens < WRAP_DENSITY_MIN and not full_bleed:
        errs.append(f"接縫過疏:{dens:.2f}")
    elif (
        dens > 0
        and dens < WRAP_DENSITY_SPARSE_MIN
        and ink < 0.22
        and not full_bleed
        and not has_mincut
    ):
        errs.append(f"接縫過疏:{dens:.2f}")

    gutter = float(row.get("gutter") or 0.0)
    gut_lim = GUTTER_VISUAL_MAX if not full_bleed else GUTTER_ERR_MAX
    if gutter > gut_lim:
        errs.append(f"接縫錯格:{gutter:.0%}")
    # period_rem：與 select 對齊——清邊補花只在「原稿週期本來乾淨卻被打散」時判錯格。
    prem = float(row.get("period_rem") or 0.0)
    src_prem = float(row.get("src_period_rem") or 0.0)
    if prem > PERIOD_REM_MAX:
        if has_refill and "最小誤差切" not in mode:
            if src_prem <= PERIOD_REM_MAX:
                errs.append(f"接縫錯格:{prem:.0%}")
        else:
            errs.append(f"接縫錯格:{prem:.0%}")

    edge_brk = float(row.get("edge_color_break") or 0.0)
    if edge_brk > EDGE_COLOR_BREAK_MAX and not full_bleed:
        errs.append(f"接縫錯色:{edge_brk:.1f}")

    if "未達標" in mode:
        errs.append("mode_未達標")
    return errs


def _job_path(folder: str, name: str) -> Path:
    return (SRC / folder / name) if folder else SRC / name


def _sheet_name(folder: str, name: str) -> str:
    rel = f"{folder}/{name}" if folder else name
    safe = rel.replace("\\", "/").replace("/", "__")
    if len(safe) > 160:
        import hashlib

        digest = hashlib.md5(rel.encode("utf-8")).hexdigest()[:12]
        safe = f"{digest}__{Path(name).name}"
    return safe


def _write_wall_thumb(img: Image.Image, dest: Path) -> None:
    """審查牆縮圖：整張 2×2 保比例，用畫面中位色墊成方格，不要裁接縫特寫。"""
    view = img.convert("RGB")
    w, h = view.size
    if w <= 0 or h <= 0:
        return
    side = max(w, h)
    arr = np.asarray(view)
    fill = tuple(int(x) for x in np.median(arr.reshape(-1, 3), axis=0))
    canvas = Image.new("RGB", (side, side), fill)
    canvas.paste(view, ((side - w) // 2, (side - h) // 2))
    canvas.thumbnail((240, 240), Image.Resampling.LANCZOS)
    dest.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(dest, quality=85)


# 單張牆鐘硬上限；頂層 Process 逾時 terminate（Pool worker 不能再 spawn）。
# Windows TerminateProcess 不可攔截：硬殺前先排空 queue，並用子行程最後一筆真結果。
# 軟目標略低於 90s，留給量測／queue.put；不可把 HARD_CASE_TIMEOUT_S 再往上調。
HARD_CASE_TIMEOUT_S = 90.0
WORKER_AIM_S = 82.0
_QUEUE_BEFORE_KILL_S = 0.25

_PROGRESS_Q = None
_WORKER_AIM_S = 0.0
_CASE_T0 = 0.0


def _worker_aim_s() -> float:
    return _WORKER_AIM_S if _WORKER_AIM_S > 0 else WORKER_AIM_S


def _seconds_left(t0: float | None = None) -> float:
    if _PROGRESS_Q is None and _WORKER_AIM_S <= 0:
        return 1e9
    start = _CASE_T0 if _CASE_T0 > 0 else (t0 if t0 is not None else time.perf_counter())
    return _worker_aim_s() - (time.perf_counter() - start)


def _apply_worker_gen_budget(aim_s: float) -> None:
    """壓低 select.gen_budget_for_shape；make_seamless_hard_cut 用它算內部 deadline。"""
    try:
        import app.select as select
    except Exception:
        return
    cap = max(8.0, float(aim_s))
    orig = select.gen_budget_for_shape

    def _capped(h: int, w: int, _orig=orig, _cap=cap) -> float:
        return min(float(_orig(h, w)), _cap)

    select.gen_budget_for_shape = _capped
    if hasattr(select, "GEN_BUDGET_S"):
        select.GEN_BUDGET_S = min(float(select.GEN_BUDGET_S), cap)
    if hasattr(select, "GEN_BUDGET_LARGE_S"):
        select.GEN_BUDGET_LARGE_S = min(float(select.GEN_BUDGET_LARGE_S), cap)


def _make_seamless_with_budget(img, bg, t_case: float):
    import inspect

    from app.processor import make_seamless_hard_cut

    if _WORKER_AIM_S <= 0:
        return make_seamless_hard_cut(img, bg)
    kwargs: dict = {}
    deadline = t_case + _WORKER_AIM_S
    try:
        params = inspect.signature(make_seamless_hard_cut).parameters
    except (TypeError, ValueError):
        params = {}
    if "deadline" in params:
        kwargs["deadline"] = deadline
    elif "budget_s" in params:
        kwargs["budget_s"] = max(1.0, deadline - time.perf_counter())
    return make_seamless_hard_cut(img, bg, **kwargs)


def _case_snap_path(folder: str, name: str) -> Path:
    return OUT / "_case_snap" / f"{_sheet_name(folder, name)}.json"


def _write_case_snap(row: dict) -> None:
    try:
        dest = _case_snap_path(row.get("folder") or "", row.get("name") or "")
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        tmp.write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")
        tmp.replace(dest)
    except Exception:
        pass


def _read_case_snap(folder: str, name: str) -> dict | None:
    path = _case_snap_path(folder, name)
    try:
        if not path.is_file():
            return None
        rec = json.loads(path.read_text(encoding="utf-8"))
        return rec if isinstance(rec, dict) else None
    except Exception:
        return None


def _clear_case_snap(folder: str, name: str) -> None:
    try:
        path = _case_snap_path(folder, name)
        if path.is_file():
            path.unlink()
        tmp = path.with_suffix(path.suffix + ".tmp")
        if tmp.is_file():
            tmp.unlink()
    except Exception:
        pass


def _publish_case_row(row: dict) -> None:
    """增量交卷：快照檔 + queue。父行程硬殺前取最後一筆可用 dict。"""
    _write_case_snap(row)
    q = _PROGRESS_Q
    if q is None:
        return
    try:
        q.put(("ok", dict(row)), timeout=0.4)
    except Exception:
        pass


def _payload_from_msg(msg) -> dict | None:
    if isinstance(msg, dict):
        return msg
    if isinstance(msg, tuple) and len(msg) == 2 and isinstance(msg[1], dict):
        return msg[1]
    return None


def _usable_case_row(row: dict | None) -> bool:
    return _row_rank(row) > 0


def _row_rank(row: dict | None) -> int:
    """3=真單元列，2=跳過／有 mode，1=例外，0=假超時或空。"""
    if not isinstance(row, dict):
        return 0
    if row.get("timed_out") and float(row.get("wrap_excess") or 0) == 999.0:
        return 0
    size = row.get("size") or [0, 0]
    has_tile = False
    if isinstance(size, (list, tuple)) and len(size) >= 2:
        try:
            has_tile = int(size[0]) > 0 and int(size[1]) > 0
        except (TypeError, ValueError):
            has_tile = False
    wrap = row.get("wrap_excess")
    real_wrap = wrap is not None and float(wrap) != 999.0
    if has_tile or real_wrap:
        return 3
    if row.get("skipped"):
        return 2
    if row.get("mode") and "超時" not in str(row.get("mode")):
        return 2
    if row.get("error") or row.get("errors") == ["EXCEPTION"]:
        return 1
    return 0


def _drain_queue(q, wait_s: float = 0.0):
    last_any = None
    last_ok = None

    def _eat(item) -> None:
        nonlocal last_any, last_ok
        last_any = item
        rank = _row_rank(_payload_from_msg(item))
        old_rank = _row_rank(_payload_from_msg(last_ok))
        if rank > old_rank or (rank == old_rank and rank > 0):
            last_ok = item

    if wait_s > 0:
        try:
            _eat(q.get(timeout=wait_s))
        except Exception:
            pass
    while True:
        try:
            _eat(q.get_nowait())
        except Exception:
            break
    return last_ok if last_ok is not None else last_any


def _prefer_msg(old, new):
    if new is None:
        return old
    new_rank = _row_rank(_payload_from_msg(new))
    old_rank = _row_rank(_payload_from_msg(old))
    if new_rank > old_rank:
        return new
    if new_rank == old_rank and new_rank > 0:
        return new
    return old if old is not None else new


def _row_from_sources(*sources) -> dict | None:
    best = None
    best_rank = 0
    for src in sources:
        row = src if isinstance(src, dict) else _payload_from_msg(src)
        rank = _row_rank(row)
        extra = 0
        if isinstance(row, dict):
            extra += 2 if row.get("sheet") else 0
            extra += 1 if row.get("wall") else 0
        best_extra = 0
        if isinstance(best, dict):
            best_extra += 2 if best.get("sheet") else 0
            best_extra += 1 if best.get("wall") else 0
        if rank > best_rank or (rank == best_rank and rank > 0 and extra > best_extra):
            best = row
            best_rank = rank
    return best


def _close_queue(q, join: bool = True) -> None:
    try:
        q.close()
    except Exception:
        pass
    if not join:
        return
    try:
        q.join_thread()
    except Exception:
        pass


def _queue_put(q, item, timeout: float = 8.0) -> None:
    try:
        q.put(item, timeout=timeout)
    except Exception:
        pass


def _case_process_main(queue, job: tuple[str, str, bool], init_args: tuple) -> None:
    """頂層 spawn 目標（必須是模組級函數，Windows 才能 pickle）。"""
    global _PROGRESS_Q, _CASE_T0
    _CASE_T0 = time.perf_counter()
    _PROGRESS_Q = queue
    try:
        _init_worker(*init_args)
        _apply_worker_gen_budget(_worker_aim_s())
        try:
            _queue_put(queue, ("ok", run_case(job)))
        except Exception as exc:  # noqa: BLE001
            folder, name, _ws = job
            _queue_put(
                queue,
                (
                    "err",
                    {
                        "folder": folder,
                        "name": name,
                        "error": f"{exc}",
                        "errors": ["EXCEPTION"],
                        "elapsed_s": 0.0,
                    },
                ),
            )
    finally:
        _PROGRESS_Q = None


def run_case(job: tuple[str, str, bool]) -> dict:
    folder, name, write_sheet = job
    t_case = time.perf_counter()
    from app.color_utils import detect_background
    from app.processor import (
        _fine_grid_aligned,
        _to_rgb_array,
        stamp_structure_view,
    )
    from app.quality import (
        axis_line_energy,
        design_error,
        edge_void_ratio,
        geometry_fidelity,
        ink_frac,
        motif_fragment_ratio,
        seam_report,
        tone_shift,
        wrap_cut_ratio,
        wrap_density_ratio,
        wrap_edge_color_break,
        wrap_gutter_error,
        wrap_hotspot,
        wrap_orphan_run,
        wrap_period_remainder,
    )
    from app.triage import VERDICT_TILEABLE, triage

    path = _job_path(folder, name)
    row: dict = {"folder": folder, "name": name}
    try:
        img = Image.open(path)
        img.load()
        decision = triage(img)
        row["triage"] = decision.verdict
        if decision.verdict != VERDICT_TILEABLE:
            row["skipped"] = decision.verdict
            row["mode"] = decision.describe()
            row["errors"] = []
            row["elapsed_s"] = 0.0
            if write_sheet:
                thumb = OUT / "wall" / f"skip__{_sheet_name(folder, name)}.jpg"
                _write_wall_thumb(img.convert("RGB"), thumb)
                row["wall"] = str(thumb.relative_to(OUT))
            return row

        bg = detect_background(img)
        src = _to_rgb_array(img, bg)
        s_rep = seam_report(src)

        t0 = time.perf_counter()
        unit, mode = _make_seamless_with_budget(img, bg, t_case)
        elapsed = time.perf_counter() - t0

        out = _to_rgb_array(unit, bg)
        o_rep = seam_report(out)

        src_view = stamp_structure_view(src)
        out_view = stamp_structure_view(out)
        row.update(
            {
                "mode": mode,
                "size": list(unit.size),
                "src_size": list(img.size),
                "src_mode": img.mode,
                "out_mode": unit.mode,
                "keeps_icc": bool(unit.info.get("icc_profile")),
                "src_wrap": round(s_rep.wrap_raw, 2),
                "src_wrap_excess": round(s_rep.wrap_excess, 2),
                "src_internal_excess": round(s_rep.internal_excess, 2),
                "src_axis_energy": round(axis_line_energy(src), 3),
                "wrap": round(o_rep.wrap_raw, 2),
                "wrap_excess": round(o_rep.wrap_excess, 2),
                "internal_excess": round(o_rep.internal_excess, 2),
                "internal_at": [
                    round(o_rep.internal_at_v, 3),
                    round(o_rep.internal_at_h, 3),
                ],
                "fidelity": round(geometry_fidelity(src, out), 3),
                "tone_shift": round(
                    tone_shift(
                        src,
                        out,
                        edge_frac=0.20 if "清邊補花" in mode else 0.0,
                    ),
                    2,
                ),
                "design_error": round(design_error(src, out), 2),
                "src_hotspot": round(wrap_hotspot(src_view), 2),
                "hotspot": round(wrap_hotspot(out_view), 2),
                "src_void": round(edge_void_ratio(src), 3),
                "void": round(edge_void_ratio(out), 3),
                "src_orphan": wrap_orphan_run(src_view),
                "orphan": wrap_orphan_run(out_view),
                "src_fragment": round(motif_fragment_ratio(src_view), 3),
                "fragment": round(motif_fragment_ratio(out_view), 3),
                "src_wrap_cut": round(wrap_cut_ratio(src_view), 3),
                "wrap_cut": round(wrap_cut_ratio(out_view), 3),
                "src_wrap_density": round(wrap_density_ratio(src_view), 3),
                "wrap_density": round(wrap_density_ratio(out_view), 3),
                "src_gutter": round(wrap_gutter_error(src_view), 3),
                "gutter": round(wrap_gutter_error(out_view), 3),
                "src_period_rem": round(wrap_period_remainder(src), 3),
                "period_rem": round(wrap_period_remainder(out), 3),
                "edge_color_break": round(wrap_edge_color_break(out_view), 2),
                "ink": round(ink_frac(out), 3),
                "fine_aligned": bool(_fine_grid_aligned(out)),
                "unchanged": bool(
                    src.shape == out.shape and np.array_equal(src, out)
                ),
                "elapsed_s": round(elapsed, 1),
            }
        )
        row["errors"] = check(row)
        row["verify_cls"] = ""
        row["verify_errors"] = []
        # 量測一完就交卷：審查牆 I/O 再慢，父行程也能拿到真圖列。
        _publish_case_row(row)

        if write_sheet:
            remain = _seconds_left(t_case)
            tag = "FAIL" if row["errors"] else "pass"
            if remain >= 3.0:
                sheet = _contact_sheet(src, out, row)
                dest = OUT / "sheets" / f"{tag}__{_sheet_name(folder, name)}.png"
                dest.parent.mkdir(parents=True, exist_ok=True)
                sheet.save(dest)
                row["sheet"] = str(dest.relative_to(OUT))
            if remain >= 1.2:
                tiled = Image.fromarray(plain_tile_2x2(out))
                thumb = OUT / "wall" / f"{tag}__{_sheet_name(folder, name)}.jpg"
                _write_wall_thumb(tiled, thumb)
                row["wall"] = str(thumb.relative_to(OUT))
            if row.get("sheet") or row.get("wall"):
                _publish_case_row(row)
    except Exception as exc:  # noqa: BLE001 — 批次要繼續
        if _row_rank(row) >= 3:
            row["sheet_error"] = f"{exc}"
            _publish_case_row(row)
            return row
        row["error"] = f"{exc}"
        row["trace"] = traceback.format_exc(limit=6)
        row["errors"] = ["EXCEPTION"]
    return row


def _timeout_fail_row(folder: str, name: str, elapsed: float) -> dict:
    return {
        "folder": folder,
        "name": name,
        "mode": f"前景 ?｜未達標［超時:{HARD_CASE_TIMEOUT_S:.0f}s］",
        "errors": ["mode_未達標", f"超時:{HARD_CASE_TIMEOUT_S:.0f}s"],
        "elapsed_s": round(elapsed, 1),
        "timed_out": True,
        "src_wrap_excess": 0.0,
        "wrap_excess": 999.0,
        "src_hotspot": 0.0,
        "hotspot": 999.0,
        "wrap_cut": 8.0,
        "wrap_density": 0.0,
        "edge_color_break": 99.0,
        "size": [0, 0],
        "src_size": [0, 0],
    }


def _run_jobs_hard_timeout(
    jobs: list[tuple[str, str, bool]],
    n_proc: int,
    init_args: tuple,
    on_result,
) -> None:
    """頂層非 daemon Process。硬殺前先排空 queue，優先用子行程已交的真結果。"""
    ctx = mp.get_context("spawn")
    pending = list(jobs)
    active: list[tuple] = []

    def _spawn(job):
        q = ctx.Queue(8)
        p = ctx.Process(target=_case_process_main, args=(q, job, init_args))
        p.daemon = False
        p.start()
        return p, q, job, time.perf_counter(), None

    def _finish(item, payload: dict, join_q: bool) -> None:
        p, q, job, _t0, _last = item
        folder, name, _ws = job
        on_result(payload)
        _clear_case_snap(folder, name)
        try:
            p.join(1)
        except Exception:
            pass
        _close_queue(q, join=join_q)
        try:
            active.remove(item)
        except ValueError:
            pass

    while pending or active:
        while pending and len(active) < max(1, n_proc):
            active.append(_spawn(pending.pop(0)))
        progressed = False
        for item in list(active):
            p, q, job, t0, last_msg = item
            folder, name, _ws = job
            last_msg = _prefer_msg(last_msg, _drain_queue(q, wait_s=0.0))
            try:
                idx = active.index(item)
            except ValueError:
                continue
            item = (p, q, job, t0, last_msg)
            active[idx] = item
            alive = p.is_alive()
            elapsed = time.perf_counter() - t0
            if not alive:
                last_msg = _prefer_msg(last_msg, _drain_queue(q, wait_s=2.0))
                payload = _row_from_sources(last_msg, _read_case_snap(folder, name))
                if payload is None:
                    payload = _timeout_fail_row(folder, name, elapsed)
                _finish(item, payload, join_q=True)
                progressed = True
            elif elapsed >= HARD_CASE_TIMEOUT_S:
                # 子行程可能已 put；TerminateProcess 之後再也交不出新結果。
                last_msg = _prefer_msg(
                    last_msg, _drain_queue(q, wait_s=_QUEUE_BEFORE_KILL_S)
                )
                payload = _row_from_sources(last_msg, _read_case_snap(folder, name))
                if p.is_alive():
                    p.terminate()
                    p.join(5)
                    last_msg = _prefer_msg(last_msg, _drain_queue(q, wait_s=0.0))
                    payload = _row_from_sources(
                        last_msg, payload, _read_case_snap(folder, name)
                    )
                if payload is None:
                    payload = _timeout_fail_row(folder, name, elapsed)
                _finish(item, payload, join_q=False)
                progressed = True
        if not progressed:
            time.sleep(0.25)



def collect(src: Path, only: str, limit: int) -> list[tuple[str, str]]:
    jobs: list[tuple[str, str]] = []
    for p in sorted(src.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in EXTS:
            continue
        rel = p.relative_to(src)
        folder = "" if rel.parent == Path(".") else rel.parent.as_posix()
        key = rel.as_posix()
        if only and only not in key and only not in p.name:
            continue
        jobs.append((folder, rel.name))
    if limit and len(jobs) > limit:
        # 均勻抽樣，避免 --limit 只掃到排序最前的同一批 SKU 單元圖
        step = len(jobs) / float(limit)
        jobs = [jobs[min(len(jobs) - 1, int(i * step))] for i in range(limit)]
    return jobs


def _pixels(folder: str, name: str) -> int:
    try:
        with Image.open(_job_path(folder, name)) as im:
            return im.size[0] * im.size[1]
    except Exception:  # noqa: BLE001
        return 0


def _init_worker(src: str, out: str, aim_s: str | float | None = None) -> None:
    global SRC, OUT, _WORKER_AIM_S
    SRC = Path(src)
    OUT = Path(out)
    if aim_s is not None and str(aim_s) != "":
        try:
            _WORKER_AIM_S = float(aim_s)
        except (TypeError, ValueError):
            _WORKER_AIM_S = WORKER_AIM_S
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    try:
        import cv2

        cv2.setNumThreads(1)
    except Exception:
        pass
    _lower_priority()


def _content_hash(folder: str, name: str) -> str:
    path = _job_path(folder, name)
    return hashlib.md5(path.read_bytes()).hexdigest()


def dedup_pairs(
    pairs: list[tuple[str, str]],
) -> tuple[list[tuple[str, str]], dict[str, list[tuple[str, str]]]]:
    """依內容雜湊去重，回傳（代表張, hash→全部路徑）。"""
    groups: dict[str, list[tuple[str, str]]] = {}
    order: list[str] = []
    for item in pairs:
        digest = _content_hash(*item)
        if digest not in groups:
            order.append(digest)
            groups[digest] = []
        groups[digest].append(item)
    unique = [groups[h][0] for h in order]
    return unique, groups


def replicate_duplicates(
    rows: list[dict],
    groups: dict[str, list[tuple[str, str]]],
) -> list[dict]:
    """把代表張的結果套到內容相同的其他路徑。"""
    by_key = {(r.get("folder") or "", r.get("name") or ""): r for r in rows}
    extra: list[dict] = []
    for members in groups.values():
        primary = members[0]
        prow = by_key.get(primary)
        if prow is None:
            continue
        for dup in members[1:]:
            copy = dict(prow)
            copy["folder"], copy["name"] = dup
            copy["duplicate_of"] = f"{primary[0]}/{primary[1]}" if primary[0] else primary[1]
            extra.append(copy)
    return extra


def _bucket(row: dict) -> str:
    if row.get("error") or (row.get("errors") and "EXCEPTION" in row["errors"]):
        return "fail"
    skipped = row.get("skipped")
    if skipped == "framed":
        return "skip_framed"
    if skipped == "not_pattern":
        return "skip_cover"
    if skipped:
        return "skip_cover"
    if row.get("errors"):
        return "fail"
    mode = row.get("mode") or ""
    wrap = float(row.get("wrap_excess") or 0)
    derr = float(row.get("design_error") or 0)
    hot = float(row.get("hotspot") or 0)
    void = float(row.get("void") or 0)
    orphan = float(row.get("orphan") or 0)
    frag = float(row.get("fragment") or 0)
    gutter = float(row.get("gutter") or 0)
    dens = float(row.get("wrap_density") or 0)
    verify = row.get("verify_errors") or []
    if (
        "未達標" in mode
        or wrap > 2.0
        or derr > 40.0
        or hot > 8.0
        or void > 0.15
        or orphan > 16
        or frag > 0.06
        or gutter > 0.20
        or dens > 1.15
        or verify
    ):
        return "suspect"
    return "pass"


def write_index(rows: list[dict]) -> Path:
    """分區 2×2 審查牆：FAIL／可疑／跳過-封面／跳過-有框／PASS。"""
    titles = (
        ("fail", "FAIL 未通過"),
        ("suspect", "可疑（接縫灰帶／還原偏高／未達標）"),
        ("skip_cover", "跳過：產品封面圖"),
        ("skip_framed", "跳過：成品外框"),
        ("pass", "PASS"),
    )
    buckets: dict[str, list[dict]] = {k: [] for k, _ in titles}
    for r in rows:
        if r.get("duplicate_of"):
            continue
        buckets[_bucket(r)].append(r)

    n_fail = len(buckets["fail"])
    n_skip = len(buckets["skip_cover"]) + len(buckets["skip_framed"])
    n_run = sum(1 for r in rows if not r.get("skipped") and not r.get("duplicate_of"))
    parts = [
        "<!doctype html><meta charset='utf-8'>",
        "<title>四方連續 2×2 審查牆</title>",
        "<style>"
        "body{font:14px system-ui;background:#111;color:#eee;margin:16px}"
        "h2{margin:28px 0 8px}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:10px}"
        "figure{margin:0;background:#1c1c1c;padding:6px;border-radius:6px}"
        "img{width:100%;aspect-ratio:1;object-fit:contain;object-position:center;background:#1c1c1c;display:block}"
        "figcaption{font-size:11px;margin-top:6px;color:#bbb;word-break:break-all}"
        "a{color:#9cf}"
        "</style>",
        f"<h1>掃描 {n_run} 張（跳過 {n_skip}），未通過 {n_fail}</h1>",
    ]
    for key, title in titles:
        items = buckets[key]
        items.sort(
            key=lambda r: (
                -float(r.get("wrap_density") or 0),
                -float(r.get("hotspot") or 0),
                -float(r.get("wrap_cut") or 0),
            )
        )
        parts.append(f"<h2>{title}（{len(items)}）</h2>")
        parts.append('<div class="grid">')
        for r in items:
            wall = r.get("wall") or r.get("sheet") or ""
            sheet = r.get("sheet") or wall
            label = f"{r.get('folder') or ''}/{r.get('name')}"
            cap = r.get("mode") or r.get("skipped") or ""
            if r.get("errors"):
                cap = "／".join(r["errors"]) + " " + cap
            dens = r.get("wrap_density")
            if dens is not None:
                cap = f"密度 {float(dens):.2f}｜{cap}"
            if "整張重排" in (r.get("mode") or ""):
                cap = "整張重排｜" + cap
            if r.get("verify_errors"):
                cap = "驗證:" + "／".join(r["verify_errors"]) + " " + cap
            hot = r.get("hotspot")
            if hot is not None:
                cap = f"熱點 {float(hot):.0f}｜" + cap
            href = sheet if sheet else "#"
            src = wall if wall else sheet
            if not src:
                continue
            parts.append(
                "<figure>"
                f"<a href='{href}' target='_blank'><img src='{src}' alt=''></a>"
                f"<figcaption>{label}<br>{cap[:160]}</figcaption>"
                "</figure>"
            )
        parts.append("</div>")
    dest = OUT / "index.html"
    dest.write_text("\n".join(parts), encoding="utf-8")
    return dest


def inventory(src: Path, probe: int = 80) -> int:
    """清點例圖並抽檢分流（含合成對照，避免誤殺黑底印花）。"""
    from PIL import ImageOps

    from app.triage import VERDICT_FRAMED, VERDICT_TILEABLE, triage

    pairs = collect(src, "", 0)
    by_ext: dict[str, int] = {}
    for _folder, name in pairs:
        ext = Path(name).suffix.lower()
        by_ext[ext] = by_ext.get(ext, 0) + 1
    print(f"來源 {src}")
    print(f"共 {len(pairs)} 張  {by_ext}")

    flower = Image.new("RGB", (240, 240), (40, 120, 80))
    for y in range(40, 200, 24):
        for x in range(40, 200, 24):
            ImageDraw.Draw(flower).ellipse((x, y, x + 14, y + 14), fill=(200, 40, 40))
    framed = ImageOps.expand(flower, border=24, fill="white")
    framed = ImageOps.expand(framed, border=12, fill="black")
    black_print = Image.new("RGB", (240, 240), (8, 8, 8))
    ImageDraw.Draw(black_print).ellipse((60, 60, 180, 180), fill=(180, 30, 40))
    ok_frame = triage(framed).verdict == VERDICT_FRAMED
    ok_black = triage(black_print).verdict == VERDICT_TILEABLE
    ok_plain = triage(flower).verdict == VERDICT_TILEABLE
    print(
        f"合成偵測  成品框={ok_frame}  黑底印花(應否)={ok_black}  "
        f"無框花布(應否)={ok_plain}"
    )
    if not (ok_frame and ok_black and ok_plain):
        print("分流合成對照失敗")
        return 1

    n_skip = 0
    n_err = 0
    shown_skip = 0
    shown_keep = 0
    for folder, name in pairs[:probe]:
        path = src / folder / name if folder else src / name
        try:
            with Image.open(path) as im:
                hit = triage(im)
        except Exception as exc:  # noqa: BLE001
            n_err += 1
            print(f"  開圖失敗 {folder}/{name}: {exc}")
            continue
        if hit.verdict != VERDICT_TILEABLE:
            n_skip += 1
            if shown_skip < 5:
                print(f"  SKIP {folder}/{name}  {hit.describe()}")
                shown_skip += 1
        elif shown_keep < 5:
            print(f"  KEEP {folder}/{name}")
            shown_keep += 1
    print(
        f"抽檢前 {min(probe, len(pairs))} 張：跳過 {n_skip}，"
        f"待跑 {min(probe, len(pairs)) - n_skip - n_err}，開圖失敗 {n_err}"
    )
    return 0


def main() -> int:
    global SRC, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(SRC), help="例圖根目錄（遞迴）")
    ap.add_argument(
        "--out",
        default=str(OUT),
        help="報告／審查牆輸出目錄",
    )
    ap.add_argument("--only", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--no-sheets", action="store_true")
    ap.add_argument("--tag", default="", help="報告檔名後綴，用來保留前後對照")
    ap.add_argument(
        "--inventory",
        action="store_true",
        help="只清點例圖並抽檢成品框，不跑無縫",
    )
    ap.add_argument(
        "--failed-from",
        default="",
        help="只重跑該 report.json 裡 errors 非空的圖",
    )
    ap.add_argument(
        "--pairs-from",
        default="",
        help="只跑該 JSON 裡的 {folder,name} 列表",
    )
    ap.add_argument(
        "--merge-into",
        default="",
        help="把本次結果寫回這份 report.json 並重寫審查牆",
    )
    ap.add_argument(
        "--no-dedup",
        action="store_true",
        help="不去重，每條路徑都跑",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="跳過 partial jsonl／既有報告裡已跑過的圖，並續寫檢查點",
    )
    args = ap.parse_args()
    _configure_stdio()
    _lower_priority()

    SRC = Path(args.src)
    OUT = Path(args.out)
    if not SRC.is_dir():
        print(f"找不到例圖目錄：{SRC}")
        return 1
    if args.inventory:
        return inventory(SRC)

    OUT.mkdir(parents=True, exist_ok=True)
    pairs = collect(SRC, args.only, args.limit)
    if args.failed_from:
        report = json.loads(Path(args.failed_from).read_text(encoding="utf-8"))
        wanted = {
            (r.get("folder") or "", r.get("name") or "")
            for r in report
            if r.get("errors")
        }
        pairs = [p for p in pairs if p in wanted]
        print(f"依報告重跑失敗集 {len(pairs)} 張（報告 {args.failed_from}）")
    if args.pairs_from:
        listed = json.loads(Path(args.pairs_from).read_text(encoding="utf-8"))
        wanted = {
            (r.get("folder") or "", r.get("name") or "")
            for r in listed
        }
        pairs = [p for p in pairs if p in wanted]
        print(f"依名單重跑 {len(pairs)} 張（{args.pairs_from}）")
    if not pairs:
        print("沒有符合的圖")
        return 1

    groups: dict[str, list[tuple[str, str]]] | None = None
    n_listed = len(pairs)
    if not args.no_dedup:
        unique, groups = dedup_pairs(pairs)
        print(f"去重 {n_listed} → {len(unique)} 張唯一內容")
        pairs = unique

    small = [(f, n) for f, n in pairs if _pixels(f, n) < HUGE_PIXELS]
    huge = [(f, n) for f, n in pairs if _pixels(f, n) >= HUGE_PIXELS]
    workers = args.workers or min(3, os.cpu_count() or 2)
    print(f"來源 {SRC}")
    print(f"輸出 {OUT}")
    print(f"共 {len(pairs)} 張（大圖 {len(huge)} 張另跑）| 行程 {workers}")

    rows: list[dict] = []
    t0 = time.perf_counter()
    done = 0
    total = len(pairs)

    def report(r: dict) -> None:
        nonlocal done
        done += 1
        key = f"{r['folder']}/{r['name']}" if r.get("folder") else r.get("name", "")
        if r.get("skipped"):
            print(f"[{done}/{total}] SKIP {key}  {r['skipped']}", flush=True)
            return
        if r.get("error"):
            print(f"[{done}/{total}] ERR  {key}: {r['error']}", flush=True)
            return
        tag = "FAIL" if r.get("errors") else "ok  "
        print(
            f"[{done}/{total}] {tag} {key:34s} "
            f"接縫 {r['src_wrap_excess']:6.1f}->{r['wrap_excess']:5.1f} "
            f"熱點 {r.get('src_hotspot', 0):5.0f}->{r.get('hotspot', 0):4.0f} "
            f"{r['elapsed_s']:5.1f}s"
            + (f"  {r['errors']}" if r.get("errors") else ""),
            flush=True,
        )

    suffix = f"_{args.tag}" if args.tag else ""
    report_path = OUT / f"report{suffix}.json"
    partial_path = OUT / f"report{suffix}.partial.jsonl"
    done_keys: set[tuple[str, str]] = set()
    if args.resume and partial_path.exists():
        for line in partial_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            done_keys.add((rec.get("folder") or "", rec.get("name") or ""))
        if done_keys:
            print(f"續跑：已有 {len(done_keys)} 張，略過", flush=True)
            done = len(done_keys)
    elif partial_path.exists():
        partial_path.unlink()

    sheets = not args.no_sheets
    init_args = (str(SRC.resolve()), str(OUT.resolve()), WORKER_AIM_S)
    partial_f = partial_path.open("a", encoding="utf-8")
    try:
        for group, n_proc in ((small, workers), (huge, 1)):
            if not group:
                continue
            jobs = [
                (f, n, sheets)
                for f, n in group
                if (f, n) not in done_keys
            ]
            if not jobs:
                continue
            def _on_result(r: dict) -> None:
                rows.append(r)
                partial_f.write(json.dumps(r, ensure_ascii=False) + "\n")
                partial_f.flush()
                report(r)

            _run_jobs_hard_timeout(jobs, n_proc, init_args, _on_result)
    finally:
        partial_f.close()

    if args.resume and done_keys:
        loaded: list[dict] = []
        if partial_path.exists():
            seen: set[tuple[str, str]] = set()
            for line in partial_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = (rec.get("folder") or "", rec.get("name") or "")
                if key in seen:
                    continue
                seen.add(key)
                loaded.append(rec)
        have = {(r.get("folder") or "", r.get("name") or "") for r in rows}
        for rec in loaded:
            key = (rec.get("folder") or "", rec.get("name") or "")
            if key not in have:
                rows.append(rec)

    if groups is not None:
        extras = replicate_duplicates(rows, groups)
        rows.extend(extras)
        if extras:
            print(f"重複內容套用結果 {len(extras)} 張", flush=True)

    elapsed = time.perf_counter() - t0
    suffix = f"_{args.tag}" if args.tag else ""
    report_path = OUT / f"report{suffix}.json"
    merge_old = None
    if args.merge_into:
        dest_pre = Path(args.merge_into)
        if dest_pre.exists():
            merge_old = json.loads(dest_pre.read_text(encoding="utf-8"))
    report_path.write_text(
        json.dumps(
            sorted(rows, key=lambda r: (r.get("folder") or "", r.get("name") or "")),
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )

    n_skip = sum(1 for r in rows if r.get("skipped"))
    n_fail = sum(1 for r in rows if r.get("errors"))
    n_unchanged = sum(1 for r in rows if r.get("unchanged"))
    n_dup = sum(1 for r in rows if r.get("duplicate_of"))
    n_run = len(rows) - n_skip
    print(f"\n{'=' * 70}")
    print(
        f"完成 {len(rows)} 張（跑 {n_run - n_dup}，"
        f"跳過 {n_skip}，重複 {n_dup}），耗時 {elapsed / 60:.1f} 分"
    )
    print(f"未通過 {n_fail}／原圖直出 {n_unchanged}")
    skip_reasons: dict[str, int] = {}
    for r in rows:
        if r.get("skipped") and not r.get("duplicate_of"):
            skip_reasons[str(r["skipped"])] = skip_reasons.get(str(r["skipped"]), 0) + 1
    for k, v in sorted(skip_reasons.items(), key=lambda kv: -kv[1]):
        print(f"  跳過 {k}: {v}")
    reasons: dict[str, int] = {}
    for r in rows:
        for e in r.get("errors") or []:
            reasons[e.split(":")[0]] = reasons.get(e.split(":")[0], 0) + 1
    for k, v in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"  {k}: {v}")
    print(f"報告 {report_path}")
    index_rows = rows
    if args.merge_into:
        dest = Path(args.merge_into)
        old = merge_old if merge_old is not None else json.loads(
            dest.read_text(encoding="utf-8")
        )
        by_key = {
            (r.get("folder") or "", r.get("name") or ""): i
            for i, r in enumerate(old)
        }
        updated = 0
        for r in rows:
            key = (r.get("folder") or "", r.get("name") or "")
            if r.get("duplicate_of"):
                continue
            idx = by_key.get(key)
            if idx is None:
                old.append(r)
                by_key[key] = len(old) - 1
            else:
                old[idx] = r
            updated += 1
            prim = f"{key[0]}/{key[1]}" if key[0] else key[1]
            for i, other in enumerate(old):
                if other.get("duplicate_of") == prim:
                    copy = dict(r)
                    copy["folder"] = other.get("folder")
                    copy["name"] = other.get("name")
                    copy["duplicate_of"] = prim
                    old[i] = copy
        dest.write_text(
            json.dumps(
                sorted(
                    old,
                    key=lambda r: (r.get("folder") or "", r.get("name") or ""),
                ),
                ensure_ascii=False,
                indent=1,
            ),
            encoding="utf-8",
        )
        print(f"已寫回 {updated} 張到 {dest}")
        index_rows = old
    if sheets:
        print(f"索引 {write_index(index_rows)}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    mp.freeze_support()
    _configure_stdio()
    raise SystemExit(main())
