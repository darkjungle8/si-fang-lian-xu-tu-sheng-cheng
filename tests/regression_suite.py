# -*- coding: utf-8 -*-
"""四方連續回歸集：用客觀指標守住已修好的圖，避免改 A 壞 B。

## 舊版為什麼會全綠卻放行一堆破圖

舊斷言的第一條是「最差線不得比原稿更差」。原圖直出時輸出等於輸入，這條
永遠成立。當時 52 個案例裡有 45 個輸出與原圖完全相同、34 個仍留著肉眼
可見的縫，測試卻一路顯示通過——它量的是「有沒有變糟」，不是「有沒有做
到」。

## 現在的斷言

第一條改成**輸出必須無縫**，這是這個工具存在的理由，做不到就是失敗，
不能用「至少沒變糟」蒙混。其餘幾條防止為了無縫而付出不該付的代價。

用法：
    python tests/regression_suite.py --write-baseline   # 記錄現狀
    python tests/regression_suite.py                    # 比對基準
    python tests/regression_suite.py --reported-only    # 只跑回報過的圖
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.color_utils import detect_background
from app.processor import _to_rgb_array, make_seamless_hard_cut
from app.quality import (
    axis_line_energy,
    design_error,
    geometry_fidelity,
    seam_report,
    tone_shift,
)
from app.triage import VERDICT_TILEABLE, triage

SRC = ROOT / "samples" / "F"
BASELINE = Path(__file__).resolve().parent / "regression_baseline.json"
EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}

# 使用者實際回報過有問題的圖，以及當初的症狀。
# 症狀只作紀錄用，斷言一律走客觀指標。
REPORTED: list[tuple[str, str, str]] = [
    ("100图-1", "2 (25).jpg", "生成圖與原圖差異過大"),
    ("100图-1", "2 (29).jpg", "處理卡住"),
    ("100图-1", "2 (47).jpg", "質心滾動把圓點切成花生形"),
    ("100图-1", "2 (52).jpg", "生成圖有問題"),
    ("100图-1", "2.jpg", "假週期加半幅偏移，花被切成十字"),
    ("100图-1", "3 (24).jpg", "生成圖有問題"),
    ("100图-1", "3 (25).jpg", "生成圖有問題"),
    ("100图-1", "3 (27).jpg", "生成圖有問題"),
    ("100图-1", "3 (36).jpg", "點綴 FAIL 後仍半幅，圖案切到中央"),
    ("100图-1", "3 (43).jpg", "生成圖有問題"),
    ("100图-1", "3 (51).jpg", "晶格週期被誤估，邊緣切到字與狗"),
    ("100图-1", "3.jpg", "soft 混合讓罌粟出現鏡像縫"),
    ("100图-1", "4 (1).jpg", "點綴 FAIL 後仍半幅，圖案切到中央"),
    ("100图-2", "1 (10).png", "錯切對齊把拼布格扳斜"),
    ("100图-2", "1 (2).jpg", "CMYK 未走 ICC 導致條紋偏青；半幅使大象朝向錯亂"),
    ("100图-2", "1 (20).jpg", "假週期讓 2×2 中縫出現連體花"),
    ("100图-2", "1 (28).jpg", "強制半幅把菱形拼布中央切開"),
    ("100图-2", "1 (37).jpg", "多餘的邊緣柔化造成漿果粉霧"),
    ("100图-2", "1 (38).jpg", "CMYK 未走 ICC，藍底過飽和"),
    ("100图-2", "1 (46).jpg", "半幅把斜枝從畫面中央切斷"),
    ("100图-2", "1 (71).jpg", "白底恐龍被誤判密花，整張保留後從身體切開"),
    ("100图-2", "1.jpg", "黑底罌粟：假週期切穿花瓣"),
]

_SAMPLE_SEED = 20260817
_SAMPLE_N = 40

# 與 tests/sweep_all.py 共用同一組門檻
# 與 app.select.SEAM_OK 一致：取自整批稿件的目視校準，超出量 5 以下看不見
SEAM_MAX = 5.0
TONE_MAX = 4.0
# 這裡看不到候選是怎麼來的，只能擋災難級。細緻的把關在 `app.select`：
# 那裡分得出裁切類（真週期應該幾乎完全還原）與最小誤差切（切掉一條帶子
# 本來就對不回原圖的相位，數值天生偏高）。
DESIGN_MAX = 90.0


def _job_path(folder: str, name: str) -> Path:
    return (SRC / folder / name) if folder else SRC / name


def iter_images(src: Path):
    for p in sorted(src.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in EXTS:
            continue
        rel = p.relative_to(src)
        folder = "" if rel.parent == Path(".") else rel.parent.as_posix()
        yield folder, rel.name, p


def sampled_cases() -> list[tuple[str, str, str]]:
    """從資料夾隨機抽樣（固定 seed），排除已在回報清單裡的圖。"""
    reported = {(f, n) for f, n, _ in REPORTED}
    rng = random.Random(_SAMPLE_SEED)
    pool: list[tuple[str, str, str]] = []
    if not SRC.is_dir():
        return []
    for folder, name, _path in iter_images(SRC):
        if (folder, name) in reported:
            continue
        pool.append((folder, name, "抽樣"))
    if not pool:
        return []
    return rng.sample(pool, min(_SAMPLE_N, len(pool)))


def run_case(folder: str, name: str) -> dict:
    path = _job_path(folder, name)
    img = Image.open(path)
    img.load()
    decision = triage(img)
    if decision.verdict != VERDICT_TILEABLE:
        return {
            "folder": folder,
            "name": name,
            "skipped": decision.verdict,
            "mode": decision.describe(),
            "errors": [],
        }
    bg = detect_background(img)
    src = _to_rgb_array(img, bg)
    s_rep = seam_report(src)

    t0 = time.perf_counter()
    unit, mode = make_seamless_hard_cut(img, bg)
    elapsed = time.perf_counter() - t0

    out = _to_rgb_array(unit, bg)
    o_rep = seam_report(out)
    return {
        "folder": folder,
        "name": name,
        "mode": mode,
        "size": list(unit.size),
        "src_size": list(img.size),
        "src_mode": img.mode,
        "out_mode": unit.mode,
        "keeps_icc": bool(unit.info.get("icc_profile")),
        "src_wrap_excess": round(s_rep.wrap_excess, 2),
        "src_internal_excess": round(s_rep.internal_excess, 2),
        "src_axis_energy": round(axis_line_energy(src), 3),
        "wrap_excess": round(o_rep.wrap_excess, 2),
        "internal_excess": round(o_rep.internal_excess, 2),
        "fidelity": round(geometry_fidelity(src, out), 3),
        "tone_shift": round(tone_shift(src, out), 2),
        "design_error": round(design_error(src, out), 2),
        "elapsed_s": round(elapsed, 1),
    }


def check(row: dict) -> list[str]:
    """客觀斷言。回傳違規清單。"""
    errs: list[str] = []

    # 1. 這是整個工具存在的理由：輸出自己接自己時不能看得出縫。
    if row["wrap_excess"] > SEAM_MAX:
        errs.append(f"接縫未消:{row['wrap_excess']}")

    # 2. 不能把縫搬到單元內部（半幅滾動的老把戲），也不能切出新斷裂。
    #    與原稿比較：條紋壁紙的硬邊在原稿就有，不是我們造成的。
    allow = max(row["src_internal_excess"], 6.0) * 1.15 + 2.0
    if row["internal_excess"] > allow:
        errs.append(f"內部新增斷裂:{row['internal_excess']}>{allow:.1f}")

    # 3. 整體色調不得跑掉
    if row["tone_shift"] > TONE_MAX:
        errs.append(f"色調偏移:{row['tone_shift']}")

    # 4. 平鋪回去要還原得了原設計：抓假週期把花距拉成兩倍這類問題
    if row["design_error"] > DESIGN_MAX:
        errs.append(f"設計被改壞:{row['design_error']}")

    # 5. 印刷稿必須維持原色彩空間並保留 profile
    if row["src_mode"] == "CMYK" and row["out_mode"] != "CMYK":
        errs.append(f"色彩空間被降級:{row['out_mode']}")

    if "未達標" in row["mode"]:
        errs.append("mode_未達標")
    return errs


def main() -> int:
    global SRC
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-baseline", action="store_true")
    ap.add_argument("--reported-only", action="store_true")
    ap.add_argument("--src", default=str(SRC), help="例圖根目錄（遞迴）")
    ap.add_argument("--only", default="", help="只跑檔名含此字串的案例")
    args = ap.parse_args()

    SRC = Path(args.src)

    cases = list(REPORTED)
    if not args.reported_only:
        cases += sampled_cases()
    if args.only:
        cases = [c for c in cases if args.only in c[1]]

    base: dict[str, dict] = {}
    if BASELINE.exists() and not args.write_baseline:
        base = json.loads(BASELINE.read_text(encoding="utf-8"))

    rows: dict[str, dict] = {}
    n_fail = 0
    n_regress = 0
    for i, (folder, name, note) in enumerate(cases, 1):
        path = _job_path(folder, name)
        key = f"{folder}/{name}"
        if not path.exists():
            print(f"[{i}/{len(cases)}] SKIP {key}（不存在）")
            continue
        try:
            row = run_case(folder, name)
        except Exception as exc:  # noqa: BLE001
            print(f"[{i}/{len(cases)}] ERROR {key}: {exc}")
            rows[key] = {"error": str(exc), "note": note}
            n_fail += 1
            continue
        if row.get("skipped"):
            print(f"[{i}/{len(cases)}] SKIP {key}  {row['skipped']}")
            continue
        row["note"] = note
        errs = check(row)
        row["errors"] = errs
        rows[key] = row

        tag = "OK  " if not errs else "FAIL"
        line = (
            f"[{i}/{len(cases)}] {tag} {key:26s} "
            f"接縫 {row['src_wrap_excess']:6.1f}->{row['wrap_excess']:5.1f} "
            f"內部 {row['internal_excess']:6.1f} "
            f"幾何 {row['fidelity']:.2f} 還原 {row['design_error']:5.1f} "
            f"{row['elapsed_s']:5.1f}s"
        )
        prev = base.get(key)
        if prev and "wrap_excess" in prev:
            delta = row["wrap_excess"] - prev["wrap_excess"]
            if delta > 1.0:
                line += f"  ⚠ 較基準退化 +{delta:.1f}"
                n_regress += 1
            elif delta < -1.0:
                line += f"  改善 {delta:.1f}"
        print(line, flush=True)
        if errs:
            print(f"        {errs} | {row['mode'][:88]}", flush=True)
            n_fail += 1

    if args.write_baseline:
        BASELINE.write_text(
            json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        print(f"\n基準已寫入 {BASELINE}（{len(rows)} 案例）")
        return 0

    print(f"\n案例 {len(rows)}，違規 {n_fail}，較基準退化 {n_regress}")
    return 1 if (n_fail or n_regress) else 0


if __name__ == "__main__":
    raise SystemExit(main())
