# -*- coding: utf-8 -*-
"""回歸測試：96885088533 + 昨日目視高接縫樣本。"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.color_utils import detect_background
from app.discrete_lattice import (
    _correct_half_pitch,
    _dual_scale_groups,
    _fg_centroids,
    _fg_mask,
    _interior_median_motif_area,
    looks_like_regular_lattice,
    try_discrete_lattice_crop,
)
from app.processor import (
    _seam_rank,
    _tile_seam_scores,
    _to_rgb_array,
    make_seamless_hard_cut,
    tile_2x2_multi,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"
OUT = Path(__file__).resolve().parent / "_out"
OUT.mkdir(exist_ok=True)


def _run_one(path: Path) -> dict:
    img = Image.open(path)
    img.load()
    bg = detect_background(img)
    src = _to_rgb_array(img, bg)
    src_v, src_h = _tile_seam_scores(src)
    t0 = time.time()
    unit, mode = make_seamless_hard_cut(img, bg)
    elapsed = time.time() - t0
    uarr = _to_rgb_array(unit, bg)
    v, h = _tile_seam_scores(uarr)
    rank = _seam_rank(uarr)
    preview, detail, _ = tile_2x2_multi(unit)
    stem = path.stem[:60]
    unit.save(OUT / f"{stem}_unit.png")
    preview.save(OUT / f"{stem}_2x2.png")
    return {
        "name": path.name,
        "mode": mode,
        "src_seam": (round(src_v, 2), round(src_h, 2)),
        "unit_seam": (round(v, 2), round(h, 2)),
        "unit_size": unit.size,
        "seam_rank": round(rank, 2),
        "preview_detail": detail,
        "elapsed_s": round(elapsed, 2),
        "ok": True,
        "errors": [],
    }


def test_96885088533() -> list[dict]:
    """稀疏花枝／花環：禁止半週期晶格；單元接縫須優於原圖或足夠低。"""
    folder = FIXTURES / "96885088533"
    rows: list[dict] = []
    for path in sorted(folder.glob("*.png")):
        row = _run_one(path)
        errors: list[str] = []
        v, h = row["unit_seam"]
        sv, sh = row["src_seam"]
        # 清邊不得在接縫明顯更差時勝出
        if "清邊後補花" in row["mode"] and (v + h) > (sv + sh) * 0.95 + 1.0:
            errors.append("clear_edge_not_better")
        # 單元接縫總分：稀疏圖應壓到合理範圍
        if (v + h) > 12.0 and (v + h) > (sv + sh) * 0.7:
            errors.append(f"seam_too_high:{v}+{h}")
        if "FAIL" in row["mode"]:
            errors.append("mode_FAIL")
        # 花環圖：若走晶格，週期不得是未校正半週期（約 71×100）
        if "19_01_51 (3)" in path.name:
            img = Image.open(path)
            img.load()
            bg = detect_background(img)
            arr = _to_rgb_array(img, bg)
            if not looks_like_regular_lattice(arr, bg, 40.0):
                errors.append("wreath_not_regular")
            cropped, detail, _tier = try_discrete_lattice_crop(arr, bg, 40.0)
            if cropped is not None and "晶格週期" in detail:
                # 解析 px×py
                try:
                    part = detail.split("晶格週期", 1)[1].split("px", 1)[0].strip()
                    px_s, py_s = part.split("×")
                    px, py = int(px_s), int(py_s)
                    if px < 100 or py < 140:
                        errors.append(f"half_period_lattice:{px}x{py}")
                except Exception:
                    errors.append("period_parse_fail")
            # 最終輸出接縫必須明顯優於原圖（花圈磚縫允許中等殘差）
            if (v + h) > 8.0:
                errors.append(f"wreath_seam:{v}+{h}")
        row["errors"] = errors
        row["ok"] = len(errors) == 0
        rows.append(row)
        status = "OK" if row["ok"] else "FAIL"
        print(f"[{status}] {path.name} seam={v}+{h} rank={row['seam_rank']} | {row['mode'][:70]}")
        if errors:
            print(f"         errors={errors}")
    return rows


def test_temu_visual() -> list[dict]:
    """昨日目視偏高接縫樣本：至少要比原圖改善，且不得 mode FAIL。"""
    folder = FIXTURES / "temu_visual"
    rows: list[dict] = []
    for path in sorted(folder.glob("*.png")):
        row = _run_one(path)
        errors: list[str] = []
        v, h = row["unit_seam"]
        sv, sh = row["src_seam"]
        if "FAIL" in row["mode"]:
            errors.append("mode_FAIL")
        # 滿鋪應明顯改善；殘差允許中等（部分 Codex 原圖本身接縫極差）
        if (v + h) > (sv + sh) * 0.75:
            errors.append(f"not_improved:{sv}+{sh}->{v}+{h}")
        if (v + h) > 90 or max(v, h) > 60:
            errors.append(f"still_high:{v}+{h}")
        if row["seam_rank"] > 120:
            errors.append(f"rank_high:{row['seam_rank']}")
        row["errors"] = errors
        row["ok"] = len(errors) == 0
        rows.append(row)
        status = "OK" if row["ok"] else "FAIL"
        print(f"[{status}] {path.name} {sv}+{sh}->{v}+{h} | {row['mode'][:70]}")
        if errors:
            print(f"         errors={errors}")
    return rows


def test_prod_8_7() -> list[dict]:
    """8-7 生產圖代表性壞例：禁止接縫相對原圖變差、禁止過小晶格週期。"""
    folder = FIXTURES / "prod_8_7"
    rows: list[dict] = []
    if not folder.is_dir():
        print("[SKIP] no prod_8_7 fixtures")
        return rows
    for path in sorted(folder.glob("*.png")):
        row = _run_one(path)
        errors: list[str] = []
        v, h = row["unit_seam"]
        sv, sh = row["src_seam"]
        if "FAIL" in row["mode"]:
            errors.append("mode_FAIL")
        if (v + h) > (sv + sh) + 2.0:
            errors.append(f"worsened:{sv}+{sh}->{v}+{h}")
        m = re.search(r"晶格週期\s*(\d+)\s*[×x]\s*(\d+)", row["mode"])
        if m and (int(m.group(1)) < 64 or int(m.group(2)) < 64):
            errors.append(f"small_period:{m.group(1)}x{m.group(2)}")
        row["errors"] = errors
        row["ok"] = len(errors) == 0
        rows.append(row)
        status = "OK" if row["ok"] else "FAIL"
        print(f"[{status}] {path.name} {sv}+{sh}->{v}+{h} | {row['mode'][:70]}")
        if errors:
            print(f"         errors={errors}")
    return rows


def test_half_pitch_helper() -> None:
    """單元測試：最近鄰約 2× 軸向 pitch 時應加倍。"""
    # 磚縫格：軸向投影半週期 70，真實 NN≈140
    pts = []
    for row in range(6):
        for col in range(6):
            x = col * 140 + (70 if row % 2 else 0)
            y = row * 100
            pts.append((float(y), float(x)))
    px, py = _correct_half_pitch(70.0, 100.0, pts)
    assert px >= 130, f"expected ~140 got {px}"
    print(f"[OK] half_pitch_helper -> {px:.1f}x{py:.1f}")


def main() -> int:
    print("=== half pitch helper ===")
    test_half_pitch_helper()
    print("=== 96885088533 ===")
    rows_a = test_96885088533()
    print("=== temu visual ===")
    rows_b = test_temu_visual()
    print("=== prod_8_7 ===")
    rows_c = test_prod_8_7()
    all_rows = rows_a + rows_b + rows_c
    summary = {
        "total": len(all_rows),
        "ok": sum(1 for r in all_rows if r["ok"]),
        "fail": sum(1 for r in all_rows if not r["ok"]),
        "rows": all_rows,
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nSUMMARY {summary['ok']}/{summary['total']} OK")
    return 0 if summary["fail"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
