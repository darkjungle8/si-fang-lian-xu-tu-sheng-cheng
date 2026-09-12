# -*- coding: utf-8 -*-
"""全量稽核正在產出的印刷 TIFF：還原單元、套接縫閘門、寫 FAIL 對照圖。

成品是單元平鋪＋裁切＋白黑邊，不能直接對整張 TIFF 跑 wrap_excess。
去邊用 DPI×(白 0.2cm＋黑 0.1cm)；paste 週期用像素 MAE，不用自相關。

用法：
    python tests/audit_prod.py --once
    python tests/audit_prod.py --watch
    python tests/audit_prod.py --rerun-src --from-jsonl tests/_out/prod_audit/report.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("CV_THREADS", "1")

from PIL import Image

Image.MAX_IMAGE_PIXELS = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))

SRC_ROOT = Path(r"D:\5EDemocache\continue\out")
PINTU_ROOT = Path(r"D:\5EDemocache\continue\pintu\111")
PROD_ROOT = Path(r"D:\5EDemocache\continue\新增資料夾")
OUT = ROOT / "tests" / "_out" / "prod_audit"
WHITE_CM = 0.2
BLACK_CM = 0.1
CM_PER_INCH = 2.54
STABLE_S = 8.0
WATCH_POLL_S = 20.0


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def _lower_priority() -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), 0x00004000)
    except Exception:
        pass


def cm_to_px(cm: float, dpi: float) -> int:
    if cm <= 0:
        return 0
    return max(1, int(round(cm / CM_PER_INCH * dpi)))


_SRC_CACHE: dict[tuple[str, str], Path] = {}


def _pick_src(hits: list[Path], sku: str = "") -> Path | None:
    if not hits:
        return None
    if sku:
        same = [p for p in hits if p.parent.name == sku]
        if same:
            hits = same
    live = [p for p in hits if SRC_ROOT in p.parents]
    if live:
        return live[0]
    return hits[0]


def source_of(prod: Path) -> Path:
    rel = prod.resolve().relative_to(PROD_ROOT.resolve())
    if rel.suffix.lower() in {".tif", ".tiff"}:
        rel = rel.with_suffix("")
    src = SRC_ROOT / rel
    if src.is_file():
        return src
    alt = PINTU_ROOT / rel
    if alt.is_file():
        return alt
    series = rel.parts[0] if rel.parts else ""
    name = rel.name
    sku = rel.parts[-2] if len(rel.parts) >= 2 else ""
    key = (series, name)
    cached = _SRC_CACHE.get(key)
    if cached is not None:
        return cached
    # 先在同系列下找（SKU 編號常對不上），再整棵 out／pintu 依檔名找。
    # 成品系列名 DIY Play Studio、備份卻在 pintu/111/100 時必須靠後者。
    search_bases: list[Path] = []
    for root in (SRC_ROOT, PINTU_ROOT):
        if series:
            search_bases.append(root / series)
        search_bases.append(root)
    seen: set[str] = set()
    for base in search_bases:
        if not base.is_dir():
            continue
        mark = str(base.resolve()).lower()
        if mark in seen:
            continue
        seen.add(mark)
        hits = [p for p in base.rglob(name) if p.is_file()]
        chosen = _pick_src(hits, sku)
        if chosen is not None:
            _SRC_CACHE[key] = chosen
            return chosen
    _SRC_CACHE[key] = src
    return src


def design_key(prod: Path) -> str:
    rel = prod.resolve().relative_to(PROD_ROOT.resolve())
    parts = rel.parts
    name = rel.name
    if name.lower().endswith(".tif"):
        name = name[:-4]
    if len(parts) >= 3:
        return f"{parts[0]}/{parts[-2]}/{name}"
    return "/".join(parts)


def load_scored(jsonl: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not jsonl.is_file():
        return out
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = row.get("prod")
        if key:
            out[key] = row
    return out


def append_jsonl(jsonl: Path, row: dict) -> None:
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    with jsonl.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def is_stable(path: Path, wait_s: float = STABLE_S) -> bool:
    try:
        s1 = path.stat()
    except OSError:
        return False
    time.sleep(min(1.5, wait_s))
    try:
        s2 = path.stat()
    except OSError:
        return False
    if s1.st_size != s2.st_size or s1.st_mtime != s2.st_mtime:
        return False
    if (time.time() - s2.st_mtime) < wait_s:
        return False
    return s2.st_size > 0


def strip_border(arr, dpi: float) -> tuple:
    import numpy as np

    b = cm_to_px(WHITE_CM, dpi) + cm_to_px(BLACK_CM, dpi)
    h, w = arr.shape[:2]
    if min(h, w) <= 2 * b + 64:
        return arr, 0
    return arr[b:-b, b:-b], b


def find_period(inner, axis: int, tol: float = 0.5) -> tuple[int | None, float]:
    """paste 單元邊長：MAE 近 0 的最小位移，但必須大於花距／雜訊。"""
    import numpy as np

    a = inner if axis == 1 else np.swapaxes(inner, 0, 1)
    h, w = a.shape[:2]
    if w < 64:
        return None, 1e9
    min_p = max(64, min(h, w) // 10)
    row_step = max(1, h // 48)
    probe = a[::row_step, 0].astype(np.int16)
    max_p = w - max(32, min(w // 6, 80))
    sl = a[::row_step]
    for p in range(min_p, max_p + 1):
        mae0 = float(np.abs(a[::row_step, p].astype(np.int16) - probe).mean())
        if mae0 > tol:
            continue
        mae = float(np.abs(sl[:, :-p].astype(np.int16) - sl[:, p:]).mean())
        if mae <= tol:
            return p, mae
    return None, 1e9


def recover_unit(arr, dpi: float) -> tuple:
    import numpy as np

    inner, border = strip_border(arr, dpi)
    pw, mae_w = find_period(inner, 1)
    ph, mae_h = find_period(inner, 0)
    if not pw or not ph:
        return None, {
            "border": border,
            "period": [pw, ph],
            "period_mae": [mae_w, mae_h],
            "inner": [int(inner.shape[1]), int(inner.shape[0])],
        }
    unit = np.ascontiguousarray(inner[:ph, :pw])
    meta = {
        "border": border,
        "period": [int(pw), int(ph)],
        "period_mae": [round(mae_w, 4), round(mae_h, 4)],
        "inner": [int(inner.shape[1]), int(inner.shape[0])],
    }
    return unit, meta


def measure_pair(src_arr, unit_arr, *, mode: str = "") -> dict:
    import numpy as np
    from app.processor import _fine_grid_aligned, stamp_structure_view
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
        wrap_gutter_error,
        wrap_hotspot,
        wrap_orphan_run,
        wrap_period_remainder,
    )

    s_rep = seam_report(src_arr)
    o_rep = seam_report(unit_arr)
    src_view = stamp_structure_view(src_arr)
    out_view = stamp_structure_view(unit_arr)
    return {
        "mode": mode,
        "size": [int(unit_arr.shape[1]), int(unit_arr.shape[0])],
        "src_size": [int(src_arr.shape[1]), int(src_arr.shape[0])],
        "src_wrap": round(s_rep.wrap_raw, 2),
        "src_wrap_excess": round(s_rep.wrap_excess, 2),
        "src_internal_excess": round(s_rep.internal_excess, 2),
        "src_axis_energy": round(axis_line_energy(src_arr), 3),
        "wrap": round(o_rep.wrap_raw, 2),
        "wrap_excess": round(o_rep.wrap_excess, 2),
        "internal_excess": round(o_rep.internal_excess, 2),
        "internal_at": [
            round(o_rep.internal_at_v, 3),
            round(o_rep.internal_at_h, 3),
        ],
        "fidelity": round(geometry_fidelity(src_arr, unit_arr), 3),
        "tone_shift": round(
            tone_shift(
                src_arr,
                unit_arr,
                edge_frac=0.20 if "清邊補花" in mode else 0.0,
            ),
            2,
        ),
        "design_error": round(design_error(src_arr, unit_arr), 2),
        "src_hotspot": round(wrap_hotspot(src_view), 2),
        "hotspot": round(wrap_hotspot(out_view), 2),
        "src_void": round(edge_void_ratio(src_arr), 3),
        "void": round(edge_void_ratio(unit_arr), 3),
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
        "src_period_rem": round(wrap_period_remainder(src_arr), 3),
        "period_rem": round(wrap_period_remainder(unit_arr), 3),
        "fine_aligned": bool(_fine_grid_aligned(unit_arr)),
        "unchanged": bool(
            src_arr.shape == unit_arr.shape and np.array_equal(src_arr, unit_arr)
        ),
        "ink": round(ink_frac(unit_arr), 3),
    }


def check_recovered(row: dict) -> list[str]:
    """成品 TIFF 沒有策略標籤。

    尺寸變了不代表一定是週期裁切（清邊補花／最小誤差切也會改尺寸），
    不能拿 crop 的 period_rem／熱點門檻去打。原圖直出才用完整 sweep 閘門；
    其餘只擋色差縫、內部新斷裂，以及稀疏圖章的切圖／過密。
    """
    import sweep_all
    from app.select import (
        GUTTER_ERR_MAX,
        ORPHAN_RUN_MAX,
        WRAP_CUT_MAX,
        WRAP_DENSITY_MAX,
    )

    ink = float(row.get("ink") or 0)
    if row.get("unchanged"):
        row["mode"] = f"原圖直出｜前景 {ink:.0%}"
        return sweep_all.check(row)

    errs: list[str] = []
    if float(row.get("wrap_excess") or 0) > 5.0:
        errs.append(f"接縫未消:{row['wrap_excess']:.1f}")
    allow = max(float(row.get("src_internal_excess") or 0), 6.0) * 1.15 + 2.0
    if float(row.get("internal_excess") or 0) > allow:
        errs.append(
            f"內部新增斷裂:{row['internal_excess']:.1f}>{allow:.1f}"
        )
    if (
        ink < 0.42
        and float(row.get("wrap_cut") or 0) > WRAP_CUT_MAX
        and float(row.get("hotspot") or 0) > 14.0
    ):
        errs.append(f"接縫切圖:{row['wrap_cut']:.0%}")
    if ink < 0.42 and float(row.get("wrap_density") or 0) > WRAP_DENSITY_MAX:
        if row.get("unchanged"):
            errs.append(f"接縫過密:{row['wrap_density']:.2f}")
    if int(row.get("orphan") or 0) > ORPHAN_RUN_MAX:
        errs.append(f"接縫殘片:{row['orphan']}px")
    if float(row.get("gutter") or 0) > GUTTER_ERR_MAX and float(
        row.get("src_gutter") or 0
    ) <= GUTTER_ERR_MAX:
        errs.append(f"接縫錯格:{row['gutter']:.0%}")
    row["mode"] = row.get("mode") or f"加工｜前景 {ink:.0%}"
    return errs


def _to_rgb(img: Image.Image):
    from app.color_utils import detect_background
    from app.processor import _to_rgb_array

    bg = detect_background(img)
    return _to_rgb_array(img, bg), bg


def audit_tiff(prod: Path) -> dict:
    import sweep_all

    src_path = source_of(prod)
    rel = str(prod.resolve().relative_to(PROD_ROOT.resolve()))
    row: dict = {
        "prod": str(prod),
        "rel": rel.replace("\\", "/"),
        "src": str(src_path),
        "design": design_key(prod),
        "kind": "tiff",
        "folder": str(Path(rel).parent).replace("\\", "/"),
        "name": prod.name,
        "elapsed_s": 0.0,
    }
    t0 = time.perf_counter()
    try:
        with Image.open(prod) as im:
            im.load()
            dpi = float((im.info.get("dpi") or (300.0, 300.0))[0] or 300.0)
            prod_mode = im.mode
            arr = __import__("numpy").asarray(
                im.convert("RGB") if im.mode != "RGB" else im
            )
        row["dpi"] = dpi
        row["out_mode"] = prod_mode
        unit, meta = recover_unit(arr, dpi)
        row.update(meta)
        if unit is None:
            row["errors"] = ["無法還原單元"]
            row["elapsed_s"] = round(time.perf_counter() - t0, 1)
            return row
        if not src_path.is_file():
            row["errors"] = ["找不到原圖"]
            row["elapsed_s"] = round(time.perf_counter() - t0, 1)
            return row
        with Image.open(src_path) as sim:
            sim.load()
            src_arr, _bg = _to_rgb(sim)
            row["src_mode"] = sim.mode
            row["src_size"] = list(sim.size)
        measured = measure_pair(src_arr, unit, mode="")
        row.update(measured)
        row["keeps_icc"] = False
        row["errors"] = check_recovered(row)
        row["elapsed_s"] = round(time.perf_counter() - t0, 1)
        if row["errors"]:
            sheet = sweep_all._contact_sheet(src_arr, unit, row)
            dest = OUT / "sheets" / f"FAIL__{row['design'].replace('/', '__')}.png"
            dest.parent.mkdir(parents=True, exist_ok=True)
            sheet.save(dest)
            row["sheet"] = str(dest.relative_to(OUT))
            tiled = Image.fromarray(sweep_all.plain_tile_2x2(unit))
            thumb = OUT / "wall" / f"FAIL__{row['design'].replace('/', '__')}.jpg"
            sweep_all._write_wall_thumb(tiled, thumb)
            row["wall"] = str(thumb.relative_to(OUT))
    except Exception as exc:  # noqa: BLE001
        row["error"] = f"{exc}"
        row["trace"] = traceback.format_exc(limit=6)
        row["errors"] = ["EXCEPTION"]
        row["elapsed_s"] = round(time.perf_counter() - t0, 1)
    return row


def audit_source(src_path: Path, *, write_sheet: bool = True) -> dict:
    import numpy as np
    import sweep_all
    from app.color_utils import detect_background
    from app.processor import _to_rgb_array, make_seamless_hard_cut
    from app.triage import VERDICT_TILEABLE, triage

    rel = (
        src_path.resolve().relative_to(SRC_ROOT.resolve())
        if SRC_ROOT in src_path.resolve().parents or src_path.resolve() == SRC_ROOT
        else Path(src_path.name)
    )
    try:
        rel = src_path.resolve().relative_to(SRC_ROOT.resolve())
    except ValueError:
        rel = Path(src_path.name)
    folder = "" if rel.parent == Path(".") else rel.parent.as_posix()
    row: dict = {
        "kind": "rerun",
        "src": str(src_path),
        "folder": folder,
        "name": src_path.name,
        "design": f"{rel.parts[0]}/{rel.parts[-2]}/{src_path.name}"
        if len(rel.parts) >= 3
        else rel.as_posix(),
    }
    t0 = time.perf_counter()
    try:
        img = Image.open(src_path)
        img.load()
        decision = triage(img)
        row["triage"] = decision.verdict
        if decision.verdict != VERDICT_TILEABLE:
            row["skipped"] = decision.verdict
            row["mode"] = decision.describe()
            row["errors"] = []
            row["elapsed_s"] = 0.0
            return row
        bg = detect_background(img)
        src = _to_rgb_array(img, bg)
        unit, mode = make_seamless_hard_cut(img, bg)
        out = _to_rgb_array(unit, bg)
        row.update(measure_pair(src, out, mode=mode))
        row["src_mode"] = img.mode
        row["out_mode"] = unit.mode
        row["keeps_icc"] = bool(unit.info.get("icc_profile"))
        row["elapsed_s"] = round(time.perf_counter() - t0, 1)
        row["errors"] = sweep_all.check(row)
        if write_sheet:
            sheet = sweep_all._contact_sheet(src, out, row)
            tag = "FAIL" if row["errors"] else "pass"
            dest = OUT / "rerun" / f"{tag}__{row['design'].replace('/', '__')}.png"
            dest.parent.mkdir(parents=True, exist_ok=True)
            sheet.save(dest)
            row["sheet"] = str(dest.relative_to(OUT))
    except Exception as exc:  # noqa: BLE001
        row["error"] = f"{exc}"
        row["trace"] = traceback.format_exc(limit=8)
        row["errors"] = ["EXCEPTION"]
        row["elapsed_s"] = round(time.perf_counter() - t0, 1)
    return row


def collect_tiffs() -> list[Path]:
    if not PROD_ROOT.is_dir():
        return []
    return sorted(
        p
        for p in PROD_ROOT.rglob("*")
        if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}
    )


def _print_row(row: dict) -> None:
    errs = row.get("errors") or []
    tag = "FAIL" if errs else "pass"
    extra = "／".join(errs) if errs else (row.get("mode") or "")
    print(
        f"{tag:4} {row.get('design') or row.get('rel')}  "
        f"unit={row.get('size')}  wrap_ex={row.get('wrap_excess')}  "
        f"hot={row.get('hotspot')}  cut={row.get('wrap_cut')}  "
        f"dens={row.get('wrap_density')}  orphan={row.get('orphan')}  "
        f"{extra}",
        flush=True,
    )


def scan_new_tiffs(
    jsonl: Path, scored: dict[str, dict], *, force: bool = False
) -> list[dict]:
    rows: list[dict] = []
    for prod in collect_tiffs():
        key = str(prod)
        prev = scored.get(key)
        try:
            mtime = prod.stat().st_mtime
        except OSError:
            continue
        if (not force) and prev and prev.get("mtime") == mtime:
            continue
        if not is_stable(prod):
            print(f"skip writing {prod.name}", flush=True)
            continue
        print(f"audit {prod.relative_to(PROD_ROOT)}", flush=True)
        row = audit_tiff(prod)
        row["mtime"] = mtime
        append_jsonl(jsonl, row)
        scored[key] = row
        _print_row(row)
        rows.append(row)
    return rows


def summarize(scored: dict[str, dict]) -> None:
    tiff_rows = [r for r in scored.values() if r.get("kind") == "tiff"]
    n = len(tiff_rows)
    fails = [r for r in tiff_rows if r.get("errors")]
    print(
        f"\n已查 TIFF {n}  通過 {n - len(fails)}  失敗 {len(fails)}",
        flush=True,
    )
    from collections import Counter

    kinds: Counter[str] = Counter()
    for r in fails:
        for e in r.get("errors") or []:
            kinds[e.split(":")[0]] += 1
    if kinds:
        print("失敗類型", dict(kinds), flush=True)
        for r in fails:
            print(f"  FAIL {r.get('rel')}  {r.get('errors')}", flush=True)


def main() -> int:
    _configure_stdio()
    _lower_priority()
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--rerun-src", action="store_true")
    parser.add_argument("--only-fails", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--from-jsonl", type=Path, default=OUT / "report.jsonl")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    jsonl = args.from_jsonl
    scored = load_scored(jsonl)

    if args.rerun_src:
        seen: set[str] = set()
        jobs: list[tuple[Path, str]] = []
        for row in scored.values():
            if row.get("kind") != "tiff":
                continue
            if args.only_fails and not row.get("errors"):
                continue
            src = Path(row.get("src") or "")
            if not src.is_file():
                prod = Path(row.get("prod") or "")
                if prod.is_file():
                    src = source_of(prod)
            key = str(row.get("design") or src)
            ident = src.name
            if not src.is_file() or key in seen or ident in seen:
                continue
            seen.add(key)
            seen.add(ident)
            jobs.append((src, key))
        rerun_log = OUT / "rerun.jsonl"
        done: set[str] = set()
        if rerun_log.is_file():
            for line in rerun_log.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                prev = json.loads(line)
                if prev.get("errors"):
                    continue
                if prev.get("design"):
                    done.add(prev["design"])
                if prev.get("name"):
                    done.add(prev["name"])
        jobs = [
            (src, key)
            for src, key in jobs
            if key not in done and src.name not in done
        ]
        print(f"重跑原圖 {len(jobs)} 張（已有 {len(done)}）", flush=True)
        for src, key in jobs:
            print(f"rerun {src}", flush=True)
            row = audit_source(src)
            row["design"] = row.get("design") or key
            append_jsonl(rerun_log, row)
            _print_row(row)
        return 0

    if not args.watch or args.once:
        scan_new_tiffs(jsonl, scored, force=args.force)
        summarize(scored)
        if not args.watch:
            return 0

    print("watch", PROD_ROOT, flush=True)
    while True:
        new_rows = scan_new_tiffs(jsonl, scored, force=False)
        if new_rows:
            summarize({r["prod"]: r for r in new_rows if r.get("prod")})
        time.sleep(WATCH_POLL_S)


if __name__ == "__main__":
    raise SystemExit(main())
