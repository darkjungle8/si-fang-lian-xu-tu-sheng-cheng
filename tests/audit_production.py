# -*- coding: utf-8 -*-
"""生產圖步驟 1 全量審計：接縫指標 + 異常樣本落盤（支援多進程）。"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_INPUT = Path(r"D:\BaiduNetdiskDownload\8-7生产图")
DEFAULT_OUT = Path(__file__).resolve().parent / "_prod_audit_8-7"
EXCLUDE_DIR_NAME = "_excluded"


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:
            pass


def _collect_images(input_dir: Path, exclude_list: set[str]) -> list[Path]:
    files: list[Path] = []
    for p in sorted(input_dir.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if EXCLUDE_DIR_NAME in p.parts:
            continue
        if p.name.lower() in {"thumbs.db", ".ds_store"}:
            continue
        rel = str(p.resolve())
        if rel in exclude_list or p.name in exclude_list:
            continue
        files.append(p)
    return files


def _load_exclude(path: Path | None) -> set[str]:
    if path is None or not path.is_file():
        return set()
    out: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        out.add(s)
    return out


def _load_flagged_paths(summary_jsonl: Path) -> set[str]:
    flagged: set[str] = set()
    if not summary_jsonl.is_file():
        return flagged
    for line in summary_jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("flagged"):
            flagged.add(row["path"])
    return flagged


def _safe_stem(path: Path, input_root: Path) -> str:
    try:
        rel = path.relative_to(input_root)
    except ValueError:
        rel = Path(path.name)
    raw = "_".join(rel.parts)
    raw = re.sub(r"[^\w.\-]+", "_", raw, flags=re.UNICODE)
    return raw[:180]


def _flag_reasons(row: dict) -> list[str]:
    reasons: list[str] = []
    if row.get("error"):
        reasons.append(f"exception:{row['error'][:120]}")
        return reasons

    mode = row.get("mode") or ""
    if "FAIL" in mode:
        reasons.append("mode_FAIL")

    src_v, src_h = row.get("src_seam") or (0.0, 0.0)
    unit_v, unit_h = row.get("unit_seam") or (0.0, 0.0)
    src_sum = float(src_v) + float(src_h)
    unit_sum = float(unit_v) + float(unit_h)
    rank = float(row.get("seam_rank") or 0.0)

    w, h = row.get("unit_size") or (0, 0)
    if min(int(w), int(h)) < 32:
        reasons.append(f"tiny_unit:{w}x{h}")

    sw, sh = row.get("src_size") or (0, 0)
    if min(int(sw), int(sh)) < 64:
        reasons.append(f"tiny_src:{sw}x{sh}")

    keep_orig = "保留原圖" in mode
    # 保留原圖：演算法已選最佳（原圖），不當「未改善」假陽性
    # 僅在接縫極高時標記，供人工決定是否排除源圖
    if keep_orig:
        if unit_sum > 50 or max(float(unit_v), float(unit_h)) > 40:
            reasons.append(f"source_seam_high:{unit_sum:.1f}")
    else:
        # 僅在接縫未實質下降時標記（已改善的不算）
        if unit_sum > 12.0 and unit_sum >= src_sum * 0.95:
            # 忽略 <2 的微小波動
            if unit_sum >= src_sum + 2.0 or unit_sum >= src_sum * 1.02:
                reasons.append(f"not_improved:{src_sum:.1f}->{unit_sum:.1f}")
    if unit_sum > 90 or max(float(unit_v), float(unit_h)) > 60:
        reasons.append(f"still_high:{unit_v}+{unit_h}")
    if rank > 120:
        reasons.append(f"rank_high:{rank:.1f}")

    m = re.search(r"晶格週期\s*(\d+)\s*[×x]\s*(\d+)", mode)
    if m:
        px, py = int(m.group(1)), int(m.group(2))
        if px < 80 or py < 80:
            reasons.append(f"small_period:{px}x{py}")

    return reasons


def _maybe_downscale(img: Image.Image, max_side: int) -> tuple[Image.Image, float]:
    if max_side <= 0:
        return img, 1.0
    w, h = img.size
    side = max(w, h)
    if side <= max_side:
        return img, 1.0
    scale = max_side / float(side)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    return img.resize((nw, nh), Image.Resampling.LANCZOS), scale


def run_one_job(job: dict) -> dict:
    """子進程入口：job = {path, out_dir, input_root, save_all, max_side}。"""
    # 延遲 import，確保子進程 path 正確
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from app.color_utils import detect_background
    from app.processor import (
        _seam_rank,
        _tile_seam_scores,
        _to_rgb_array,
        make_seamless_hard_cut,
        tile_2x2_multi,
    )

    path = Path(job["path"])
    out_dir = Path(job["out_dir"])
    input_root = Path(job["input_root"])
    save_all = bool(job["save_all"])
    max_side = int(job.get("max_side") or 0)

    t0 = time.time()
    row: dict = {
        "path": str(path.resolve()),
        "name": path.name,
        "rel": str(path.relative_to(input_root)) if path.is_relative_to(input_root) else path.name,
        "ok": False,
        "flagged": False,
        "reasons": [],
        "error": None,
        "mode": "",
        "src_seam": None,
        "unit_seam": None,
        "seam_rank": None,
        "src_size": None,
        "proc_size": None,
        "unit_size": None,
        "scale": 1.0,
        "preview_detail": "",
        "elapsed_s": 0.0,
    }
    try:
        img = Image.open(path)
        img.load()
        row["src_size"] = list(img.size)
        img, scale = _maybe_downscale(img, max_side)
        row["scale"] = round(scale, 4)
        row["proc_size"] = list(img.size)

        bg = detect_background(img)
        src = _to_rgb_array(img, bg)
        src_v, src_h = _tile_seam_scores(src)
        row["src_seam"] = [round(src_v, 2), round(src_h, 2)]

        unit, mode = make_seamless_hard_cut(img, bg)
        uarr = _to_rgb_array(unit, bg)
        v, h = _tile_seam_scores(uarr)
        rank = _seam_rank(uarr)
        preview, detail, _ = tile_2x2_multi(unit)

        row["mode"] = mode
        row["unit_seam"] = [round(v, 2), round(h, 2)]
        row["seam_rank"] = round(rank, 2)
        row["unit_size"] = list(unit.size)
        row["preview_detail"] = detail
        row["elapsed_s"] = round(time.time() - t0, 2)

        reasons = _flag_reasons(row)
        row["reasons"] = reasons
        row["flagged"] = len(reasons) > 0
        row["ok"] = not row["flagged"]

        if save_all or row["flagged"]:
            stem = _safe_stem(path, input_root)
            sub = out_dir / ("flagged" if row["flagged"] else "ok")
            sub.mkdir(parents=True, exist_ok=True)
            unit.save(sub / f"{stem}_unit.png")
            preview.save(sub / f"{stem}_2x2.png")
            (sub / f"{stem}_meta.json").write_text(
                json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8"
            )
    except Exception as exc:  # noqa: BLE001
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["reasons"] = _flag_reasons(row)
        row["flagged"] = True
        row["ok"] = False
        row["elapsed_s"] = round(time.time() - t0, 2)
        row["traceback"] = traceback.format_exc()
    return row


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "ok",
        "flagged",
        "rel",
        "name",
        "mode",
        "src_seam",
        "unit_seam",
        "seam_rank",
        "src_size",
        "proc_size",
        "unit_size",
        "scale",
        "elapsed_s",
        "reasons",
        "error",
        "path",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            flat = dict(r)
            for k in ("src_seam", "unit_seam", "src_size", "proc_size", "unit_size"):
                flat[k] = json.dumps(r.get(k), ensure_ascii=False)
            flat["reasons"] = "|".join(r.get("reasons") or [])
            w.writerow(flat)


def _finalize(out_dir: Path, rows: list[dict]) -> dict:
    csv_path = out_dir / "summary.csv"
    flagged_path = out_dir / "flagged.txt"
    _write_csv(csv_path, rows)
    flagged_rows = [r for r in rows if r.get("flagged")]
    flagged_path.write_text(
        "\n".join(r["path"] for r in flagged_rows) + ("\n" if flagged_rows else ""),
        encoding="utf-8",
    )
    overview = {
        "total": len(rows),
        "ok": sum(1 for r in rows if r.get("ok")),
        "flagged": len(flagged_rows),
        "errors": sum(1 for r in rows if r.get("error")),
        "input": None,
        "out": str(out_dir),
    }
    (out_dir / "overview.json").write_text(
        json.dumps(overview, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return overview


def main(argv: list[str] | None = None) -> int:
    _configure_stdio()
    ap = argparse.ArgumentParser(description="生產圖步驟 1 審計")
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--exclude-list", type=Path, default=None)
    ap.add_argument("--only-flagged", action="store_true")
    ap.add_argument("--save-all", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--workers", type=int, default=max(1, min(6, (os.cpu_count() or 4))))
    ap.add_argument(
        "--max-side",
        type=int,
        default=1600,
        help="處理前最長邊上限（加速；0=不縮放）。預設 1600",
    )
    args = ap.parse_args(argv)

    input_root = args.input.resolve()
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "summary.jsonl"
    exclude_path = args.exclude_list or (out_dir / "exclude.txt")

    exclude = _load_exclude(exclude_path if exclude_path.is_file() else None)
    files = _collect_images(input_root, exclude)

    done_paths: set[str] = set()
    existing_rows: list[dict] = []
    # resume / only-flagged 都需要讀既有 summary，避免覆寫時丟掉通過樣本
    if (args.resume or args.only_flagged) and jsonl_path.is_file() and jsonl_path.stat().st_size > 0:
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            existing_rows.append(row)
            done_paths.add(row["path"])
    elif args.only_flagged:
        # jsonl 已被清空時，嘗試從 csv / backup 恢復
        bak = out_dir / "summary_ok_backup.jsonl"
        csv_path_existing = out_dir / "summary.csv"
        if bak.is_file():
            for line in bak.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                existing_rows.append(row)
                done_paths.add(row["path"])
        elif csv_path_existing.is_file():
            import csv as _csv

            with csv_path_existing.open(encoding="utf-8-sig", newline="") as fh:
                for r in _csv.DictReader(fh):
                    if r.get("ok") not in ("True", "true", "1"):
                        continue
                    row = {
                        "ok": True,
                        "flagged": False,
                        "rel": r.get("rel"),
                        "name": r.get("name"),
                        "mode": r.get("mode"),
                        "src_seam": json.loads(r["src_seam"]) if r.get("src_seam") else None,
                        "unit_seam": json.loads(r["unit_seam"]) if r.get("unit_seam") else None,
                        "seam_rank": float(r["seam_rank"]) if r.get("seam_rank") else None,
                        "src_size": json.loads(r["src_size"]) if r.get("src_size") else None,
                        "unit_size": json.loads(r["unit_size"]) if r.get("unit_size") else None,
                        "elapsed_s": float(r["elapsed_s"]) if r.get("elapsed_s") else 0.0,
                        "reasons": [],
                        "error": None,
                        "path": r.get("path"),
                    }
                    existing_rows.append(row)
                    done_paths.add(row["path"])

    flagged_targets: set[str] = set()
    if args.only_flagged:
        # 優先讀 flagged.txt（不受 jsonl 被覆寫影響）
        ft = out_dir / "flagged.txt"
        if ft.is_file():
            flagged_targets = {ln.strip() for ln in ft.read_text(encoding="utf-8").splitlines() if ln.strip()}
        if not flagged_targets:
            flagged_targets = _load_flagged_paths(jsonl_path)
        files = [p for p in files if str(p.resolve()) in flagged_targets]
        existing_rows = [r for r in existing_rows if r.get("path") not in flagged_targets]
        done_paths -= flagged_targets

    if args.limit and args.limit > 0:
        files = files[: args.limit]

    pending = [p for p in files if str(p.resolve()) not in done_paths]
    total = len(pending)
    workers = max(1, int(args.workers))

    print(f"輸入：{input_root}", flush=True)
    print(f"輸出：{out_dir}", flush=True)
    print(
        f"待處理：{total}（已完成跳過 {len(done_paths)}，排除 {len(exclude)}）"
        f" workers={workers} max_side={args.max_side}",
        flush=True,
    )

    rows: list[dict] = list(existing_rows)
    jobs = [
        {
            "path": str(p.resolve()),
            "out_dir": str(out_dir),
            "input_root": str(input_root),
            "save_all": bool(args.save_all),
            "max_side": int(args.max_side),
        }
        for p in pending
    ]

    # 重寫或 append：並行時統一最後重寫 jsonl，過程中增量 append
    with jsonl_path.open("a" if (args.resume and existing_rows and not args.only_flagged) else "w", encoding="utf-8") as jf:
        if args.only_flagged:
            for r in existing_rows:
                jf.write(json.dumps(r, ensure_ascii=False) + "\n")
            jf.flush()

        if total == 0:
            print("無待處理項目", flush=True)
        elif workers == 1:
            for i, job in enumerate(jobs, 1):
                print(f"[{i}/{total}] {Path(job['path']).name}", flush=True)
                row = run_one_job(job)
                rows.append(row)
                jf.write(json.dumps(row, ensure_ascii=False) + "\n")
                jf.flush()
                status = "FLAG" if row["flagged"] else "OK"
                print(
                    f"  -> {status} seam={row.get('unit_seam')} rank={row.get('seam_rank')} "
                    f"{row.get('elapsed_s')}s | {(row.get('mode') or row.get('error') or '')[:80]}",
                    flush=True,
                )
                if row["reasons"]:
                    print(f"     reasons={row['reasons']}", flush=True)
        else:
            done_n = 0
            with ProcessPoolExecutor(max_workers=workers) as ex:
                fut_map = {ex.submit(run_one_job, job): job for job in jobs}
                for fut in as_completed(fut_map):
                    done_n += 1
                    job = fut_map[fut]
                    try:
                        row = fut.result()
                    except Exception as exc:  # noqa: BLE001
                        row = {
                            "path": job["path"],
                            "name": Path(job["path"]).name,
                            "rel": job["path"],
                            "ok": False,
                            "flagged": True,
                            "reasons": [f"exception:{exc}"],
                            "error": f"{type(exc).__name__}: {exc}",
                            "mode": "",
                            "elapsed_s": 0.0,
                        }
                    rows.append(row)
                    jf.write(json.dumps(row, ensure_ascii=False) + "\n")
                    jf.flush()
                    status = "FLAG" if row["flagged"] else "OK"
                    print(
                        f"[{done_n}/{total}] {status} {row.get('rel') or row.get('name')} "
                        f"seam={row.get('unit_seam')} {row.get('elapsed_s')}s "
                        f"| {(row.get('mode') or row.get('error') or '')[:70]}",
                        flush=True,
                    )
                    if row.get("reasons"):
                        print(f"     reasons={row['reasons']}", flush=True)

    # 並行完成順序亂，重寫排序後的 jsonl
    rows_sorted = sorted(rows, key=lambda r: r.get("rel") or r.get("path") or "")
    jsonl_path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows_sorted),
        encoding="utf-8",
    )
    overview = _finalize(out_dir, rows_sorted)
    overview["input"] = str(input_root)
    (out_dir / "overview.json").write_text(
        json.dumps(overview, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"\nOVERVIEW ok={overview['ok']} flagged={overview['flagged']} total={overview['total']}",
        flush=True,
    )
    return 0 if overview["flagged"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
