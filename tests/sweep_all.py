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
import json
import multiprocessing as mp
import os
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
        f"還原 {row['design_error']:.0f}   色調 {row['tone_shift']:.1f}   "
        f"{row['src_mode']}→{row['out_mode']}"
        f"{'+ICC' if row['keeps_icc'] else ''}   "
        f"{'／'.join(row['errors']) if row['errors'] else row['mode'][:66]}"
    )
    d.text((pad, height - cap_h + 6), head, fill=(20, 20, 20), font=_font(15))
    d.text((pad, height - cap_h + 30), body, fill=(60, 60, 60), font=_font(13))
    return sheet


def check(row: dict) -> list[str]:
    """驗收條件。與回歸集共用同一套判準。"""
    errs: list[str] = []

    # 1. 這是全部的重點：輸出自己接自己時不能看得出縫。
    #    門檻與 `app.select.SEAM_OK` 一致，取自整批稿件的目視校準。
    if row["wrap_excess"] > 5.0:
        errs.append(f"接縫未消:{row['wrap_excess']:.1f}")

    # 2. 不能把縫搬到單元內部（半幅滾動的老把戲），也不能切出新斷裂。
    #    與原稿比較：條紋壁紙的硬邊在原稿就有，不算我們造成的。
    allow = max(row["src_internal_excess"], 6.0) * 1.15 + 2.0
    if row["internal_excess"] > allow:
        errs.append(f"內部新增斷裂:{row['internal_excess']:.1f}>{allow:.1f}")

    # 3. 整體色調不能跑掉
    if row["tone_shift"] > 4.0:
        errs.append(f"色調偏移:{row['tone_shift']:.1f}")

    # 4. 平鋪回去要還原得了原設計。這取代了「幾何保真」與「單元不能太小」
    #    兩個舊條件：抓到真週期的裁切就算只剩原圖百分之一大也完全還原，
    #    而假週期即使尺寸沒變也還原不了。
    #    這裡看不到候選是怎麼來的，只能擋災難級；細緻的把關在
    #    `app.select`，那裡分得出裁切類與最小誤差切（後者本來就對不回相位）。
    if row["design_error"] > 90.0:
        errs.append(f"設計被改壞:{row['design_error']:.0f}")

    # 5. 印刷稿必須維持原色彩空間
    if row["src_mode"] == "CMYK" and row["out_mode"] != "CMYK":
        errs.append(f"色彩空間被降級:{row['out_mode']}")

    if "未達標" in row["mode"]:
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


def run_case(job: tuple[str, str, bool]) -> dict:
    folder, name, write_sheet = job
    from app.color_utils import detect_background, has_finished_border
    from app.processor import _to_rgb_array, make_seamless_hard_cut
    from app.quality import (
        axis_line_energy,
        design_error,
        geometry_fidelity,
        seam_report,
        tone_shift,
    )

    path = _job_path(folder, name)
    row: dict = {"folder": folder, "name": name}
    try:
        img = Image.open(path)
        img.load()
        if has_finished_border(img):
            row["skipped"] = "finished_border"
            row["mode"] = "跳過：成品黑白邊"
            row["errors"] = []
            row["elapsed_s"] = 0.0
            return row

        bg = detect_background(img)
        src = _to_rgb_array(img, bg)
        s_rep = seam_report(src)

        t0 = time.perf_counter()
        unit, mode = make_seamless_hard_cut(img, bg)
        elapsed = time.perf_counter() - t0

        out = _to_rgb_array(unit, bg)
        o_rep = seam_report(out)

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
                "tone_shift": round(tone_shift(src, out), 2),
                "design_error": round(design_error(src, out), 2),
                "unchanged": bool(
                    src.shape == out.shape and np.array_equal(src, out)
                ),
                "elapsed_s": round(elapsed, 1),
            }
        )
        row["errors"] = check(row)

        if write_sheet:
            sheet = _contact_sheet(src, out, row)
            tag = "FAIL" if row["errors"] else "pass"
            dest = OUT / "sheets" / f"{tag}__{_sheet_name(folder, name)}.png"
            dest.parent.mkdir(parents=True, exist_ok=True)
            sheet.save(dest)
            row["sheet"] = str(dest.relative_to(OUT))
    except Exception as exc:  # noqa: BLE001 — 批次要繼續
        row["error"] = f"{exc}"
        row["trace"] = traceback.format_exc(limit=6)
        row["errors"] = ["EXCEPTION"]
    return row


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


def _init_worker(src: str, out: str) -> None:
    global SRC, OUT
    SRC = Path(src)
    OUT = Path(out)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    try:
        import cv2

        cv2.setNumThreads(1)
    except Exception:
        pass
    _lower_priority()


def write_index(rows: list[dict]) -> Path:
    """依嚴重度排序的 HTML 索引，方便一路往下看。"""

    def severity(r: dict) -> tuple:
        return (
            -len(r.get("errors") or []),
            -(r.get("wrap_excess") or 0),
            -(r.get("internal_excess") or 0),
        )

    visible = [r for r in rows if r.get("sheet")]
    ordered = sorted(visible, key=severity)
    n_fail = sum(1 for r in rows if r.get("errors"))
    n_skip = sum(1 for r in rows if r.get("skipped"))
    n_run = len(rows) - n_skip
    parts = [
        "<!doctype html><meta charset='utf-8'>",
        "<style>body{font:14px system-ui;background:#f6f6f6;margin:16px}"
        "img{max-width:100%;border:1px solid #ccc;background:#fff}"
        "div{margin-bottom:18px}</style>",
        f"<h2>掃描 {n_run} 張（跳過成品框 {n_skip}），未通過 {n_fail}</h2>",
    ]
    for r in ordered:
        parts.append(f"<div><img src='{r['sheet']}'></div>")
    dest = OUT / "index.html"
    dest.write_text("\n".join(parts), encoding="utf-8")
    return dest


def inventory(src: Path, probe: int = 80) -> int:
    """清點例圖並抽檢成品框偵測（含合成對照，避免誤殺黑底印花）。"""
    from PIL import ImageOps

    from app.color_utils import has_finished_border

    pairs = collect(src, "", 0)
    by_ext: dict[str, int] = {}
    for _folder, name in pairs:
        ext = Path(name).suffix.lower()
        by_ext[ext] = by_ext.get(ext, 0) + 1
    print(f"來源 {src}")
    print(f"共 {len(pairs)} 張  {by_ext}")

    # 合成：成品雙框應為 True；黑底印花（無白框）應為 False
    flower = Image.new("RGB", (240, 240), (40, 120, 80))
    for y in range(40, 200, 24):
        for x in range(40, 200, 24):
            ImageDraw.Draw(flower).ellipse((x, y, x + 14, y + 14), fill=(200, 40, 40))
    framed = ImageOps.expand(flower, border=24, fill="white")
    framed = ImageOps.expand(framed, border=12, fill="black")
    black_print = Image.new("RGB", (240, 240), (8, 8, 8))
    ImageDraw.Draw(black_print).ellipse((60, 60, 180, 180), fill=(180, 30, 40))
    ok_frame = has_finished_border(framed)
    ok_black = not has_finished_border(black_print)
    ok_plain = not has_finished_border(flower)
    print(
        f"合成偵測  成品框={ok_frame}  黑底印花(應否)={ok_black}  "
        f"無框花布(應否)={ok_plain}"
    )
    if not (ok_frame and ok_black and ok_plain):
        print("成品框偵測合成對照失敗")
        return 1

    n_skip = 0
    n_err = 0
    shown_skip = 0
    shown_keep = 0
    for folder, name in pairs[:probe]:
        path = src / folder / name if folder else src / name
        try:
            with Image.open(path) as im:
                hit = has_finished_border(im)
        except Exception as exc:  # noqa: BLE001
            n_err += 1
            print(f"  開圖失敗 {folder}/{name}: {exc}")
            continue
        if hit:
            n_skip += 1
            if shown_skip < 5:
                print(f"  SKIP {folder}/{name}")
                shown_skip += 1
        elif shown_keep < 5:
            print(f"  KEEP {folder}/{name}")
            shown_keep += 1
    print(
        f"抽檢前 {min(probe, len(pairs))} 張：成品框 {n_skip}，"
        f"待跑 {min(probe, len(pairs)) - n_skip - n_err}，開圖失敗 {n_err}"
    )
    return 0


def main() -> int:
    global SRC
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(SRC), help="例圖根目錄（遞迴）")
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
    args = ap.parse_args()
    _configure_stdio()
    _lower_priority()

    SRC = Path(args.src)
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
    if not pairs:
        print("沒有符合的圖")
        return 1

    small = [(f, n) for f, n in pairs if _pixels(f, n) < HUGE_PIXELS]
    huge = [(f, n) for f, n in pairs if _pixels(f, n) >= HUGE_PIXELS]
    workers = args.workers or min(3, os.cpu_count() or 2)
    print(f"來源 {SRC}")
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
            f"內部 {r['src_internal_excess']:6.1f}->{r['internal_excess']:6.1f} "
            f"幾何 {r['fidelity']:.2f} {r['elapsed_s']:5.1f}s"
            + (f"  {r['errors']}" if r.get("errors") else ""),
            flush=True,
        )

    sheets = not args.no_sheets
    init_args = (str(SRC.resolve()), str(OUT.resolve()))
    for group, n_proc in ((small, workers), (huge, 1)):
        if not group:
            continue
        jobs = [(f, n, sheets) for f, n in group]
        with mp.Pool(
            processes=n_proc,
            initializer=_init_worker,
            initargs=init_args,
            maxtasksperchild=1,
        ) as pool:
            for r in pool.imap_unordered(run_case, jobs, chunksize=1):
                rows.append(r)
                report(r)

    elapsed = time.perf_counter() - t0
    suffix = f"_{args.tag}" if args.tag else ""
    report_path = OUT / f"report{suffix}.json"
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
    n_run = len(rows) - n_skip
    print(f"\n{'=' * 70}")
    print(f"完成 {len(rows)} 張（跑 {n_run}，跳過成品框 {n_skip}），耗時 {elapsed / 60:.1f} 分")
    print(f"未通過 {n_fail}／原圖直出 {n_unchanged}")
    reasons: dict[str, int] = {}
    for r in rows:
        for e in r.get("errors") or []:
            reasons[e.split(":")[0]] = reasons.get(e.split(":")[0], 0) + 1
    for k, v in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"  {k}: {v}")
    print(f"報告 {report_path}")
    if sheets:
        print(f"索引 {write_index(rows)}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    mp.freeze_support()
    _configure_stdio()
    raise SystemExit(main())
