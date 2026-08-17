# -*- coding: utf-8 -*-
"""全量掃描：把整批圖跑過一次，產生可目視審查的對照圖與量測報告。

回歸集只跑抽樣，抽樣看不出「整體策略退化成什麼都不做」這種問題。這支
腳本跑完整批，並且**用純網格 2×2 拼接**來驗證——實際擴圖
（`kuotu.image_pipeline.tile_to_canvas`）就是 `canvas.paste` 逐格貼，
`processor.tile_2x2_multi` 那種多圖錯位預覽會把接縫遮起來，不能拿來驗收。

用法：
    python tests/sweep_all.py                    # 全部
    python tests/sweep_all.py --limit 20         # 先跑前 20 張
    python tests/sweep_all.py --only "3 (25)"    # 只跑檔名含此字串的
    python tests/sweep_all.py --workers 8
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

import numpy as np
from PIL import Image, ImageDraw, ImageFont

Image.MAX_IMAGE_PIXELS = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SRC = Path(r"D:\5EDemocache\continue\100图")
OUT = ROOT / "tests" / "sweep_out"
EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

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


def run_case(job: tuple[str, str, bool]) -> dict:
    folder, name, write_sheet = job
    from app.color_utils import detect_background
    from app.processor import _to_rgb_array, make_seamless_hard_cut
    from app.quality import (
        axis_line_energy,
        design_error,
        geometry_fidelity,
        seam_report,
        tone_shift,
    )

    path = SRC / folder / name
    row: dict = {"folder": folder, "name": name}
    try:
        img = Image.open(path)
        img.load()
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
            safe = name.replace("/", "_")
            dest = OUT / "sheets" / f"{tag}__{folder}__{safe}.png"
            dest.parent.mkdir(parents=True, exist_ok=True)
            sheet.save(dest)
            row["sheet"] = str(dest.relative_to(OUT))
    except Exception as exc:  # noqa: BLE001 — 批次要繼續
        row["error"] = f"{exc}"
        row["trace"] = traceback.format_exc(limit=6)
        row["errors"] = ["EXCEPTION"]
    return row


def collect(only: str, limit: int) -> list[tuple[str, str]]:
    jobs: list[tuple[str, str]] = []
    for folder in sorted(p.name for p in SRC.iterdir() if p.is_dir()):
        for p in sorted((SRC / folder).glob("*.*")):
            if p.suffix.lower() not in EXTS:
                continue
            if only and only not in p.name:
                continue
            jobs.append((folder, p.name))
    return jobs[:limit] if limit else jobs


def _pixels(folder: str, name: str) -> int:
    try:
        with Image.open(SRC / folder / name) as im:
            return im.size[0] * im.size[1]
    except Exception:  # noqa: BLE001
        return 0


def write_index(rows: list[dict]) -> Path:
    """依嚴重度排序的 HTML 索引，方便一路往下看。"""

    def severity(r: dict) -> tuple:
        return (
            -len(r.get("errors") or []),
            -(r.get("wrap_excess") or 0),
            -(r.get("internal_excess") or 0),
        )

    ordered = sorted(rows, key=severity)
    n_fail = sum(1 for r in rows if r.get("errors"))
    parts = [
        "<!doctype html><meta charset='utf-8'>",
        "<style>body{font:14px system-ui;background:#f6f6f6;margin:16px}"
        "img{max-width:100%;border:1px solid #ccc;background:#fff}"
        "div{margin-bottom:18px}</style>",
        f"<h2>全量掃描 {len(rows)} 張，未通過 {n_fail}</h2>",
    ]
    for r in ordered:
        if not r.get("sheet"):
            continue
        parts.append(f"<div><img src='{r['sheet']}'></div>")
    dest = OUT / "index.html"
    dest.write_text("\n".join(parts), encoding="utf-8")
    return dest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--no-sheets", action="store_true")
    ap.add_argument("--tag", default="", help="報告檔名後綴，用來保留前後對照")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    pairs = collect(args.only, args.limit)
    if not pairs:
        print("沒有符合的圖")
        return 1

    small = [(f, n) for f, n in pairs if _pixels(f, n) < HUGE_PIXELS]
    huge = [(f, n) for f, n in pairs if _pixels(f, n) >= HUGE_PIXELS]
    workers = args.workers or min(12, os.cpu_count() or 4)
    print(f"共 {len(pairs)} 張（大圖 {len(huge)} 張另跑）| 行程 {workers}")

    rows: list[dict] = []
    t0 = time.perf_counter()
    done = 0
    total = len(pairs)

    def report(r: dict) -> None:
        nonlocal done
        done += 1
        tag = "FAIL" if r.get("errors") else "ok  "
        key = f"{r['folder']}/{r['name']}"
        if r.get("error"):
            print(f"[{done}/{total}] ERR  {key}: {r['error']}", flush=True)
            return
        print(
            f"[{done}/{total}] {tag} {key:34s} "
            f"接縫 {r['src_wrap_excess']:6.1f}->{r['wrap_excess']:5.1f} "
            f"內部 {r['src_internal_excess']:6.1f}->{r['internal_excess']:6.1f} "
            f"幾何 {r['fidelity']:.2f} {r['elapsed_s']:5.1f}s"
            + (f"  {r['errors']}" if r.get("errors") else ""),
            flush=True,
        )

    sheets = not args.no_sheets
    for group, n_proc in ((small, workers), (huge, 2)):
        if not group:
            continue
        jobs = [(f, n, sheets) for f, n in group]
        with mp.Pool(processes=n_proc, maxtasksperchild=1) as pool:
            for r in pool.imap_unordered(run_case, jobs, chunksize=1):
                rows.append(r)
                report(r)

    elapsed = time.perf_counter() - t0
    suffix = f"_{args.tag}" if args.tag else ""
    report_path = OUT / f"report{suffix}.json"
    report_path.write_text(
        json.dumps(
            sorted(rows, key=lambda r: (r["folder"], r["name"])),
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )

    n_fail = sum(1 for r in rows if r.get("errors"))
    n_unchanged = sum(1 for r in rows if r.get("unchanged"))
    print(f"\n{'=' * 70}")
    print(f"完成 {len(rows)} 張，耗時 {elapsed / 60:.1f} 分")
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
    raise SystemExit(main())
