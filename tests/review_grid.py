# -*- coding: utf-8 -*-
"""把 `sweep_all.py` 產出的接縫特寫排成網格，供逐張目視審查。

一張一張開 199 個對照圖太慢，但只看縮圖又看不出接縫。折衷做法是只取每張
對照圖最右邊那格「接縫 1:1 特寫」——那是判斷有沒有縫唯一可靠的視角——
再排成網格。有問題的再回去看完整對照圖。

用法：
    python tests/review_grid.py              # 全部
    python tests/review_grid.py --changed    # 只看實際動過刀的
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "sweep_out"
CELL = 250
COLS = 5
ROWS = 3


def _font(size: int) -> ImageFont.ImageFont:
    for name in ("msyh.ttc", "simhei.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(f"C:/Windows/Fonts/{name}", size)
        except OSError:
            continue
    return ImageFont.load_default()


def _zoom_of(sheet_path: Path) -> Image.Image:
    """對照圖最右邊那格就是接縫 1:1 特寫，右緣固定留 10px 邊距。"""
    sheet = Image.open(sheet_path)
    pad = 10
    side = min(420, sheet.height - pad - 56 - 20)
    box = (sheet.width - pad - side, pad, sheet.width - pad, pad + side)
    return sheet.crop(box).resize((CELL, CELL), Image.Resampling.LANCZOS)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default="")
    ap.add_argument("--changed", action="store_true", help="只看實際動過刀的")
    args = ap.parse_args()

    reports = sorted(OUT.glob("report_*.json"), key=lambda p: p.stat().st_mtime)
    path = Path(args.report) if args.report else (reports[-1] if reports else None)
    if path is None or not path.exists():
        print("找不到掃描報告，請先跑 tests/sweep_all.py")
        return 1
    rows = json.loads(path.read_text(encoding="utf-8"))
    rows = [r for r in rows if r.get("sheet")]
    if args.changed:
        rows = [r for r in rows if not r.get("unchanged")]
    rows.sort(key=lambda r: (-(r.get("wrap_excess") or 0), r["folder"], r["name"]))

    dest_dir = OUT / ("review_changed" if args.changed else "review")
    dest_dir.mkdir(parents=True, exist_ok=True)
    for old in dest_dir.glob("*.png"):
        old.unlink()

    per = COLS * ROWS
    pages = (len(rows) + per - 1) // per
    label_h = 20
    for page in range(pages):
        chunk = rows[page * per : (page + 1) * per]
        w = 8 + COLS * (CELL + 8)
        h = 8 + ROWS * (CELL + label_h + 8)
        grid = Image.new("RGB", (w, h), (248, 248, 248))
        d = ImageDraw.Draw(grid)
        for i, r in enumerate(chunk):
            cx = 8 + (i % COLS) * (CELL + 8)
            cy = 8 + (i // COLS) * (CELL + label_h + 8)
            try:
                grid.paste(_zoom_of(OUT / r["sheet"]), (cx, cy))
            except Exception:  # noqa: BLE001 — 缺圖不該中斷整份審查
                d.rectangle([cx, cy, cx + CELL, cy + CELL], fill=(230, 230, 230))
            colour = (200, 40, 40) if r.get("errors") else (70, 70, 70)
            d.rectangle(
                [cx, cy, cx + CELL - 1, cy + CELL - 1], outline=(180, 180, 180)
            )
            # 特寫正中央就是接縫十字，畫在框外免得遮住要看的東西
            d.line([cx + CELL // 2, cy - 5, cx + CELL // 2, cy - 1], fill=(220, 40, 40), width=2)
            d.line([cx - 5, cy + CELL // 2, cx - 1, cy + CELL // 2], fill=(220, 40, 40), width=2)
            tag = "改" if not r.get("unchanged") else "原"
            d.text(
                (cx + 1, cy + CELL + 3),
                f"[{tag}] {r['name'][:22]} 縫{r.get('wrap_excess', 0):.1f}",
                fill=colour,
                font=_font(12),
            )
        out = dest_dir / f"page{page + 1:02d}.png"
        grid.save(out)
        print(f"{out}  （{len(chunk)} 張）")
    print(f"\n共 {pages} 頁 / {len(rows)} 張")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
