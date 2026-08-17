# -*- coding: utf-8 -*-
"""單張除錯：印出分類、每個候選的量測與最終取捨。

用法：
    python tests/debug_one.py "100图-1/2 (29).jpg"
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SRC = Path(r"D:\5EDemocache\continue\100图")


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    rel = sys.argv[1].replace("\\", "/")
    path = SRC / rel
    if not path.exists():
        print(f"找不到 {path}")
        return 1

    from app.color_utils import detect_background
    from app.processor import _to_rgb_array, make_seamless_hard_cut
    from app.quality import design_error, seam_report, tone_shift

    img = Image.open(path)
    img.load()
    bg = detect_background(img)
    src = _to_rgb_array(img, bg)
    print(f"{rel}  {img.size[0]}×{img.size[1]}  mode={img.mode}  bg={bg}")
    print(f"原稿 {seam_report(src).describe()}")
    print("-" * 78)

    t0 = time.perf_counter()
    unit, mode = make_seamless_hard_cut(img, bg, log=print)
    elapsed = time.perf_counter() - t0

    out = _to_rgb_array(unit, bg)
    print("-" * 78)
    print(f"結果 {unit.size[0]}×{unit.size[1]}  {elapsed:.1f}s")
    print(f"     {seam_report(out).describe()}")
    print(f"     還原 {design_error(src, out):.2f}  色調 {tone_shift(src, out):.2f}")
    print(f"     {mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
