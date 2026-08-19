# -*- coding: utf-8 -*-
"""分流校準：封面召回 ≥ 90%，花布誤殺 = 0。合成外框／黑底印花也要對。"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.color_utils import has_finished_border
from app.triage import VERDICT_FRAMED, VERDICT_NOT_PATTERN, VERDICT_TILEABLE, triage

SRC = ROOT / "samples" / "F"
LABELS = Path(__file__).resolve().parent / "triage_labels.json"
RECALL_MIN = 0.90


def _load_labels() -> dict:
    return json.loads(LABELS.read_text(encoding="utf-8"))


class SyntheticFrameTests(unittest.TestCase):
    def test_kuotu_double_frame_is_framed(self) -> None:
        flower = Image.new("RGB", (240, 240), (40, 120, 80))
        draw = ImageDraw.Draw(flower)
        for y in range(40, 200, 24):
            for x in range(40, 200, 24):
                draw.ellipse((x, y, x + 14, y + 14), fill=(200, 40, 40))
        framed = ImageOps.expand(flower, border=24, fill="white")
        framed = ImageOps.expand(framed, border=12, fill="black")
        result = triage(framed)
        self.assertEqual(result.verdict, VERDICT_FRAMED)
        self.assertTrue(has_finished_border(framed))

    def test_solid_black_frame_is_framed(self) -> None:
        inner = Image.new("RGB", (200, 200), (220, 80, 90))
        framed = ImageOps.expand(inner, border=18, fill=(8, 8, 8))
        self.assertEqual(triage(framed).verdict, VERDICT_FRAMED)

    def test_solid_white_frame_is_framed(self) -> None:
        inner = Image.new("RGB", (200, 200), (30, 90, 40))
        framed = ImageOps.expand(inner, border=16, fill=(250, 250, 250))
        self.assertEqual(triage(framed).verdict, VERDICT_FRAMED)

    def test_black_print_is_tileable(self) -> None:
        black_print = Image.new("RGB", (240, 240), (8, 8, 8))
        ImageDraw.Draw(black_print).ellipse((60, 60, 180, 180), fill=(180, 30, 40))
        result = triage(black_print)
        self.assertEqual(result.verdict, VERDICT_TILEABLE)
        self.assertFalse(has_finished_border(black_print))

    def test_plain_floral_is_tileable(self) -> None:
        flower = Image.new("RGB", (240, 240), (40, 120, 80))
        draw = ImageDraw.Draw(flower)
        for y in range(16, 224, 24):
            for x in range(16, 224, 24):
                draw.ellipse((x, y, x + 14, y + 14), fill=(200, 40, 40))
        result = triage(flower)
        self.assertEqual(result.verdict, VERDICT_TILEABLE)
        self.assertFalse(has_finished_border(flower))


@unittest.skipUnless(SRC.is_dir(), "samples/F 不存在")
class CalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.labels = _load_labels()

    def _run(self, rel: str):
        path = SRC / rel
        self.assertTrue(path.is_file(), f"找不到例圖 {rel}")
        img = Image.open(path)
        img.load()
        return triage(img)

    def test_covers_recall(self) -> None:
        covers = self.labels["covers"]
        hits = 0
        misses: list[str] = []
        for item in covers:
            result = self._run(item["path"])
            if result.verdict == VERDICT_NOT_PATTERN:
                hits += 1
            else:
                misses.append(
                    f"{item['path']} -> {result.verdict} {result.reasons} "
                    f"quad={result.signals.get('quad', 0):.3f} "
                    f"het={result.signals.get('het', 0):.1f} "
                    f"bg={result.signals.get('bg_frac', 0):.2f} "
                    f"grad={result.signals.get('backdrop_range', 0):.1f} "
                    f"text={result.signals.get('text_rows', 0):.0f} "
                    f"lop={result.signals.get('lopsided', 0):.0f}"
                )
        recall = hits / max(len(covers), 1)
        self.assertGreaterEqual(
            recall,
            RECALL_MIN,
            f"封面召回 {recall:.0%} < {RECALL_MIN:.0%}\n" + "\n".join(misses),
        )

    def test_hard_tileable_not_killed(self) -> None:
        killed: list[str] = []
        for item in self.labels["tileable_hard"]:
            result = self._run(item["path"])
            if result.verdict != VERDICT_TILEABLE:
                killed.append(
                    f"{item['path']} -> {result.verdict} {result.reasons} "
                    f"{item.get('why', '')}"
                )
        self.assertEqual(killed, [], "誤殺花布：\n" + "\n".join(killed))


if __name__ == "__main__":
    unittest.main()
