# -*- coding: utf-8 -*-
"""結構接縫：wrap 色差為 0 仍可能在 2×2 裂開。"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.processor import (
    _compact_period_jobs,
    _foreground_mask,
    _join_foreground,
    _luminance_map,
    make_seamless_hard_cut,
)
from app.quality import wrap_hotspot, wrap_hotspot_axes, seam_report
from app.select import (
    Candidate,
    MOTIF_CUT_DENSE_INK,
    MOTIF_CUT_MAX,
    compact_min_edge,
    make_seamless_variants,
    measure,
    min_unit_edge,
    source_facts,
    source_looks_seamless,
)


class WrapHotspotTests(unittest.TestCase):
    def test_flat_field_is_cold(self) -> None:
        arr = np.full((80, 80, 3), 240, dtype=np.uint8)
        self.assertLess(wrap_hotspot(arr), 1.0)

    def test_sparse_stamp_mismatch_is_hot(self) -> None:
        arr = np.full((120, 120, 3), 250, dtype=np.uint8)
        arr[40:80, :10] = (180, 20, 20)
        arr[40:80, -10:] = (20, 40, 180)
        self.assertGreater(wrap_hotspot(arr), 14.0)
        src = source_facts(arr)
        self.assertFalse(source_looks_seamless(src.rep, src.hotspot))


class CompactStripeTests(unittest.TestCase):
    def test_keeps_full_width_jobs_when_both_axes_peak(self) -> None:
        h = w = 240
        arr = np.full((h, w, 3), 240, dtype=np.uint8)
        for y in range(0, h, 40):
            arr[y : y + 20] = (40, 80, 160)
        for x in range(0, w, 80):
            arr[:, x : x + 2] = np.clip(
                arr[:, x : x + 2].astype(np.int16) - 12, 0, 255
            ).astype(np.uint8)
        jobs = _compact_period_jobs(arr, _luminance_map(arr))
        full_width = [j for j in jobs if j[2] == w]
        self.assertTrue(full_width, f"沒有滿幅條帶候選：{jobs}")


class MotifCutSparseTests(unittest.TestCase):
    def test_band_density_does_not_relax_sparse_stamps(self) -> None:
        arr = np.full((64, 64, 3), 245, dtype=np.uint8)
        arr[8:20, 8:20] = (200, 30, 30)
        src = source_facts(arr)
        self.assertLess(src.ink, MOTIF_CUT_DENSE_INK)
        cand = Candidate(
            arr,
            "mincut",
            True,
            [],
            motif_cut=0.04,
            motif_dense=0.62,
        )
        measure(src, cand)
        self.assertTrue(
            any(e.startswith("切線剖開圖案") for e in cand.errors),
            cand.errors,
        )
        self.assertLessEqual(MOTIF_CUT_MAX, 0.02)


class HotspotForcesMincutTests(unittest.TestCase):
    def _mismatch(self) -> np.ndarray:
        arr = np.full((120, 120, 3), 250, dtype=np.uint8)
        arr[40:80, :10] = (180, 20, 20)
        arr[40:80, -10:] = (20, 40, 180)
        return arr

    def _zero_excess_hot_wrap(self) -> np.ndarray:
        """地色接得上、局部圖章對不上：平均超出量 ≈ 0，熱點仍高。"""
        h = w = 160
        yy, xx = np.mgrid[:h, :w]
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        arr[..., 0] = (
            210 + 35 * np.sin(2 * np.pi * xx / 20) + 15 * np.sin(2 * np.pi * yy / 16)
        ).clip(0, 255).astype(np.uint8)
        arr[..., 1] = (200 + 25 * np.cos(2 * np.pi * xx / 20)).clip(0, 255).astype(
            np.uint8
        )
        arr[..., 2] = (
            190 + 20 * np.sin(2 * np.pi * xx / 16) + 10 * np.sin(2 * np.pi * yy / 20)
        ).clip(0, 255).astype(np.uint8)
        arr[20:60, :2] = (80, 40, 40)
        arr[20:60, -2:] = (40, 40, 80)
        return arr

    def test_zero_excess_mismatch_still_expands_mincut(self) -> None:
        arr = self._zero_excess_hot_wrap()
        src = source_facts(arr)
        self.assertFalse(source_looks_seamless(src.rep, src.hotspot))
        self.assertLessEqual(src.rep.wrap_excess, 1.0)
        hot_v, hot_h = wrap_hotspot_axes(arr)
        self.assertGreater(max(hot_v, hot_h), 14.0)
        cands = make_seamless_variants(arr, "原圖", lossless=True, recipe=[])
        self.assertTrue(
            any("最小誤差切" in c.label for c in cands),
            [c.label for c in cands],
        )

    def test_mincut_recipe_skips_struct_seam_gate(self) -> None:
        arr = self._mismatch()
        src = source_facts(arr)
        cand = Candidate(arr, "mincut", True, [("mincut", object())])
        measure(src, cand)
        self.assertFalse(
            any(e.startswith("結構接縫") for e in cand.errors),
            cand.errors,
        )

    def test_mincut_label_skips_struct_seam_without_recipe(self) -> None:
        """清邊補花 recipe 是 None，後接最小誤差切不能再被結構接縫誤殺。"""
        arr = self._mismatch()
        src = source_facts(arr)
        cand = Candidate(
            arr,
            "清邊補花＋最小誤差切(V帶10px)",
            False,
            None,
            motif_cut=0.0,
        )
        measure(src, cand)
        self.assertFalse(
            any(e.startswith("結構接縫") for e in cand.errors),
            cand.errors,
        )

    def test_raw_source_still_flags_struct_seam(self) -> None:
        arr = self._mismatch()
        src = source_facts(arr)
        cand = Candidate(arr, "原圖", True, [])
        measure(src, cand)
        self.assertTrue(
            any(e.startswith("結構接縫") for e in cand.errors),
            cand.errors,
        )

    def test_medium_ink_uses_dense_motif_allowance(self) -> None:
        arr = np.full((64, 64, 3), 245, dtype=np.uint8)
        src = source_facts(arr)
        src.ink = 0.16
        cand = Candidate(
            arr,
            "mincut",
            True,
            [("mincut", object())],
            motif_cut=0.03,
            motif_dense=0.50,
        )
        measure(src, cand)
        self.assertFalse(
            any(e.startswith("切線剖開圖案") for e in cand.errors),
            cand.errors,
        )


class ThinLineJoinTests(unittest.TestCase):
    def test_four_connected_splits_diagonal_stroke(self) -> None:
        arr = np.full((80, 80, 3), 20, dtype=np.uint8)
        for i in range(10, 70):
            arr[i, i] = (200, 200, 200)
        bg = (20, 20, 20)
        raw = _foreground_mask(arr, bg, 40.0).astype(np.uint8)
        n4, *_ = cv2.connectedComponentsWithStats(raw, connectivity=4)
        joined = _join_foreground(arr, bg, 40.0)
        n8, *_ = cv2.connectedComponentsWithStats(joined, connectivity=8)
        self.assertGreater(n4 - 1, 20)
        self.assertEqual(n8 - 1, 1)


class ThinLineScatterTests(unittest.TestCase):
    def _star(self, arr: np.ndarray, cx: int, cy: int, r: int, color: tuple[int, int, int]) -> None:
        for i in range(8):
            ang = i * np.pi / 4 + 0.18
            x2 = int(round(cx + r * np.cos(ang)))
            y2 = int(round(cy + r * np.sin(ang)))
            cv2.line(arr, (int(cx), int(cy)), (x2, y2), color, 1)

    def test_edge_cut_thin_stars_are_not_unmet(self) -> None:
        n = 220
        bg = (128, 8, 40)
        fg = (240, 230, 200)
        arr = np.full((n, n, 3), bg, dtype=np.uint8)
        for cx, cy in ((70, 70), (150, 70), (70, 150), (150, 150), (110, 110)):
            self._star(arr, cx, cy, 28, fg)
        self._star(arr, 8, 110, 32, fg)
        self._star(arr, n - 8, 80, 32, fg)
        self._star(arr, 90, 6, 30, fg)
        self._star(arr, 140, n - 6, 30, fg)
        unit, mode = make_seamless_hard_cut(Image.fromarray(arr), bg=bg)
        self.assertNotIn("未達標", mode, mode)
        u = np.asarray(unit.convert("RGB"))
        self.assertLessEqual(seam_report(u).wrap_excess, 2.0, mode)
        self.assertLessEqual(wrap_hotspot(u), 14.0, mode)


class UnitSizeGateTests(unittest.TestCase):
    def test_measure_rejects_relative_tiny_crop(self) -> None:
        src_arr = np.full((400, 400, 3), 200, dtype=np.uint8)
        src = source_facts(src_arr)
        tiny = Candidate(
            src_arr[:80, :80].copy(),
            "crop",
            True,
            [("crop", (0, 0, 80, 80))],
        )
        measure(src, tiny)
        self.assertTrue(
            any(e.startswith("單元過小") for e in tiny.errors),
            tiny.errors,
        )

    def test_measure_keeps_reasonable_stripe(self) -> None:
        src_arr = np.full((400, 400, 3), 200, dtype=np.uint8)
        src = source_facts(src_arr)
        stripe = Candidate(
            src_arr[:, :100].copy(),
            "crop",
            True,
            [("crop", (0, 0, 400, 100))],
        )
        measure(src, stripe)
        self.assertFalse(
            any(e.startswith("單元過小") for e in stripe.errors),
            stripe.errors,
        )

    def test_compact_jobs_skip_one_cell_weave(self) -> None:
        n = 320
        arr = np.full((n, n, 3), 210, dtype=np.uint8)
        for y in range(0, n, 16):
            arr[y : y + 8] = (180, 140, 110)
        for x in range(0, n, 16):
            arr[:, x : x + 8] = np.clip(
                arr[:, x : x + 8].astype(np.int16) - 20, 0, 255
            ).astype(np.uint8)
        jobs = _compact_period_jobs(arr, _luminance_map(arr))
        need = compact_min_edge(n, n)
        for _px, _py, cw, ch in jobs:
            self.assertGreaterEqual(min(cw, ch), need)

    def test_weave_unit_keeps_usable_edge(self) -> None:
        n = 256
        arr = np.full((n, n, 3), 210, dtype=np.uint8)
        for y in range(0, n, 16):
            arr[y : y + 8] = (180, 140, 110)
        for x in range(0, n, 16):
            arr[:, x : x + 8] = np.clip(
                arr[:, x : x + 8].astype(np.int16) - 20, 0, 255
            ).astype(np.uint8)
        unit, mode = make_seamless_hard_cut(Image.fromarray(arr), bg=(210, 210, 210))
        u = np.asarray(unit.convert("RGB"))
        self.assertGreaterEqual(min(u.shape[:2]), min_unit_edge(n, n), mode)
        self.assertLessEqual(seam_report(u).wrap_excess, 2.0, mode)


if __name__ == "__main__":
    unittest.main()
