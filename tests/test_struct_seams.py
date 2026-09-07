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
from app.quality import (
    motif_fragment_ratio,
    wrap_cut_ratio,
    wrap_hotspot,
    wrap_hotspot_axes,
    wrap_orphan_run,
    seam_report,
)
from app.select import (
    Candidate,
    HOTSPOT_CROP_OK,
    HOTSPOT_CUT_OK,
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
        arr = self._zero_excess_hot_wrap()
        src = source_facts(arr)
        cand = Candidate(arr, "mincut", True, [("mincut", object())])
        measure(src, cand)
        self.assertLess(cand.hotspot, HOTSPOT_CROP_OK)
        self.assertFalse(
            any(e.startswith("結構接縫") for e in cand.errors),
            cand.errors,
        )

    def test_mincut_label_skips_struct_seam_without_recipe(self) -> None:
        """清邊補花 recipe 是 None，後接最小誤差切不能再被結構接縫誤殺。"""
        arr = self._zero_excess_hot_wrap()
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
        src.ink = 0.50
        cand = Candidate(
            arr,
            "mincut",
            True,
            [("mincut", object())],
            motif_cut=0.018,
            motif_dense=0.50,
        )
        measure(src, cand)
        self.assertFalse(
            any(e.startswith("切線剖開圖案") for e in cand.errors),
            cand.errors,
        )

    def test_scatter_floral_does_not_get_dense_allowance(self) -> None:
        arr = np.full((64, 64, 3), 245, dtype=np.uint8)
        src = source_facts(arr)
        src.ink = 0.25
        cand = Candidate(
            arr,
            "mincut",
            True,
            [("mincut", object())],
            motif_cut=0.045,
            motif_dense=0.54,
        )
        measure(src, cand)
        self.assertTrue(
            any(e.startswith("切線剖開圖案") for e in cand.errors),
            cand.errors,
        )

    def test_mincut_hot_cut_is_struct_seam(self) -> None:
        arr = self._mismatch()
        src = source_facts(arr)
        cand = Candidate(
            arr,
            "mincut",
            True,
            [("mincut", object())],
            motif_cut=0.01,
        )
        measure(src, cand)
        self.assertGreater(cand.hotspot, HOTSPOT_CUT_OK)
        self.assertTrue(
            any(e.startswith("結構接縫") for e in cand.errors),
            cand.errors,
        )

    def test_refill_hot_is_struct_seam(self) -> None:
        arr = self._mismatch()
        src = source_facts(arr)
        cand = Candidate(arr, "清邊補花", False, None, motif_cut=0.0)
        measure(src, cand)
        self.assertTrue(
            any(e.startswith("結構接縫") for e in cand.errors),
            cand.errors,
        )

    def test_refill_does_not_fall_through_to_hotspot_ok(self) -> None:
        """清邊補花熱點 14–28 只能走 REFILL 閘門，不能被 HOTSPOT_OK 誤殺。"""
        arr = np.full((64, 64, 3), 240, dtype=np.uint8)
        arr[:, 0] = (200, 40, 40)
        arr[:, -1] = (180, 50, 40)
        src = source_facts(arr)
        cand = Candidate(arr, "清邊補花", False, None, motif_cut=0.0)
        measure(src, cand)
        self.assertGreater(cand.hotspot, 14.0)
        self.assertLessEqual(cand.hotspot, 28.0)
        self.assertFalse(
            any(e.startswith("結構接縫") for e in cand.errors),
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


class MotifIntegrityTests(unittest.TestCase):
    def test_bitten_stamps_flag_fragments(self) -> None:
        bg = (30, 40, 50)
        arr = np.full((260, 260, 3), bg, dtype=np.uint8)
        centers = (
            (50, 50),
            (130, 50),
            (210, 50),
            (50, 130),
            (130, 130),
            (210, 130),
            (90, 210),
            (170, 210),
        )
        for cx, cy in centers:
            cv2.circle(arr, (cx, cy), 24, (240, 240, 240), -1)
        cv2.circle(arr, (50 + 16, 50), 12, bg, -1)
        cv2.circle(arr, (130 + 16, 130), 12, bg, -1)
        cv2.circle(arr, (210 + 16, 50), 12, bg, -1)
        self.assertGreater(motif_fragment_ratio(arr), 0.10)

    def test_uniform_stars_are_not_fragments(self) -> None:
        bg = (30, 40, 50)
        arr = np.full((260, 260, 3), bg, dtype=np.uint8)
        for cx, cy in (
            (50, 50),
            (130, 50),
            (210, 50),
            (50, 130),
            (130, 130),
            (210, 130),
            (90, 210),
            (170, 210),
        ):
            for i in range(8):
                ang = i * np.pi / 4
                x2 = int(round(cx + 22 * np.cos(ang)))
                y2 = int(round(cy + 22 * np.sin(ang)))
                cv2.line(arr, (cx, cy), (x2, y2), (240, 240, 240), 2)
        self.assertLessEqual(motif_fragment_ratio(arr), 0.10)

    def test_edge_only_stamp_is_orphan(self) -> None:
        arr = np.full((160, 160, 3), 250, dtype=np.uint8)
        arr[40:100, :25] = (20, 20, 180)
        self.assertGreater(wrap_orphan_run(arr), 28)
        src = source_facts(arr)
        cand = Candidate(arr, "原圖", True, [])
        measure(src, cand)
        self.assertTrue(
            any(e.startswith("接縫殘片") for e in cand.errors),
            cand.errors,
        )

    def test_wrapping_stamp_is_not_orphan(self) -> None:
        arr = np.full((160, 160, 3), 250, dtype=np.uint8)
        arr[40:100, :20] = (20, 20, 180)
        arr[40:100, -20:] = (20, 20, 180)
        self.assertLessEqual(wrap_orphan_run(arr), 8)
        self.assertLessEqual(wrap_cut_ratio(arr), 0.15)

    def test_half_stamp_on_edge_is_wrap_cut(self) -> None:
        arr = np.full((220, 220, 3), 245, dtype=np.uint8)
        for cx, cy in ((70, 70), (150, 70), (70, 150), (150, 150)):
            cv2.circle(arr, (cx, cy), 22, (30, 30, 180), -1)
        arr[80:140, :18] = (30, 30, 180)
        self.assertGreater(wrap_cut_ratio(arr), 0.40)
        src = source_facts(arr)
        cand = Candidate(arr, "原圖", True, [])
        measure(src, cand)
        self.assertTrue(
            any(e.startswith("接縫切圖") or e.startswith("接縫殘片") for e in cand.errors),
            cand.errors,
        )

    def test_two_nicked_stamps_on_opposite_edges_are_wrap_cut(self) -> None:
        arr = np.full((220, 220, 3), 245, dtype=np.uint8)
        for cx, cy in ((70, 70), (150, 70), (70, 150), (150, 150)):
            cv2.circle(arr, (cx, cy), 22, (30, 30, 180), -1)
        cv2.circle(arr, (18, 110), 22, (30, 30, 180), -1)
        cv2.circle(arr, (201, 110), 22, (30, 30, 180), -1)
        self.assertGreater(wrap_cut_ratio(arr), 0.40)

    def test_true_wrap_halves_are_not_wrap_cut(self) -> None:
        arr = np.full((220, 220, 3), 245, dtype=np.uint8)
        for cx, cy in ((70, 70), (150, 70), (70, 150), (150, 150)):
            cv2.circle(arr, (cx, cy), 22, (30, 30, 180), -1)
        cv2.circle(arr, (0, 110), 22, (30, 30, 180), -1)
        cv2.circle(arr, (219, 110), 22, (30, 30, 180), -1)
        self.assertLessEqual(wrap_cut_ratio(arr), 0.15)

    def test_cold_mincut_on_cut_source_is_fake_join(self) -> None:
        src_arr = np.full((220, 220, 3), 245, dtype=np.uint8)
        for cx, cy in ((70, 70), (150, 70), (70, 150), (150, 150)):
            cv2.circle(src_arr, (cx, cy), 22, (30, 30, 180), -1)
        src_arr[80:140, :18] = (30, 30, 180)
        src = source_facts(src_arr)
        self.assertGreater(src.wrap_cut, 0.40)
        cand_arr = np.full((220, 220, 3), 245, dtype=np.uint8)
        for cx, cy in ((70, 70), (150, 70), (70, 150), (150, 150)):
            cv2.circle(cand_arr, (cx, cy), 22, (30, 30, 180), -1)
        cand_arr[80:140, :20] = (30, 30, 180)
        cand_arr[80:140, -20:] = (30, 30, 180)
        cand = Candidate(
            cand_arr,
            "mincut",
            True,
            [("mincut", object())],
            motif_cut=0.0,
        )
        measure(src, cand)
        self.assertTrue(
            any(e.startswith("接縫切圖") for e in cand.errors),
            cand.errors,
        )
        crop = Candidate(
            cand_arr,
            "週期裁切",
            True,
            [("crop", (0, 0, 220, 220))],
            motif_cut=0.0,
        )
        measure(src, crop)
        self.assertTrue(
            any(e.startswith("接縫切圖") for e in crop.errors),
            crop.errors,
        )


class ClearRefillRepairTests(unittest.TestCase):
    def test_edge_cut_disks_wrap_complete(self) -> None:
        n = 280
        bg = (245, 245, 240)
        fg = (40, 80, 160)
        arr = np.full((n, n, 3), bg, dtype=np.uint8)
        for cx, cy in (
            (70, 70),
            (140, 70),
            (210, 70),
            (70, 140),
            (140, 140),
            (210, 140),
            (70, 210),
            (140, 210),
            (210, 210),
        ):
            cv2.circle(arr, (cx, cy), 22, fg, -1)
        cv2.circle(arr, (6, 140), 22, fg, -1)
        cv2.circle(arr, (n - 6, 90), 22, fg, -1)
        cv2.circle(arr, (90, 6), 22, fg, -1)
        cv2.circle(arr, (180, n - 6), 22, fg, -1)
        from app.processor import _clear_and_refill

        filled = _clear_and_refill(arr, bg, 40.0, 8)
        self.assertIsNotNone(filled)
        self.assertLessEqual(wrap_cut_ratio(filled), 0.40, wrap_cut_ratio(filled))
        self.assertLessEqual(wrap_orphan_run(filled), 28, wrap_orphan_run(filled))
        self.assertLessEqual(wrap_hotspot(filled), 14.0, wrap_hotspot(filled))

    def test_interior_bite_is_replaced(self) -> None:
        n = 280
        bg = (30, 40, 50)
        arr = np.full((n, n, 3), bg, dtype=np.uint8)
        centers = (
            (70, 70),
            (140, 70),
            (210, 70),
            (70, 140),
            (140, 140),
            (210, 140),
            (70, 210),
            (140, 210),
            (210, 210),
        )
        for cx, cy in centers:
            cv2.circle(arr, (cx, cy), 24, (240, 240, 240), -1)
        cv2.circle(arr, (140 + 16, 140), 12, bg, -1)
        cv2.circle(arr, (70 + 16, 70), 12, bg, -1)
        self.assertGreater(motif_fragment_ratio(arr), 0.10)
        from app.processor import _clear_and_refill

        filled = _clear_and_refill(arr, bg, 40.0, 8)
        self.assertIsNotNone(filled)
        self.assertLessEqual(motif_fragment_ratio(filled), 0.10)

    def test_overlay_gingham_stamps_are_detected(self) -> None:
        n = 240
        arr = np.zeros((n, n, 3), dtype=np.uint8)
        cell = 12
        for y in range(0, n, cell):
            for x in range(0, n, cell):
                shade = 200 if ((x // cell) + (y // cell)) % 2 else 110
                arr[y : y + cell, x : x + cell] = shade
        yellow = (240, 200, 40)
        for cx, cy in (
            (40, 40),
            (90, 40),
            (140, 40),
            (190, 40),
            (40, 100),
            (90, 100),
            (140, 100),
            (190, 100),
            (40, 160),
            (90, 160),
            (140, 160),
            (190, 160),
        ):
            cv2.circle(arr, (cx, cy), 14, yellow, -1)
        cv2.circle(arr, (4, 120), 14, yellow, -1)
        cv2.circle(arr, (n - 4, 80), 14, yellow, -1)
        from app.processor import _clear_and_refill, _overlay_stamp_mask

        bg = (156, 156, 156)
        self.assertIsNotNone(_overlay_stamp_mask(arr, bg, 40.0))
        filled = _clear_and_refill(arr, bg, 40.0, 8)
        self.assertIsNotNone(filled)
        from app.processor import stamp_structure_view

        view = stamp_structure_view(filled, bg, 40.0)
        self.assertLessEqual(wrap_cut_ratio(view), 0.40, wrap_cut_ratio(view))
        self.assertLessEqual(wrap_orphan_run(view), 28, wrap_orphan_run(view))

    def test_opening_splits_stem_linked_disks(self) -> None:
        n = 320
        bg = (250, 250, 250)
        fg = (30, 120, 40)
        arr = np.full((n, n, 3), bg, dtype=np.uint8)
        centers = [
            (60, 60),
            (140, 60),
            (220, 60),
            (60, 140),
            (140, 140),
            (220, 140),
            (60, 220),
            (140, 220),
            (220, 220),
            (60, 280),
            (140, 280),
            (220, 280),
        ]
        for cx, cy in centers:
            cv2.circle(arr, (cx, cy), 18, fg, -1)
        # 細莖把同一列的圓連成一塊
        for y in (60, 140, 220, 280):
            cv2.line(arr, (60, y), (220, y), fg, 2)
        cv2.circle(arr, (8, 140), 18, fg, -1)
        cv2.circle(arr, (n - 8, 200), 18, fg, -1)
        from app.processor import _split_touching_stamps, _foreground_mask

        raw = _foreground_mask(arr, bg, 40.0).astype(np.uint8)
        n0, _, st0, _ = cv2.connectedComponentsWithStats(raw, 8)
        big0 = int((st0[1:, cv2.CC_STAT_AREA] >= 200).sum())
        split = _split_touching_stamps(raw)
        n1, _, st1, _ = cv2.connectedComponentsWithStats(split, 8)
        big1 = int((st1[1:, cv2.CC_STAT_AREA] >= 200).sum())
        self.assertGreater(big1, big0)
        from app.processor import _clear_and_refill

        filled = _clear_and_refill(arr, bg, 40.0, 8)
        self.assertIsNotNone(filled)
        self.assertLessEqual(wrap_cut_ratio(filled), 0.40, wrap_cut_ratio(filled))

    def test_forced_edge_snaps_inward_centroid(self) -> None:
        from app.processor import MotifStamp, _snap_wrap_placement

        mask = np.ones((40, 40), dtype=bool)
        motif = MotifStamp(patch=np.zeros((40, 40, 3), dtype=np.uint8), mask=mask, area=1600, cy=20.0, cx=20.0)
        cy, cx, is_edge = _snap_wrap_placement(
            80.0, 80.0, 700, motif, 240, 240, forced_edge=True
        )
        self.assertTrue(is_edge)
        self.assertTrue(cx < 1.0 or cx > 239.0 or cy < 1.0 or cy > 239.0)


if __name__ == "__main__":
    unittest.main()
