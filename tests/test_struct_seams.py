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
    wrap_gutter_error,
    wrap_hotspot,
    wrap_hotspot_axes,
    wrap_orphan_run,
    wrap_period_remainder,
    seam_report,
)
from app.select import (
    Candidate,
    GUTTER_ERR_MAX,
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


def _max_abs_path_step(path: np.ndarray | None) -> int:
    if path is None or len(path) < 2:
        return 0
    return int(np.abs(np.diff(path.astype(np.int32))).max())


def _longest_true_run(mask_1d: np.ndarray) -> int:
    if not mask_1d.any():
        return 0
    padded = np.concatenate(([False], mask_1d.astype(bool), [False]))
    d = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return int((ends - starts).max()) if len(starts) else 0


def _max_sandwiched_bg_run(
    arr: np.ndarray, bg: tuple[int, int, int], thr: float = 28.0
) -> int:
    """最長的「上下（或左右）都是墨水、中間卻是地色」連段——掃描線斷層。"""
    bgv = np.asarray(bg, dtype=np.float32)
    dist = np.linalg.norm(arr.astype(np.float32) - bgv, axis=2)
    is_bg = dist < thr
    is_ink = ~is_bg
    worst = 0
    h, w = is_bg.shape
    for y in range(1, h - 1):
        worst = max(
            worst, _longest_true_run(is_bg[y] & is_ink[y - 1] & is_ink[y + 1])
        )
    for x in range(1, w - 1):
        worst = max(
            worst, _longest_true_run(is_bg[:, x] & is_ink[:, x - 1] & is_ink[:, x + 1])
        )
    return worst


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
        self.assertFalse(
            source_looks_seamless(
                src.rep,
                src.hotspot,
                wrap_cut=src.wrap_cut,
                orphan=src.orphan,
                fragment=src.fragment,
                gutter=src.gutter,
                wrap_density=src.wrap_density,
            )
        )

    def test_dense_matched_wrap_allows_crop_hotspot(self) -> None:
        """滿版碎花色差縫為 0、切圖為 0 時，熱點 61 不當結構縫。"""
        from unittest.mock import patch

        arr = np.full((96, 96, 3), 28, dtype=np.uint8)
        src = source_facts(arr)
        cand = Candidate(arr, "原圖", True, [])
        with patch("app.select.wrap_hotspot", return_value=61.0):
            measure(src, cand)
        self.assertFalse(
            any(e.startswith("結構接縫") for e in cand.errors),
            cand.errors,
        )
        self.assertTrue(
            source_looks_seamless(
                src.rep,
                61.0,
                wrap_cut=0.0,
                orphan=0,
                fragment=0.0,
                gutter=0.0,
                wrap_density=1.0,
            )
        )


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


class CompactPolkaStripTests(unittest.TestCase):
    def test_skips_full_width_when_x_period_does_not_divide(self) -> None:
        h = w = 260
        arr = np.full((h, w, 3), (40, 160, 90), dtype=np.uint8)
        for y in range(30, h, 50):
            for x in range(30, w, 50):
                cv2.circle(arr, (x, y), 12, (250, 250, 250), -1)
        jobs = _compact_period_jobs(arr, _luminance_map(arr))
        full_width = [j for j in jobs if j[2] == w]
        self.assertFalse(full_width, f"不該有滿幅條帶：{jobs}")


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
        self.assertFalse(
            source_looks_seamless(
                src.rep,
                src.hotspot,
                wrap_cut=src.wrap_cut,
                orphan=src.orphan,
                fragment=src.fragment,
                gutter=src.gutter,
                wrap_density=src.wrap_density,
            )
        )
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


    def test_mincut_path_stays_8_connected(self) -> None:
        from app.seamless_core import wrap_mincut

        h, w = 200, 240
        arr = np.full((h, w, 3), 240, dtype=np.uint8)
        # 寬圓錯開時八連通切線繞不過，這是預期：不該用數十像素橫跳撕出地色條。
        cv2.circle(arr, (35, 100), 55, (20, 40, 180), -1)
        cv2.circle(arr, (w - 35, 100), 55, (180, 40, 20), -1)
        cut, info = wrap_mincut(arr, do_v=True, do_h=False, max_band_frac=0.50)
        self.assertGreater(info.band_v, 0)
        self.assertEqual(cut.shape[0], h)
        self.assertLessEqual(_max_abs_path_step(info.path_v), 1, info.describe())

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

    def test_refill_thin_line_wrap_hot_is_not_struct_seam(self) -> None:
        """細線跨縫：切圖與色差已過時，熱點高不算結構縫。"""
        arr = np.full((120, 120, 3), 220, dtype=np.uint8)
        arr[:, :4] = (250, 250, 250)
        arr[:, -4:] = (250, 250, 250)
        src = source_facts(arr)
        cand = Candidate(arr, "清邊補花", False, None, motif_cut=0.0)
        measure(src, cand)
        self.assertFalse(
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


class MincutScanlineTearTests(unittest.TestCase):
    def test_thin_motifs_do_not_get_background_scanlines(self) -> None:
        from app.seamless_core import wrap_mincut

        h, w = 180, 220
        bg = (220, 40, 40)
        arr = np.full((h, w, 3), bg, dtype=np.uint8)
        white = (250, 250, 250)
        for cx in (36, w // 2, w - 36):
            for cy in (36, 90, 144):
                cv2.circle(arr, (cx, cy), 16, white, 2)
                cv2.line(arr, (cx - 18, cy), (cx + 18, cy), white, 2)
                cv2.line(arr, (cx, cy - 18), (cx, cy + 18), white, 2)
        cut, info = wrap_mincut(arr, do_v=True, do_h=True, max_band_frac=0.30)
        self.assertLessEqual(_max_abs_path_step(info.path_v), 1, info.describe())
        self.assertLessEqual(_max_abs_path_step(info.path_h), 1, info.describe())
        # 八連通階梯最多留下 1–2px 缺口；舊的 64px 橫跳會拉出整條地色掃描線。
        self.assertLessEqual(_max_sandwiched_bg_run(cut, bg), 6, info.describe())


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
    def test_mincut_ignores_exploded_wrap_cut_metric(self) -> None:
        """滿版碎花 mincut 的 wrap_cut 連通域可到 100%+，改看切線／熱點。"""
        from unittest.mock import patch

        arr = np.full((96, 96, 3), 18, dtype=np.uint8)
        arr[:, :3] = arr[:, -3:]
        arr[:3] = arr[-3:]
        src = source_facts(arr)
        cand = Candidate(
            arr,
            "最小誤差切",
            True,
            [("mincut", None)],
            motif_cut=0.002,
            motif_dense=0.2,
        )
        with patch("app.select.wrap_cut_ratio", return_value=1.83):
            measure(src, cand)
        self.assertFalse(
            any(e.startswith("接縫切圖") for e in cand.errors),
            cand.errors,
        )

    def test_refill_orphan_29_is_antialias_not_half_stamp(self) -> None:
        from unittest.mock import patch

        arr = np.full((96, 96, 3), 18, dtype=np.uint8)
        src = source_facts(arr)
        cand = Candidate(arr, "清邊補花", False, None)
        with patch("app.select.wrap_orphan_run", return_value=29):
            measure(src, cand)
        self.assertFalse(
            any(e.startswith("接縫殘片") for e in cand.errors),
            cand.errors,
        )

    def test_period_crop_skips_vine_fragment_false_positive(self) -> None:
        """週期裁切切圖為 0 時，藤蔓實心度離群不算圖案殘缺。"""
        from unittest.mock import patch

        arr = np.full((96, 96, 3), 18, dtype=np.uint8)
        arr[:, :4] = arr[:, -4:]
        arr[:4] = arr[-4:]
        src = source_facts(arr)
        cand = Candidate(
            arr,
            "週期裁切",
            True,
            [("crop", (0, 0, 96, 96))],
        )
        with patch("app.select.motif_fragment_ratio", return_value=0.11):
            measure(src, cand)
        self.assertFalse(
            any(e.startswith("圖案殘缺") for e in cand.errors),
            cand.errors,
        )

    def test_refill_still_flags_fragments(self) -> None:
        from unittest.mock import patch

        arr = np.full((96, 96, 3), 18, dtype=np.uint8)
        src = source_facts(arr)
        cand = Candidate(arr, "清邊補花", False, None)
        with patch("app.select.motif_fragment_ratio", return_value=0.11):
            measure(src, cand)
        self.assertTrue(
            any(e.startswith("圖案殘缺") for e in cand.errors),
            cand.errors,
        )

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

    def test_connected_field_is_not_huge_wrap_cut(self) -> None:
        """滿版連通底紋碰四邊，不能把整塊當切圖算出幾百倍。"""
        arr = np.full((220, 220, 3), 245, dtype=np.uint8)
        cv2.rectangle(arr, (0, 0), (219, 219), (20, 80, 40), 14)
        cv2.rectangle(arr, (0, 90), (219, 130), (20, 80, 40), -1)
        cv2.rectangle(arr, (90, 0), (130, 219), (20, 80, 40), -1)
        for cx, cy in ((45, 45), (175, 45), (45, 175), (175, 175)):
            cv2.circle(arr, (cx, cy), 12, (20, 80, 40), -1)
        self.assertLessEqual(wrap_cut_ratio(arr), 0.40)

    def test_specks_do_not_inflate_wrap_cut(self) -> None:
        """內部碎點不能把跨縫大圖章算成幾百倍切圖。"""
        arr = np.full((320, 320, 3), 245, dtype=np.uint8)
        for cx, cy in ((90, 90), (230, 90), (90, 230), (230, 230)):
            cv2.circle(arr, (cx, cy), 36, (30, 30, 180), -1)
        rng = np.random.default_rng(0)
        for _ in range(40):
            cv2.circle(
                arr,
                (int(rng.integers(50, 270)), int(rng.integers(50, 270))),
                2,
                (30, 30, 180),
                -1,
            )
        cv2.circle(arr, (0, 160), 36, (30, 30, 180), -1)
        cv2.circle(arr, (319, 160), 36, (30, 30, 180), -1)
        self.assertLessEqual(wrap_cut_ratio(arr), 0.40)

    def test_half_stamp_with_specks_is_still_wrap_cut(self) -> None:
        arr = np.full((320, 320, 3), 245, dtype=np.uint8)
        for cx, cy in ((90, 90), (230, 90), (90, 230), (230, 230)):
            cv2.circle(arr, (cx, cy), 36, (30, 30, 180), -1)
        rng = np.random.default_rng(1)
        for _ in range(40):
            cv2.circle(
                arr,
                (int(rng.integers(50, 270)), int(rng.integers(50, 270))),
                2,
                (30, 30, 180),
                -1,
            )
        arr[120:200, :28] = (30, 30, 180)
        self.assertGreater(wrap_cut_ratio(arr), 0.40)

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

    def test_merged_edge_slivers_are_wrap_cut(self) -> None:
        arr = np.full((240, 240, 3), 245, dtype=np.uint8)
        for cx, cy in ((70, 70), (170, 70), (70, 170), (170, 170)):
            cv2.circle(arr, (cx, cy), 28, (30, 30, 180), -1)
        arr[100:168, :11] = (30, 30, 180)
        arr[100:168, -9:] = (30, 30, 180)
        self.assertGreater(wrap_cut_ratio(arr), 0.40)

    def test_tiny_wrap_specks_do_not_floor_wrap_cut(self) -> None:
        """補花後接縫上兩粒碎點環面相接，不能抬成 0.55 切圖。"""
        arr = np.full((320, 320, 3), 245, dtype=np.uint8)
        for cx, cy in ((90, 90), (230, 90), (90, 230), (230, 230)):
            cv2.circle(arr, (cx, cy), 36, (30, 30, 180), -1)
        cv2.circle(arr, (0, 160), 36, (30, 30, 180), -1)
        cv2.circle(arr, (319, 160), 36, (30, 30, 180), -1)
        arr[40:48, :4] = (30, 30, 180)
        arr[40:48, -4:] = (30, 30, 180)
        self.assertLessEqual(wrap_cut_ratio(arr), 0.40, wrap_cut_ratio(arr))

    def test_crowded_wrap_gutter_is_mismatch(self) -> None:
        period, radius, cells = 64, 16, 4
        n = period * cells
        arr = np.full((n, n, 3), (20, 140, 70), dtype=np.uint8)
        for y in range(period // 2, n, period):
            for x in range(period // 2, n, period):
                cv2.circle(arr, (x, y), radius, (240, 240, 240), -1)
        self.assertLessEqual(wrap_gutter_error(arr), 0.20)
        bad = arr[:, 6 : n - 6]
        self.assertGreater(wrap_gutter_error(bad), GUTTER_ERR_MAX)
        src = source_facts(arr)
        cand = Candidate(
            bad,
            "週期裁切",
            True,
            [("crop", (0, 6, n, n - 12))],
        )
        measure(src, cand)
        self.assertTrue(
            any(e.startswith("接縫錯格") for e in cand.errors),
            cand.errors,
        )

    def test_crop_width_not_multiple_of_period_is_mismatch(self) -> None:
        period, radius, cells = 64, 16, 4
        n = period * cells
        arr = np.full((n, n, 3), (20, 140, 70), dtype=np.uint8)
        for y in range(period // 2, n, period):
            for x in range(period // 2, n, period):
                cv2.circle(arr, (x, y), radius, (240, 240, 240), -1)
        self.assertLessEqual(wrap_period_remainder(arr), 0.05)
        bad = arr[:, : n - 28]
        self.assertGreater(wrap_period_remainder(bad), 0.12)
        src = source_facts(arr)
        cand = Candidate(
            bad,
            "週期裁切",
            True,
            [("crop", (0, 0, n, n - 28))],
        )
        measure(src, cand)
        self.assertTrue(
            any(e.startswith("接縫錯格") for e in cand.errors),
            cand.errors,
        )

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

    def test_dense_separated_disks_are_refilled(self) -> None:
        """前景過 42% 的散開大圖章仍走清邊補花，不能未達標。"""
        n = 360
        bg = (245, 240, 220)
        fg = (40, 90, 80)
        arr = np.full((n, n, 3), bg, dtype=np.uint8)
        for y in range(55, n, 90):
            for x in range(55, n, 90):
                cv2.circle(arr, (x, y), 38, fg, -1)
        cv2.circle(arr, (0, 145), 38, fg, -1)
        cv2.circle(arr, (n - 1, 235), 38, fg, -1)
        cv2.circle(arr, (145, 0), 38, fg, -1)
        cv2.circle(arr, (235, n - 1), 38, fg, -1)
        from app.processor import _has_separated_stamps, foreground_ratio

        self.assertGreaterEqual(foreground_ratio(arr, bg, 40.0), 0.42)
        self.assertTrue(_has_separated_stamps(arr, bg, 40.0))
        unit, mode = make_seamless_hard_cut(Image.fromarray(arr), bg=bg)
        self.assertNotIn("未達標", mode, mode)
        u = np.asarray(unit.convert("RGB"))
        self.assertLessEqual(wrap_cut_ratio(u), 0.40, mode)

    def test_sliver_join_is_refilled(self) -> None:
        """對邊杏仁殘片環面假接，第二輪必須清掉再貼完整圖章。"""
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
        arr[100:180, :5] = fg
        arr[100:180, n - 5 :] = fg
        from app.processor import _clear_and_refill

        filled = _clear_and_refill(arr, bg, 40.0, 8)
        self.assertIsNotNone(filled)
        self.assertLessEqual(wrap_cut_ratio(filled), 0.40, wrap_cut_ratio(filled))
        self.assertLessEqual(wrap_orphan_run(filled), 28, wrap_orphan_run(filled))

    def test_mismatch_wrap_halves_are_refilled(self) -> None:
        """同列不同色的兩截被環面併在一起，熱點高、必須重貼。"""
        n = 280
        bg = (245, 245, 240)
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
            cv2.circle(arr, (cx, cy), 22, (40, 80, 160), -1)
        cv2.circle(arr, (0, 140), 22, (40, 80, 160), -1)
        cv2.circle(arr, (n - 1, 140), 22, (180, 40, 40), -1)
        from app.processor import _clear_and_refill

        filled = _clear_and_refill(arr, bg, 40.0, 8)
        self.assertIsNotNone(filled)
        self.assertLessEqual(wrap_cut_ratio(filled), 0.40, wrap_cut_ratio(filled))
        self.assertLessEqual(wrap_hotspot(filled), 28.0, wrap_hotspot(filled))

    def test_small_mismatch_halves_are_leftover_killed(self) -> None:
        """兩截加起來像一隻完整圖章，two_wholes 不開火，色差仍要清。"""
        n = 280
        bg = (245, 245, 240)
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
            cv2.circle(arr, (cx, cy), 22, (40, 80, 160), -1)
        cv2.circle(arr, (0, 140), 16, (40, 80, 160), -1)
        cv2.circle(arr, (n - 1, 140), 16, (180, 40, 40), -1)
        from app.quality import wrap_cut_repair

        rep = wrap_cut_repair(arr)
        self.assertGreater(len(rep.kill_ids), 0, (rep.ratio, len(rep.kill_ids)))
        from app.processor import _clear_and_refill

        filled = _clear_and_refill(arr, bg, 40.0, 8)
        self.assertIsNotNone(filled)
        self.assertLessEqual(wrap_hotspot(filled), 28.0, wrap_hotspot(filled))

    def test_true_same_color_wrap_not_color_killed(self) -> None:
        """真跨縫同色兩半，不能只因為環面合併就被色差 leftover 清掉。"""
        arr = np.full((220, 220, 3), 245, dtype=np.uint8)
        for cx, cy in ((70, 70), (150, 70), (70, 150), (150, 150)):
            cv2.circle(arr, (cx, cy), 22, (30, 30, 180), -1)
        cv2.circle(arr, (0, 110), 22, (30, 30, 180), -1)
        cv2.circle(arr, (219, 110), 22, (30, 30, 180), -1)
        from app.quality import _wrap_group_color_mismatch

        _n, labels, _stats, _c = cv2.connectedComponentsWithStats(
            (arr.mean(axis=2) < 200).astype(np.uint8), connectivity=8
        )
        left = int(labels[110, 0])
        right = int(labels[110, -1])
        self.assertTrue(left and right and left != right)
        self.assertFalse(_wrap_group_color_mismatch(arr, labels, [left, right]))

    def test_band8_unmatched_is_refilled(self) -> None:
        """只碰 8px 帶、沒進 2px 畫框的殘片也要補。"""
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
        arr[120:160, 2:7] = fg
        arr[90:130, n - 7 : n - 2] = fg
        from app.processor import _clear_and_refill

        filled = _clear_and_refill(arr, bg, 40.0, 8)
        self.assertIsNotNone(filled)
        self.assertLessEqual(wrap_cut_ratio(filled), 0.40, wrap_cut_ratio(filled))

    def test_refill_tone_ignores_wrap_band(self) -> None:
        """清邊補花改的是畫框，內部均值不該被整張通道差誤殺。"""
        from app.quality import tone_shift

        n = 240
        bg = (240, 230, 210)
        arr = np.full((n, n, 3), bg, dtype=np.uint8)
        for cx, cy in ((70, 70), (170, 70), (70, 170), (170, 170)):
            cv2.circle(arr, (cx, cy), 28, (40, 120, 90), -1)
        out = arr.copy()
        cv2.circle(out, (0, 120), 28, (40, 120, 90), -1)
        cv2.circle(out, (n - 1, 120), 28, (40, 120, 90), -1)
        self.assertGreater(tone_shift(arr, out), 6.5)
        self.assertLess(tone_shift(arr, out, edge_frac=0.20), 2.0)

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
        # wrap_cut 走 RGB ink_mask 會把細莖連成整列；這項測的是 opening 有沒有把圓拆開。

    def test_forced_edge_snaps_inward_centroid(self) -> None:
        from app.processor import MotifStamp, _snap_wrap_placement

        mask = np.ones((40, 40), dtype=bool)
        motif = MotifStamp(patch=np.zeros((40, 40, 3), dtype=np.uint8), mask=mask, area=1600, cy=20.0, cx=20.0)
        cy, cx, is_edge = _snap_wrap_placement(
            80.0, 80.0, 700, motif, 240, 240, forced_edge=True
        )
        self.assertTrue(is_edge)
        self.assertTrue(cx < 1.0 or cx > 239.0 or cy < 1.0 or cy > 239.0)

    def test_forced_edge_snaps_nearest_axis_not_corner(self) -> None:
        """左側殘片即使靠近頂部，也不能被大帶寬吸到左上角。"""
        from app.processor import MotifStamp, _snap_wrap_placement

        mask = np.ones((80, 80), dtype=bool)
        motif = MotifStamp(
            patch=np.zeros((80, 80, 3), dtype=np.uint8),
            mask=mask,
            area=6400,
            cy=40.0,
            cx=40.0,
        )
        cy, cx, is_edge = _snap_wrap_placement(
            120.0, 30.0, 800, motif, 400, 400, forced_edge=True
        )
        self.assertTrue(is_edge)
        self.assertLess(cx, 1.0)
        self.assertGreater(cy, 80.0)
        self.assertLess(cy, 160.0)

    def test_wrap_span_overlap_keeps_one(self) -> None:
        from app.processor import MotifStamp, _dedupe_wrap_jobs

        mask = np.ones((80, 40), dtype=bool)
        a = MotifStamp(
            patch=np.zeros((80, 40, 3), dtype=np.uint8),
            mask=mask,
            area=3200,
            cy=40.0,
            cx=20.0,
        )
        b = MotifStamp(
            patch=np.zeros((80, 40, 3), dtype=np.uint8),
            mask=mask.copy(),
            area=3200,
            cy=40.0,
            cx=20.0,
        )
        kept = _dedupe_wrap_jobs(
            [(100.0, 0.0, a), (134.0, 0.0, b)], 280, 280
        )
        self.assertEqual(len(kept), 1)

    def test_space_wrap_jobs_drops_opposite_edge_pile(self) -> None:
        """左右殘片吸到同一條縫後，間距要比照圖章尺寸，不能並排硬貼。"""
        from app.processor import MotifStamp, _space_wrap_jobs

        mask = np.ones((50, 50), dtype=bool)
        m = MotifStamp(
            patch=np.zeros((50, 50, 3), dtype=np.uint8),
            mask=mask,
            area=2500,
            cy=25.0,
            cx=25.0,
        )
        kept = _space_wrap_jobs(
            [(80.0, 0.0, m), (95.0, 0.0, m), (200.0, 0.0, m)],
            360,
            360,
        )
        self.assertEqual(len(kept), 2)
        ys = sorted(j[0] for j in kept)
        self.assertAlmostEqual(ys[0], 80.0)
        self.assertAlmostEqual(ys[1], 200.0)

    def test_opposite_edges_do_not_double_wrap_density(self) -> None:
        """左右兩邊各一排半圓，補花後接縫帶不能比內部密一倍。"""
        n = 360
        bg = (245, 245, 240)
        fg = (40, 80, 160)
        arr = np.full((n, n, 3), bg, dtype=np.uint8)
        for y in range(55, n, 90):
            for x in range(55, n, 90):
                cv2.circle(arr, (x, y), 20, fg, -1)
        for y in range(55, n, 90):
            cv2.circle(arr, (3, y), 20, fg, -1)
            cv2.circle(arr, (n - 4, y), 20, fg, -1)
        from app.processor import _clear_and_refill, _foreground_mask

        filled = _clear_and_refill(arr, bg, 40.0, 8)
        self.assertIsNotNone(filled)
        from app.quality import wrap_density_ratio

        crowd = wrap_density_ratio(filled)
        self.assertLessEqual(crowd, 1.15, crowd)
        self.assertLessEqual(wrap_cut_ratio(filled), 0.40, wrap_cut_ratio(filled))

    def test_large_edge_blobs_snap_across_wrap(self) -> None:
        """大雪人停在 8px wrap 帶裡：清邊必須整隻拿走並跨縫貼回。"""
        n = 400
        bg = (220, 40, 40)
        fg = (250, 250, 250)
        arr = np.full((n, n, 3), bg, dtype=np.uint8)
        for cy in (90, 200, 310):
            for cx in (90, 200, 310):
                cv2.circle(arr, (cx, cy), 36, fg, -1)
        cv2.circle(arr, (6, 145), 36, fg, -1)
        cv2.circle(arr, (n - 7, 255), 36, fg, -1)
        cv2.circle(arr, (165, 5), 36, fg, -1)
        from app.processor import _clear_and_refill

        filled = _clear_and_refill(arr, bg, 40.0, 8)
        self.assertIsNotNone(filled)
        self.assertLessEqual(wrap_cut_ratio(filled), 0.40, wrap_cut_ratio(filled))
        self.assertLessEqual(wrap_hotspot(filled), 28.0, wrap_hotspot(filled))
        unit, mode = make_seamless_hard_cut(Image.fromarray(arr), bg=bg)
        self.assertNotIn("未達標", mode, mode)


class MotifGroupTests(unittest.TestCase):
    def test_wrapping_paw_is_not_wrap_cut(self) -> None:
        """掌在左緣、趾在右緣：群組後是一枚跨縫圖章，不是切圖。"""
        n = 320
        bg = (248, 244, 236)
        arr = np.full((n, n, 3), bg, dtype=np.uint8)
        pad = (40, 140, 200)
        toes = (220, 90, 40)

        def stamp_paw(cx: int, cy: int) -> None:
            cv2.circle(arr, (cx, cy), 22, pad, -1)
            for dx, dy in ((-20, -26), (-6, -30), (10, -28), (22, -16)):
                cv2.circle(arr, ((cx + dx) % n, cy + dy), 7, toes, -1)

        for cx, cy in ((100, 100), (220, 100), (100, 220), (220, 220)):
            stamp_paw(cx, cy)
        stamp_paw(8, 160)
        self.assertLessEqual(wrap_cut_ratio(arr), 0.40, wrap_cut_ratio(arr))

    def test_paw_toes_group_with_pad(self) -> None:
        from app.motif_layout import group_components_into_motifs
        from app.processor import _foreground_mask

        n = 320
        bg = (248, 244, 236)
        arr = np.full((n, n, 3), bg, dtype=np.uint8)
        pad = (40, 140, 200)
        toes = (220, 90, 40)
        cx, cy = 160, 160
        cv2.circle(arr, (cx, cy), 22, pad, -1)
        for dx, dy in ((-20, -26), (-6, -30), (10, -28), (22, -16)):
            cv2.circle(arr, (cx + dx, cy + dy), 7, toes, -1)
        fg = _foreground_mask(arr, bg, 40.0)
        _gmap, groups, _l, _s, _c = group_components_into_motifs(fg, 40)
        self.assertEqual(len(groups), 1, [g.area for g in groups])
        self.assertGreater(groups[0].area, 800)

    def test_jittered_polka_is_regular_lattice(self) -> None:
        from app.discrete_lattice import looks_like_regular_lattice

        n = 360
        bg = (250, 250, 250)
        arr = np.full((n, n, 3), bg, dtype=np.uint8)
        rng = np.random.default_rng(3)
        for y in range(40, n, 48):
            for x in range(40, n, 48):
                jx = int(rng.integers(-3, 4))
                jy = int(rng.integers(-3, 4))
                cv2.circle(arr, (x + jx, y + jy), 8, (20, 40, 90), -1)
        self.assertTrue(looks_like_regular_lattice(arr, bg, 40.0))

    def test_noisy_bg_refill_has_no_flat_ghost(self) -> None:
        n = 280
        rng = np.random.default_rng(1)
        noise = rng.integers(236, 248, size=(n, n, 3), dtype=np.uint8)
        arr = noise.copy()
        for y in range(50, n, 70):
            for x in range(50, n, 70):
                cv2.circle(arr, (x, y), 14, (30, 70, 160), -1)
        for y in range(50, n, 70):
            cv2.circle(arr, (2, y), 14, (30, 70, 160), -1)
        from app.processor import _clear_and_refill

        bg = (242, 242, 242)
        filled = _clear_and_refill(arr, bg, 40.0, 8)
        self.assertIsNotNone(filled)
        # 清掉的外圈不該是死平色圓盤（相對原稿背景雜訊）
        band = np.zeros((n, n), dtype=bool)
        band[:, :8] = True
        band[:, n - 8 :] = True
        std = float(filled[band].reshape(-1, 3).std())
        self.assertGreater(std, 1.5, std)

    def test_source_period_remainder_rejects_direct_out(self) -> None:
        from app.select import PERIOD_REM_SOURCE_MAX, source_looks_seamless

        n = 220
        p = 48
        arr = np.full((n, n, 3), (40, 160, 90), dtype=np.uint8)
        for y in range(20, n, p):
            for x in range(20, n, p):
                cv2.circle(arr, (x, y), 10, (250, 250, 250), -1)
        src = source_facts(arr)
        self.assertGreater(src.period_rem, PERIOD_REM_SOURCE_MAX)
        self.assertFalse(
            source_looks_seamless(
                src.rep,
                src.hotspot,
                period_rem=src.period_rem,
            )
        )


class FineGridAlignTests(unittest.TestCase):
    def _gingham(self, n: int, cell: int, phase: int = 0) -> np.ndarray:
        arr = np.full((n, n, 3), 245, dtype=np.uint8)
        yy, xx = np.indices((n, n))
        chk = (((xx + phase) // cell + (yy + phase) // cell) % 2) == 0
        arr[chk] = (90, 55, 35)
        return arr

    def test_luma_detects_square_cell(self) -> None:
        from app.processor import _luma_square_grid_pitch

        arr = self._gingham(400, 20, phase=7)
        pitch = _luma_square_grid_pitch(arr)
        self.assertIsNotNone(pitch)
        px, py = pitch
        self.assertTrue(min(abs(px - 20), abs(px - 40)) <= 3, pitch)
        self.assertTrue(min(abs(py - 20), abs(py - 40)) <= 3, pitch)

    def test_period_crop_keeps_grid_in_phase(self) -> None:
        from app.processor import try_period_crop
        from app.quality import wrap_gutter_error, wrap_period_remainder
        from app.select import Candidate, measure, source_facts

        arr = self._gingham(400, 20, phase=7)
        tile, detail = try_period_crop(arr)
        self.assertIsNotNone(tile, detail)
        self.assertEqual(tile.shape[0] % 20, 0, (tile.shape, detail))
        self.assertEqual(tile.shape[1] % 20, 0, (tile.shape, detail))
        self.assertLessEqual(wrap_period_remainder(tile), 0.08, detail)
        self.assertLessEqual(wrap_gutter_error(tile), 0.12, detail)
        src = source_facts(arr)
        cand = Candidate(
            tile,
            "週期裁切",
            True,
            [("crop", (0, 0, tile.shape[0], tile.shape[1]))],
        )
        measure(src, cand)
        self.assertFalse(
            any(e.startswith("結構接縫") for e in cand.errors),
            cand.errors,
        )

    def test_downsampled_gingham_keeps_fundamental(self) -> None:
        from app.processor import _luma_square_grid_pitch, try_period_crop
        from app.quality import wrap_period_remainder

        # 大於 512 會縮圖；3 格諧波（96）不能贏過 32／64px 基本格。
        arr = self._gingham(640, 32, phase=5)
        pitch = _luma_square_grid_pitch(arr)
        self.assertIsNotNone(pitch)
        px, py = pitch
        self.assertTrue(min(abs(px - 32), abs(px - 64)) <= 3, pitch)
        self.assertTrue(min(abs(py - 32), abs(py - 64)) <= 3, pitch)
        tile, detail = try_period_crop(arr)
        self.assertIsNotNone(tile, detail)
        self.assertEqual(tile.shape[0] % 32, 0, (tile.shape, detail))
        self.assertEqual(tile.shape[1] % 32, 0, (tile.shape, detail))
        self.assertLessEqual(wrap_period_remainder(tile), 0.08, detail)

    def test_off_size_gingham_crops_to_cells(self) -> None:
        from app.processor import try_period_crop
        from app.quality import wrap_gutter_error, wrap_period_remainder

        arr = self._gingham(650, 32, phase=5)
        tile, detail = try_period_crop(arr)
        self.assertIsNotNone(tile, detail)
        step = 32
        self.assertEqual(tile.shape[0] % step, 0, (tile.shape, detail))
        self.assertEqual(tile.shape[1] % step, 0, (tile.shape, detail))
        self.assertLessEqual(wrap_period_remainder(tile), 0.08, detail)
        self.assertLessEqual(wrap_gutter_error(tile), 0.12, detail)

    def test_high_ink_gingham_is_discrete(self) -> None:
        from app.processor import _looks_like_discrete_motifs

        arr = self._gingham(360, 18)
        self.assertTrue(_looks_like_discrete_motifs(arr, (245, 245, 245), 40.0))

    def test_watercolor_halo_refill_is_kept(self) -> None:
        """紙紋上的淺色描邊不能被當成補花毀圖。"""
        n = 280
        rng = np.random.default_rng(4)
        arr = rng.integers(236, 248, size=(n, n, 3), dtype=np.uint8)
        fg = (90, 50, 140)
        halo = (252, 252, 252)
        for y in range(48, n, 72):
            for x in range(48, n, 72):
                cv2.circle(arr, (x, y), 18, halo, 4)
                cv2.circle(arr, (x, y), 14, fg, -1)
        for y in range(48, n, 72):
            cv2.circle(arr, (2, y), 18, halo, 4)
            cv2.circle(arr, (2, y), 14, fg, -1)
        from app.processor import _clear_and_refill

        filled = _clear_and_refill(arr, (242, 242, 242), 40.0, 8)
        self.assertIsNotNone(filled)


if __name__ == "__main__":
    unittest.main()
