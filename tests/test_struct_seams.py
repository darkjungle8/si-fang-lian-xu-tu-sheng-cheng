# -*- coding: utf-8 -*-
"""結構接縫：wrap 色差為 0 仍可能在 2×2 裂開。"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.processor import _compact_period_jobs, _luminance_map
from app.quality import wrap_hotspot
from app.select import Candidate, MOTIF_CUT_MAX, measure, source_facts


class WrapHotspotTests(unittest.TestCase):
    def test_flat_field_is_cold(self) -> None:
        arr = np.full((80, 80, 3), 240, dtype=np.uint8)
        self.assertLess(wrap_hotspot(arr), 1.0)

    def test_sparse_stamp_mismatch_is_hot(self) -> None:
        arr = np.full((120, 120, 3), 250, dtype=np.uint8)
        arr[40:80, :10] = (180, 20, 20)
        arr[40:80, -10:] = (20, 40, 180)
        self.assertGreater(wrap_hotspot(arr), 14.0)


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
        self.assertLess(src.ink, 0.40)
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


if __name__ == "__main__":
    unittest.main()
