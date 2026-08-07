"""背景色取樣與色差判斷。"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from PIL import Image


def sample_corner_background(
    image: Image.Image,
    sample_size: int = 8,
) -> tuple[int, int, int]:
    """從四角取樣中位數作為背景色 (RGB)。"""
    rgb = image.convert("RGB")
    arr = np.asarray(rgb, dtype=np.int32)
    h, w = arr.shape[:2]
    s = max(1, min(sample_size, h // 2, w // 2))

    patches = [
        arr[0:s, 0:s],
        arr[0:s, w - s : w],
        arr[h - s : h, 0:s],
        arr[h - s : h, w - s : w],
    ]
    samples = np.concatenate([p.reshape(-1, 3) for p in patches], axis=0)
    median = np.median(samples, axis=0).astype(np.int32)
    return int(median[0]), int(median[1]), int(median[2])


def _median_of_samples(samples: np.ndarray) -> tuple[int, int, int]:
    rough = np.median(samples, axis=0)
    dist = np.sqrt(np.sum((samples - rough) ** 2, axis=1))
    cutoff = np.percentile(dist, 70)
    keep = samples[dist <= max(cutoff, 1.0)]
    if len(keep) < 10:
        keep = samples
    median = np.median(keep, axis=0).astype(np.int32)
    return int(median[0]), int(median[1]), int(median[2])


def detect_majority_background(image: Image.Image, top_k: int = 3) -> tuple[int, int, int]:
    """
    取低梯度區域裡偏亮像素的中位色。
    豹紋等黑斑可佔大面積時，平坦區偏亮部分才是布面底色。
    """
    del top_k
    rgb = image.convert("RGB")
    side = max(80, min(160, max(rgb.size) // 6))
    small = rgb.resize((side, side), Image.Resampling.BILINEAR)
    arr = np.asarray(small, dtype=np.float64)
    g = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    gy = np.abs(np.diff(g, axis=0, prepend=g[:1]))
    gx = np.abs(np.diff(g, axis=1, prepend=g[:, :1]))
    grad = gx + gy
    flat = grad <= float(np.percentile(grad, 45))
    if int(np.count_nonzero(flat)) < 30:
        flat = np.ones_like(g, dtype=bool)
    g_flat = g[flat]
    # 平坦區裡丟掉最暗的斑點殘留，取偏亮側
    lo = float(np.percentile(g_flat, 40))
    keep = flat & (g >= lo)
    if int(np.count_nonzero(keep)) < 20:
        keep = flat
    med = np.median(arr[keep], axis=0).astype(np.int32)
    return int(med[0]), int(med[1]), int(med[2])


def detect_background(
    image: Image.Image,
    border: int = 6,
) -> tuple[int, int, int]:
    """
    自動識別背景色：
    1) 四邊邊緣帶去離群中位數
    2) 全圖多數色
    取「與圖面平均色距較小、且邊緣出現較多」者，避免豹紋角落黑點把底色吸成黑色。
    """
    rgb = image.convert("RGB")
    arr = np.asarray(rgb, dtype=np.float64)
    h, w = arr.shape[:2]
    b = max(1, min(border, h // 4, w // 4))

    strips = [
        arr[:b, :, :],
        arr[-b:, :, :],
        arr[:, :b, :],
        arr[:, -b:, :],
    ]
    edge_samples = np.concatenate([s.reshape(-1, 3) for s in strips], axis=0)
    edge_bg = _median_of_samples(edge_samples)
    maj_bg = detect_majority_background(image)

    def score(bg: tuple[int, int, int]) -> float:
        bg_a = np.asarray(bg, dtype=np.float64)
        dist = np.sqrt(np.sum((edge_samples - bg_a) ** 2, axis=1))
        edge_hit = float(np.mean(dist < 45.0))
        small = np.asarray(
            rgb.resize((64, 64), Image.Resampling.BILINEAR), dtype=np.float64
        ).reshape(-1, 3)
        d2 = np.sqrt(np.sum((small - bg_a) ** 2, axis=1))
        near = float(np.mean(d2 < 50.0))
        balance = 1.0 - abs(near - 0.55) * 1.2
        return edge_hit * 2.0 + max(0.0, balance) + near * 0.5

    maj_lum = 0.299 * maj_bg[0] + 0.587 * maj_bg[1] + 0.114 * maj_bg[2]
    edge_lum = 0.299 * edge_bg[0] + 0.587 * edge_bg[1] + 0.114 * edge_bg[2]
    # 邊緣底色過暗（常被黑斑污染）且平坦區色明顯更亮 → 用平坦區色（豹紋）
    if edge_lum < 40 and maj_lum > edge_lum + 25:
        return maj_bg
    if score(maj_bg) >= score(edge_bg) * 0.92:
        return maj_bg
    return edge_bg


def color_distance(pixel: np.ndarray, bg: Sequence[int]) -> np.ndarray:
    """計算每個像素與背景色的歐氏距離。"""
    bg_arr = np.asarray(bg, dtype=np.float64)
    diff = pixel.astype(np.float64) - bg_arr
    return np.sqrt(np.sum(diff * diff, axis=-1))


def rgb_to_hex(rgb: Sequence[int]) -> str:
    return f"#{int(rgb[0]):02x}{int(rgb[1]):02x}{int(rgb[2]):02x}"


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    text = value.strip().lstrip("#")
    if len(text) != 6:
        raise ValueError(f"無效的色碼: {value}")
    return int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16)
