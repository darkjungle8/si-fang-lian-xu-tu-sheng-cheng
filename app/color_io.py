"""色彩空間的進出：CMYK 進、CMYK 出，原始 ICC 一路帶著走。

這批稿件有 132／199 張是印刷用的 CMYK JPEG。舊流程在 `to_srgb` 把它們
轉成 sRGB 之後就把 profile 丟了，中間暫存的 PNG、最終的 TIFF／PNG／JPEG
全都是無標記 RGB。對印刷來說這本身就是一次色偏，而且是不可逆的。

處理過程仍在 sRGB 進行：接縫判斷、週期偵測、圖種分類都是為 RGB 語意寫
的，硬改成四通道風險太高。改成在**出口**用原始 profile 轉回去，並把
profile 嵌回檔案。

`ColorContext` 就是為此在讀檔當下記下來的那點資訊，跟著圖走到存檔為止。
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

# 轉回印刷色空間時保留相對明度關係，並補償黑點；預設的感知意圖會讓整體
# 密度往下掉一截。
_INTENT_RELATIVE = 1


@dataclass(frozen=True)
class ColorContext:
    """讀檔當下的色彩身分。處理完要靠它把圖送回原本的空間。"""

    mode: str
    icc: bytes | None
    dpi: tuple[float, float] | None

    @property
    def needs_restore(self) -> bool:
        return self.mode in ("CMYK", "L") or self.icc is not None

    def describe(self) -> str:
        return f"{self.mode}{'+ICC' if self.icc else ''}"


def context_of(image: Image.Image) -> ColorContext:
    dpi = image.info.get("dpi")
    if dpi is not None:
        try:
            dpi = (float(dpi[0]), float(dpi[1]))
        except (TypeError, ValueError, IndexError):
            dpi = None
    return ColorContext(
        mode=image.mode, icc=image.info.get("icc_profile"), dpi=dpi
    )


def _profile_transform(
    src_icc: bytes | None,
    dst_icc: bytes | None,
    in_mode: str,
    out_mode: str,
):
    from PIL import ImageCms

    def _load(icc: bytes | None):
        if icc is None:
            return ImageCms.createProfile("sRGB")
        return ImageCms.ImageCmsProfile(io.BytesIO(icc))

    flags = 0
    try:
        flags = ImageCms.Flags.BLACKPOINTCOMPENSATION
    except AttributeError:
        flags = ImageCms.FLAGS.get("BLACKPOINTCOMPENSATION", 0)
    return ImageCms.buildTransform(
        _load(src_icc),
        _load(dst_icc),
        in_mode,
        out_mode,
        renderingIntent=_INTENT_RELATIVE,
        flags=flags,
    )


def restore(rgb: Image.Image, ctx: ColorContext) -> Image.Image:
    """
    把處理完的 sRGB 圖送回原始色彩空間，並把原始 profile 嵌回去。

    轉不動就原樣回傳 sRGB——輸出一張顏色大致正確的圖，永遠好過因為色彩
    管理失敗而讓整批中斷。
    """
    if rgb.mode != "RGB" or not ctx.needs_restore:
        out = rgb
    elif ctx.mode == "CMYK":
        out = None
        if ctx.icc is not None:
            try:
                from PIL import ImageCms

                out = ImageCms.applyTransform(
                    rgb, _profile_transform(None, ctx.icc, "RGB", "CMYK")
                )
            except Exception:  # noqa: BLE001 — 色彩管理失敗不該中斷批次
                out = None
        if out is None:
            # 原稿沒有內嵌 profile 時，來回都走 PIL 內建公式，至少是對稱的
            out = rgb.convert("CMYK")
    elif ctx.mode == "L":
        out = rgb.convert("L")
    elif ctx.icc is not None:
        try:
            from PIL import ImageCms

            out = ImageCms.applyTransform(
                rgb, _profile_transform(None, ctx.icc, "RGB", "RGB")
            )
        except Exception:  # noqa: BLE001
            out = rgb
    else:
        out = rgb

    if ctx.icc is not None and out.mode == ctx.mode:
        out.info["icc_profile"] = ctx.icc
    if ctx.dpi is not None:
        out.info["dpi"] = ctx.dpi
    return out


def save_kwargs(image: Image.Image, dest: Path) -> dict:
    """依副檔名組出存檔參數，並把 ICC 與 DPI 帶上。"""
    kw: dict = {}
    icc = image.info.get("icc_profile")
    if icc:
        kw["icc_profile"] = icc
    dpi = image.info.get("dpi")
    if dpi:
        kw["dpi"] = dpi
    suffix = dest.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        kw["quality"] = 95
        kw["subsampling"] = 0
    elif suffix in {".tif", ".tiff"}:
        kw["compression"] = "tiff_lzw"
    return kw


def save_image(image: Image.Image, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    image.save(dest, **save_kwargs(image, dest))


def intermediate_suffix(image: Image.Image) -> str:
    """
    中間暫存檔該用什麼格式。

    PNG 存不了 CMYK，也存不了四通道以外的印刷資料；一律改用無損 TIFF，
    才不會在流水線中途把色彩空間降級。
    """
    return ".png" if image.mode in ("RGB", "RGBA", "L", "P") else ".tif"
