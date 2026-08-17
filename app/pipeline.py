"""流水線：四方連續單元圖 → 平鋪擴展 → 裁切 → 白邊黑邊 → 匯出。"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PIL import Image

from app.color_io import intermediate_suffix, save_image
from app.color_utils import detect_background
from app.paths import ensure_kuotu_on_path
from app.processor import make_seamless_hard_cut, tile_2x2_multi

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
SKIP_DIR_NAMES = frozenset({"output", "pipeline_out", "__pycache__"})

EXPAND_FORMATS = {
    "TIFF": ".tif",
    "PNG": ".png",
    "JPEG": ".jpg",
}

LogFn = Callable[[str], None]
ProgressFn = Callable[[float], None]


def _path_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def collect_folder_images(
    root: Path,
    *,
    recursive: bool = True,
    exclude_roots: list[Path] | None = None,
) -> list[Path]:
    """收集資料夾內圖片；遞迴時略過 output／pipeline_out 等目錄。"""
    root = root.resolve()
    excludes = [p.resolve() for p in (exclude_roots or [])]
    pattern = "**/*" if recursive else "*"
    files: list[Path] = []
    for p in sorted(root.glob(pattern)):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        try:
            rel = p.resolve().relative_to(root)
        except ValueError:
            continue
        if any(part in SKIP_DIR_NAMES for part in rel.parts[:-1]):
            continue
        if any(_path_under(p, ex) for ex in excludes):
            continue
        files.append(p)
    return files


def mirror_dest(src: Path, src_root: Path, out_root: Path, *, name: str | None = None) -> Path:
    """依 src 相對 src_root 的路徑，鏡射到 out_root；可覆寫檔名。"""
    try:
        rel = src.resolve().relative_to(src_root.resolve())
    except ValueError:
        rel = Path(src.name)
    dest_name = name if name is not None else rel.name
    return out_root / rel.parent / dest_name


def expand_output_name(src: Path, expand_ext: str) -> str:
    """
    批次擴圖檔名：保留來源副檔名，避免 2 (21).jpg 與 2 (21).png
    都變成 2 (21).tif 而互相覆蓋。
    例：2 (21).jpg + .tif → 2 (21).jpg.tif
    """
    ext = expand_ext if expand_ext.startswith(".") else f".{expand_ext}"
    return f"{src.name}{ext}"


@dataclass
class ExpandSettings:
    dpi: float = 300.0
    target_w_cm: float = 100.0
    target_h_cm: float = 100.0
    crop_x: int = 0
    crop_y: int = 0
    crop_w_cm: float = 30.0
    crop_h_cm: float = 30.0
    white_cm: float = 0.2
    black_cm: float = 0.1
    force_dpi: bool = False


@dataclass
class PipelineItemResult:
    source: str
    unit_path: str | None
    preview_path: str | None
    expand_path: str | None
    mode: str
    error: str | None = None
    sku: str | None = None
    crop_w_cm: float | None = None
    crop_h_cm: float | None = None

    @property
    def tiff_path(self) -> str | None:
        """相容舊欄位名稱。"""
        return self.expand_path


def normalize_expand_ext(fmt: str) -> str:
    """回傳標準副檔名，例如 '.tif' / '.png' / '.jpg'。"""
    key = fmt.strip().upper().lstrip(".")
    aliases = {
        "TIF": "TIFF",
        "TIFF": "TIFF",
        "PNG": "PNG",
        "JPG": "JPEG",
        "JPEG": "JPEG",
    }
    canon = aliases.get(key)
    if canon is None:
        raise ValueError(f"不支援的擴圖格式：{fmt}（請用 TIFF / PNG / JPEG）")
    return EXPAND_FORMATS[canon]


def expand_unit(
    unit_path: Path,
    dest: Path,
    settings: ExpandSettings,
    log: LogFn = print,
) -> None:
    """呼叫 kuotu：平鋪擴展 → 裁切 → 白邊黑邊 → 依 dest 副檔名匯出。"""
    ensure_kuotu_on_path()
    from image_pipeline import process_one  # noqa: WPS433

    process_one(
        unit_path,
        dest,
        fallback_dpi=settings.dpi,
        target_w_cm=settings.target_w_cm,
        target_h_cm=settings.target_h_cm,
        crop_x=settings.crop_x,
        crop_y=settings.crop_y,
        crop_w_cm=settings.crop_w_cm,
        crop_h_cm=settings.crop_h_cm,
        white_cm=settings.white_cm,
        black_cm=settings.black_cm,
        use_image_dpi=not settings.force_dpi,
        log=log,
    )


def expand_unit_to_tiff(
    unit_path: Path,
    dest: Path,
    settings: ExpandSettings,
    log: LogFn = print,
) -> None:
    """相容舊名稱；實際依 dest 副檔名輸出。"""
    expand_unit(unit_path, dest, settings, log=log)


def _settings_with_crop(
    base: ExpandSettings,
    crop_w_cm: float,
    crop_h_cm: float,
) -> ExpandSettings:
    """依裁切尺寸調整擴展畫布（至少蓋住裁切框）。"""
    return ExpandSettings(
        dpi=base.dpi,
        target_w_cm=max(base.target_w_cm, crop_w_cm),
        target_h_cm=max(base.target_h_cm, crop_h_cm),
        crop_x=base.crop_x,
        crop_y=base.crop_y,
        crop_w_cm=crop_w_cm,
        crop_h_cm=crop_h_cm,
        white_cm=base.white_cm,
        black_cm=base.black_cm,
        force_dpi=base.force_dpi,
    )


def _process_one_image(
    path: Path,
    expand_dest: Path | None,
    *,
    margin: float,
    margin_is_percent: bool,
    threshold: float,
    auto_bg: bool,
    manual_bg: tuple[int, int, int],
    expand: ExpandSettings,
    do_expand: bool,
    log: LogFn,
) -> PipelineItemResult:
    """生成單元圖後直接擴圖；中間產物不落盤。"""
    item = PipelineItemResult(
        source=str(path),
        unit_path=None,
        preview_path=None,
        expand_path=None,
        mode="",
        crop_w_cm=expand.crop_w_cm if do_expand else None,
        crop_h_cm=expand.crop_h_cm if do_expand else None,
    )
    img = Image.open(path)
    img.load()
    use_bg = None if auto_bg else manual_bg
    if use_bg is None:
        use_bg = detect_background(img)

    unit, mode = make_seamless_hard_cut(
        img,
        bg=use_bg,
        margin=margin,
        threshold=threshold,
        margin_is_percent=margin_is_percent,
        log=log,
    )
    # 仍跑 2×2 僅為模式說明；不寫入磁碟
    _, preview_detail, _ = tile_2x2_multi(unit)
    item.mode = f"{mode}；{preview_detail}"

    if do_expand:
        if expand_dest is None:
            raise ValueError("擴圖輸出路徑未指定")
        expand_dest.parent.mkdir(parents=True, exist_ok=True)
        # 中間檔跟著單元圖的色彩空間走：PNG 存不了 CMYK，硬存會把印刷稿
        # 降級成 RGB，等於在流水線中途製造色偏
        with tempfile.NamedTemporaryFile(
            suffix=intermediate_suffix(unit), delete=False
        ) as tmp:
            tmp_path = Path(tmp.name)
        try:
            save_image(unit, tmp_path)
            expand_unit(tmp_path, expand_dest, expand, log=log)
            item.expand_path = str(expand_dest)
        finally:
            tmp_path.unlink(missing_ok=True)
    else:
        # 未開擴圖時，最終結果就是單元圖
        if expand_dest is None:
            raise ValueError("輸出路徑未指定")
        save_image(unit, expand_dest)
        item.unit_path = str(expand_dest)
        item.expand_path = str(expand_dest)

    return item


def run_full_pipeline(
    input_dir: Path,
    *,
    output_dir: Path | None = None,
    margin: float = 0.0,
    margin_is_percent: bool = True,
    threshold: float = 40.0,
    auto_bg: bool = True,
    manual_bg: tuple[int, int, int] = (255, 255, 255),
    expand: ExpandSettings | None = None,
    do_expand: bool = True,
    expand_format: str = "TIFF",
    excel_path: Path | None = None,
    log: LogFn = print,
    on_progress: ProgressFn | None = None,
) -> list[PipelineItemResult]:
    """
    對資料夾內圖片跑完整流程，只輸出最終結果（不寫 units／previews）。

    - 一般模式：遞迴處理 input_dir 內圖片，輸出保留相對資料夾結構
    - SKU 模式（提供 excel_path）：第一層子資料夾＝SKU，其內可再有巢狀目錄
    """
    if do_expand:
        ensure_kuotu_on_path()

    expand = expand or ExpandSettings()
    expand_ext = normalize_expand_ext(expand_format) if do_expand else ".png"
    input_dir = input_dir.resolve()
    out = (output_dir or (input_dir / "pipeline_out")).resolve()
    out.mkdir(parents=True, exist_ok=True)

    if excel_path is not None:
        return _run_sku_pipeline(
            input_dir,
            excel_path.resolve(),
            out,
            margin=margin,
            margin_is_percent=margin_is_percent,
            threshold=threshold,
            auto_bg=auto_bg,
            manual_bg=manual_bg,
            expand=expand,
            do_expand=do_expand,
            expand_ext=expand_ext,
            log=log,
            on_progress=on_progress,
        )

    files = collect_folder_images(input_dir, recursive=True, exclude_roots=[out])
    if not files:
        raise ValueError(f"資料夾內沒有支援的圖片：{input_dir}")

    log(f"輸入：{input_dir}")
    log(f"輸出：{out}")
    fmt_label = expand_ext.lstrip(".").upper() if do_expand else "單元圖"
    log(f"共 {len(files)} 張 | 最終輸出={fmt_label}（保留子資料夾結構）")

    results: list[PipelineItemResult] = []
    total = len(files)

    for i, path in enumerate(files):
        dest = mirror_dest(
            path, input_dir, out, name=expand_output_name(path, expand_ext)
        )
        if dest.exists():
            try:
                rel_label = str(path.relative_to(input_dir))
            except ValueError:
                rel_label = path.name
            log(f"  [{i + 1}/{total}] {rel_label} 跳過（輸出已存在）")
            results.append(
                PipelineItemResult(
                    source=str(path),
                    unit_path=None,
                    preview_path=None,
                    expand_path=str(dest),
                    mode="跳過（已存在）",
                )
            )
            if on_progress is not None:
                on_progress((i + 1) / total)
            continue
        try:
            item = _process_one_image(
                path,
                dest,
                margin=margin,
                margin_is_percent=margin_is_percent,
                threshold=threshold,
                auto_bg=auto_bg,
                manual_bg=manual_bg,
                expand=expand,
                do_expand=do_expand,
                log=log,
            )
            try:
                rel_label = str(path.relative_to(input_dir))
            except ValueError:
                rel_label = path.name
            log(f"  [{i + 1}/{total}] {rel_label} OK")
        except Exception as exc:  # noqa: BLE001 — 批次繼續
            item = PipelineItemResult(
                source=str(path),
                unit_path=None,
                preview_path=None,
                expand_path=None,
                mode="",
                error=str(exc),
            )
            log(f"  [{i + 1}/{total}] {path.name} 失敗：{exc}")

        results.append(item)
        if on_progress is not None:
            on_progress((i + 1) / total)

    ok_n = sum(1 for r in results if not r.error)
    log(f"\n完成：成功 {ok_n}/{total} → {out}")
    return results


def _run_sku_pipeline(
    parent_dir: Path,
    excel_path: Path,
    out: Path,
    *,
    margin: float,
    margin_is_percent: bool,
    threshold: float,
    auto_bg: bool,
    manual_bg: tuple[int, int, int],
    expand: ExpandSettings,
    do_expand: bool,
    expand_ext: str,
    log: LogFn,
    on_progress: ProgressFn | None,
) -> list[PipelineItemResult]:
    """第一層子資料夾 = SKU；其內遞迴讀圖，輸出保留 SKU 資料夾名與相對結構。"""
    ensure_kuotu_on_path()
    from image_pipeline import (  # noqa: WPS433
        extract_sku_from_folder_name,
        load_sku_catalog,
    )

    if not parent_dir.is_dir():
        raise ValueError(f"SKU 模式輸入必須是父資料夾：{parent_dir}")

    sku_catalog = load_sku_catalog(excel_path)
    sku_sizes = sku_catalog.sizes
    sku_dirs = sorted(
        p
        for p in parent_dir.iterdir()
        if p.is_dir() and p.name not in SKIP_DIR_NAMES and not _path_under(p, out)
    )
    if not sku_dirs:
        raise ValueError(f"父資料夾下沒有子資料夾：{parent_dir}")

    jobs: list[tuple[Path, Path, Path, str, float, float]] = []
    skipped = 0

    for sku_dir in sku_dirs:
        folder_name = sku_dir.name.strip()
        sku = extract_sku_from_folder_name(folder_name)
        if sku is None:
            log(f"略過資料夾 {folder_name}: 無法從名稱辨識 SKU")
            skipped += 1
            continue

        size = sku_sizes.get(sku)
        if size is None:
            if sku in sku_catalog.all_skus:
                reason = "Excel 有此 SKU 但無尺碼"
            else:
                reason = "Excel 中無此 SKU"
            log(f"略過資料夾 {folder_name}（SKU={sku}）: {reason}")
            skipped += 1
            continue

        crop_w_cm, crop_h_cm = size
        files = collect_folder_images(sku_dir, recursive=True, exclude_roots=[out])
        if not files:
            log(f"略過資料夾 {folder_name}（SKU={sku}）: 無圖片")
            skipped += 1
            continue

        log(
            f"匹配 {folder_name} → SKU {sku} → 裁切 {crop_w_cm:g}x{crop_h_cm:g} cm"
            f"（{len(files)} 張）"
        )
        sku_out = out / folder_name
        for src in files:
            dest = mirror_dest(
                src, sku_dir, sku_out, name=expand_output_name(src, expand_ext)
            )
            jobs.append((src, dest, sku_dir, sku, crop_w_cm, crop_h_cm))

    log(f"輸入父目錄：{parent_dir}")
    log(f"Excel：{excel_path}")
    log(f"輸出：{out}")
    log(
        f"SKU 對照表 {len(sku_sizes)} 筆 | 子資料夾 {len(sku_dirs)} 個 | "
        f"待處理 {len(jobs)} 張 | 略過資料夾 {skipped} 個"
    )
    log("只輸出最終結果；每個 SKU 資料夾名稱與張數對應輸出\n")

    if not jobs:
        raise ValueError("沒有可處理的圖片（請確認 SKU 資料夾名稱與 Excel 尺碼）。")

    results: list[PipelineItemResult] = []
    total = len(jobs)

    for i, (src, dest, sku_dir, sku, crop_w_cm, crop_h_cm) in enumerate(jobs, start=1):
        job_expand = _settings_with_crop(expand, crop_w_cm, crop_h_cm)
        try:
            rel_label = str(src.relative_to(sku_dir.parent))
        except ValueError:
            rel_label = f"{src.parent.name}/{src.name}"
        if dest.exists():
            log(f"  [{i}/{total}] {rel_label} 跳過（輸出已存在）")
            results.append(
                PipelineItemResult(
                    source=str(src),
                    unit_path=None,
                    preview_path=None,
                    expand_path=str(dest),
                    mode="跳過（已存在）",
                    sku=sku,
                    crop_w_cm=crop_w_cm,
                    crop_h_cm=crop_h_cm,
                )
            )
            if on_progress is not None:
                on_progress(i / total)
            continue
        try:
            item = _process_one_image(
                src,
                dest,
                margin=margin,
                margin_is_percent=margin_is_percent,
                threshold=threshold,
                auto_bg=auto_bg,
                manual_bg=manual_bg,
                expand=job_expand,
                do_expand=do_expand,
                log=log,
            )
            item.sku = sku
            item.crop_w_cm = crop_w_cm
            item.crop_h_cm = crop_h_cm
            log(f"  [{i}/{total}] {rel_label} OK")
        except Exception as exc:  # noqa: BLE001
            item = PipelineItemResult(
                source=str(src),
                unit_path=None,
                preview_path=None,
                expand_path=None,
                mode="",
                error=str(exc),
                sku=sku,
                crop_w_cm=crop_w_cm,
                crop_h_cm=crop_h_cm,
            )
            log(f"  [{i}/{total}] {rel_label} 失敗：{exc}")

        results.append(item)
        if on_progress is not None:
            on_progress(i / total)

    ok_n = sum(1 for r in results if not r.error)
    log(f"\n完成：成功 {ok_n}/{total}（略過資料夾 {skipped}）→ {out}")
    return results
