"""四方連續圖 customtkinter 主視窗。"""

from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import colorchooser, filedialog, messagebox
from typing import Callable

import customtkinter as ctk
from PIL import Image, ImageTk

from app.color_utils import detect_background, hex_to_rgb, rgb_to_hex
from app.pipeline import (
    ExpandSettings,
    expand_unit,
    normalize_expand_ext,
    run_full_pipeline,
)
from app.processor import make_seamless_hard_cut, tile_2x2_multi

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
EXPAND_FILETYPES = [
    ("TIFF", "*.tif;*.tiff"),
    ("PNG", "*.png"),
    ("JPEG", "*.jpg;*.jpeg"),
]


class SeamlessTileApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("四方連續圖工具 · 擴圖匯出")
        self.geometry("1100x760")
        self.minsize(900, 640)

        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")

        self.source_image: Image.Image | None = None
        self.unit_image: Image.Image | None = None
        self.preview_2x2: Image.Image | None = None
        self._seam_cross: tuple[int, int] | None = None
        self.source_path: Path | None = None
        self.bg_rgb: tuple[int, int, int] = (255, 255, 255)
        self.pick_mode = False
        self._photo: ImageTk.PhotoImage | None = None
        self._busy = False

        self._build_ui()

    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        main = ctk.CTkFrame(self, fg_color="transparent")
        main.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)
        main.grid_columnconfigure(0, weight=3)
        main.grid_columnconfigure(1, weight=1)
        main.grid_rowconfigure(0, weight=1)

        # ---- 左側預覽 ----
        left = ctk.CTkFrame(main)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        left.grid_columnconfigure(0, weight=1)
        left.grid_rowconfigure(1, weight=1)

        self.view_mode = ctk.StringVar(value="原圖")
        view_bar = ctk.CTkSegmentedButton(
            left,
            values=["原圖", "單元圖", "2×2 預覽"],
            variable=self.view_mode,
            command=lambda _: self._refresh_preview(),
        )
        view_bar.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 6))

        self.canvas = tk.Canvas(left, bg="#e8e8e8", highlightthickness=0, cursor="crosshair")
        self.canvas.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        self.canvas.bind("<Configure>", lambda _e: self._refresh_preview())
        self.canvas.bind("<Button-1>", self._on_canvas_click)

        self.status_label = ctk.CTkLabel(left, text="請開啟圖片", anchor="w")
        self.status_label.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 10))

        # ---- 右側控制 ----
        right = ctk.CTkScrollableFrame(main, width=280)
        right.grid(row=0, column=1, sticky="nsew")

        ctk.CTkLabel(right, text="檔案", font=ctk.CTkFont(size=14, weight="bold")).pack(
            anchor="w", padx=8, pady=(8, 4)
        )
        ctk.CTkButton(right, text="開啟圖片…", command=self.open_image).pack(
            fill="x", padx=8, pady=3
        )
        ctk.CTkButton(right, text="批次處理資料夾…", command=self.batch_folder).pack(
            fill="x", padx=8, pady=3
        )
        ctk.CTkButton(
            right,
            text="完整流水線（只輸出最終結果）…",
            fg_color="#0d6e6a",
            hover_color="#085550",
            command=self.batch_full_pipeline,
        ).pack(fill="x", padx=8, pady=3)

        ctk.CTkLabel(right, text="背景色", font=ctk.CTkFont(size=14, weight="bold")).pack(
            anchor="w", padx=8, pady=(16, 4)
        )
        color_row = ctk.CTkFrame(right, fg_color="transparent")
        color_row.pack(fill="x", padx=8, pady=3)
        self.color_swatch = ctk.CTkButton(
            color_row,
            text="",
            width=40,
            height=28,
            fg_color=rgb_to_hex(self.bg_rgb),
            hover=False,
            command=self.choose_color,
        )
        self.color_swatch.pack(side="left")
        self.hex_entry = ctk.CTkEntry(color_row, width=90)
        self.hex_entry.insert(0, rgb_to_hex(self.bg_rgb))
        self.hex_entry.pack(side="left", padx=6)
        ctk.CTkButton(color_row, text="套用", width=50, command=self.apply_hex).pack(side="left")

        self.auto_bg_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            right,
            text="自動識別背景色（建議開啟，批次必用）",
            variable=self.auto_bg_var,
        ).pack(anchor="w", padx=8, pady=(8, 3))

        ctk.CTkButton(right, text="重新自動識別", command=self.auto_sample_bg).pack(
            fill="x", padx=8, pady=3
        )
        self.pick_btn = ctk.CTkButton(
            right, text="點選圖片吸色", command=self.toggle_pick_mode
        )
        self.pick_btn.pack(fill="x", padx=8, pady=3)

        ctk.CTkLabel(right, text="邊緣帶寬度", font=ctk.CTkFont(size=14, weight="bold")).pack(
            anchor="w", padx=8, pady=(16, 4)
        )
        self.margin_mode = ctk.StringVar(value="百分比")
        ctk.CTkSegmentedButton(
            right,
            values=["百分比", "像素"],
            variable=self.margin_mode,
            command=self._on_margin_mode_change,
        ).pack(fill="x", padx=8, pady=3)

        self.margin_slider = ctk.CTkSlider(
            right, from_=0.0, to=15.0, number_of_steps=150, command=self._on_margin_slide
        )
        self.margin_slider.set(0.0)
        self.margin_slider.pack(fill="x", padx=8, pady=3)
        self.margin_label = ctk.CTkLabel(right, text="0 %（不改圖）")
        self.margin_label.pack(anchor="w", padx=8)

        ctk.CTkLabel(right, text="色差閾值", font=ctk.CTkFont(size=14, weight="bold")).pack(
            anchor="w", padx=8, pady=(16, 4)
        )
        self.threshold_slider = ctk.CTkSlider(
            right, from_=5, to=120, number_of_steps=115, command=self._on_threshold_slide
        )
        self.threshold_slider.set(40)
        self.threshold_slider.pack(fill="x", padx=8, pady=3)
        self.threshold_label = ctk.CTkLabel(right, text="40")
        self.threshold_label.pack(anchor="w", padx=8)

        ctk.CTkLabel(
            right,
            text="滿鋪／密鋪：2×2 錯位後空白處緊挨補同一張圖；疏點綴請加大邊緣帶清邊補花。",
            wraplength=240,
            text_color="#666666",
            justify="left",
        ).pack(anchor="w", padx=8, pady=(12, 4))

        ctk.CTkButton(
            right, text="處理", height=36, command=self.process_current
        ).pack(fill="x", padx=8, pady=(12, 3))
        ctk.CTkButton(right, text="匯出單元圖…", command=self.export_unit).pack(
            fill="x", padx=8, pady=3
        )
        ctk.CTkButton(right, text="匯出 2×2…", command=self.export_2x2).pack(
            fill="x", padx=8, pady=3
        )
        ctk.CTkButton(
            right, text="匯出擴圖…", command=self.export_expand
        ).pack(fill="x", padx=8, pady=3)

        ctk.CTkLabel(
            right, text="擴圖裁切加邊", font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor="w", padx=8, pady=(16, 4))

        self.do_expand_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            right, text="流水線執行擴圖裁切加邊", variable=self.do_expand_var
        ).pack(anchor="w", padx=8, pady=3)

        self.sku_batch_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            right,
            text="SKU 批次（子資料夾＝SKU，尺碼讀 Excel）",
            variable=self.sku_batch_var,
            command=self._on_sku_mode_change,
        ).pack(anchor="w", padx=8, pady=3)

        excel_row = ctk.CTkFrame(right, fg_color="transparent")
        excel_row.pack(fill="x", padx=8, pady=2)
        self.excel_var = ctk.StringVar(value="")
        self.excel_entry = ctk.CTkEntry(
            excel_row, textvariable=self.excel_var, placeholder_text="SKU 尺碼 Excel…"
        )
        self.excel_entry.pack(side="left", fill="x", expand=True)
        self.excel_btn = ctk.CTkButton(
            excel_row, text="選擇…", width=56, command=self._pick_excel
        )
        self.excel_btn.pack(side="left", padx=(4, 0))

        ctk.CTkLabel(right, text="擴圖匯出格式", anchor="w").pack(anchor="w", padx=8, pady=(6, 2))
        self.expand_format_var = ctk.StringVar(value="TIFF")
        ctk.CTkSegmentedButton(
            right,
            values=["TIFF", "PNG", "JPEG"],
            variable=self.expand_format_var,
        ).pack(fill="x", padx=8, pady=3)

        self.crop_w_var = ctk.StringVar(value="30")
        self.crop_h_var = ctk.StringVar(value="30")
        self.target_w_var = ctk.StringVar(value="100")
        self.target_h_var = ctk.StringVar(value="100")
        self.white_cm_var = ctk.StringVar(value="0.2")
        self.black_cm_var = ctk.StringVar(value="0.1")
        self.dpi_var = ctk.StringVar(value="300")

        for label, var_a, var_b, a_name, b_name in (
            ("擴展 cm", self.target_w_var, self.target_h_var, "寬", "高"),
            ("裁切 cm", self.crop_w_var, self.crop_h_var, "寬", "高"),
            ("邊框 cm", self.white_cm_var, self.black_cm_var, "白", "黑"),
        ):
            row = ctk.CTkFrame(right, fg_color="transparent")
            row.pack(fill="x", padx=8, pady=2)
            ctk.CTkLabel(row, text=label, width=56, anchor="w").pack(side="left")
            ctk.CTkLabel(row, text=a_name).pack(side="left")
            entry_a = ctk.CTkEntry(row, textvariable=var_a, width=48)
            entry_a.pack(side="left", padx=2)
            ctk.CTkLabel(row, text=b_name).pack(side="left", padx=(6, 0))
            entry_b = ctk.CTkEntry(row, textvariable=var_b, width=48)
            entry_b.pack(side="left", padx=2)
            if label == "裁切 cm":
                self.crop_w_entry = entry_a
                self.crop_h_entry = entry_b

        self.crop_hint = ctk.CTkLabel(
            right, text="", text_color="#666666", anchor="w"
        )
        self.crop_hint.pack(anchor="w", padx=8)

        dpi_row = ctk.CTkFrame(right, fg_color="transparent")
        dpi_row.pack(fill="x", padx=8, pady=2)
        ctk.CTkLabel(dpi_row, text="DPI", width=56, anchor="w").pack(side="left")
        ctk.CTkEntry(dpi_row, textvariable=self.dpi_var, width=64).pack(side="left")

        self.progress = ctk.CTkProgressBar(right)
        self.progress.pack(fill="x", padx=8, pady=(16, 8))
        self.progress.set(0)

        self._on_sku_mode_change()

    # ---- 參數 ----
    def _on_margin_mode_change(self, _value: str | None = None) -> None:
        if self.margin_mode.get() == "百分比":
            self.margin_slider.configure(from_=0.0, to=15.0, number_of_steps=150)
            self.margin_slider.set(0.0)
            self.margin_label.configure(text="0 %（不改圖）")
        else:
            self.margin_slider.configure(from_=0, to=80, number_of_steps=80)
            self.margin_slider.set(0)
            self.margin_label.configure(text="0 px（不改圖）")

    def _on_margin_slide(self, value: float) -> None:
        v = float(value)
        if self.margin_mode.get() == "百分比":
            if v <= 0.05:
                self.margin_label.configure(text="0 %（不改圖）")
            else:
                self.margin_label.configure(text=f"{v:.1f} %")
        else:
            px = int(round(v))
            if px <= 0:
                self.margin_label.configure(text="0 px（不改圖）")
            else:
                self.margin_label.configure(text=f"{px} px")

    def _on_threshold_slide(self, value: float) -> None:
        self.threshold_label.configure(text=str(int(round(float(value)))))

    def _params(self) -> tuple[float, bool, float]:
        margin_raw = float(self.margin_slider.get())
        is_percent = self.margin_mode.get() == "百分比"
        margin = (margin_raw / 100.0) if is_percent else margin_raw
        threshold = float(self.threshold_slider.get())
        return margin, is_percent, threshold

    def _on_sku_mode_change(self) -> None:
        sku = self.sku_batch_var.get()
        crop_state = "disabled" if sku else "normal"
        excel_state = "normal" if sku else "disabled"
        if hasattr(self, "crop_w_entry"):
            self.crop_w_entry.configure(state=crop_state)
            self.crop_h_entry.configure(state=crop_state)
        if hasattr(self, "excel_entry"):
            self.excel_entry.configure(state=excel_state)
            self.excel_btn.configure(state=excel_state)
        if hasattr(self, "crop_hint"):
            self.crop_hint.configure(
                text="裁切尺寸由 Excel 尺碼決定" if sku else ""
            )

    def _pick_excel(self) -> None:
        path = filedialog.askopenfilename(
            title="選擇 SKU 尺碼 Excel",
            filetypes=[
                ("Excel", "*.xlsx;*.xlsm;*.xltx;*.xltm"),
                ("所有檔案", "*.*"),
            ],
        )
        if path:
            self.excel_var.set(path)

    def _expand_settings(self) -> ExpandSettings:
        return ExpandSettings(
            dpi=float(self.dpi_var.get()),
            target_w_cm=float(self.target_w_var.get()),
            target_h_cm=float(self.target_h_var.get()),
            crop_w_cm=float(self.crop_w_var.get() or "30"),
            crop_h_cm=float(self.crop_h_var.get() or "30"),
            white_cm=float(self.white_cm_var.get()),
            black_cm=float(self.black_cm_var.get()),
        )

    def _set_bg(self, rgb: tuple[int, int, int]) -> None:
        self.bg_rgb = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
        hex_color = rgb_to_hex(self.bg_rgb)
        self.color_swatch.configure(fg_color=hex_color)
        self.hex_entry.delete(0, "end")
        self.hex_entry.insert(0, hex_color)

    # ---- 檔案 ----
    def open_image(self) -> None:
        path = filedialog.askopenfilename(
            title="選擇圖片",
            filetypes=[
                ("圖片", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff;*.webp"),
                ("所有檔案", "*.*"),
            ],
        )
        if not path:
            return
        self._load_image(Path(path))

    def _load_image(self, path: Path) -> None:
        try:
            img = Image.open(path)
            img.load()
        except OSError as exc:
            messagebox.showerror("讀取失敗", str(exc))
            return
        self.source_path = path
        self.source_image = img
        self.unit_image = None
        self.preview_2x2 = None
        self._seam_cross = None
        self._set_bg(detect_background(img))
        self.view_mode.set("原圖")
        self.status_label.configure(
            text=f"已載入：{path.name}  ({img.width}×{img.height})  背景 {rgb_to_hex(self.bg_rgb)}"
        )
        self._refresh_preview()

    # ---- 背景色 ----
    def choose_color(self) -> None:
        result = colorchooser.askcolor(color=rgb_to_hex(self.bg_rgb), title="選擇背景色")
        if result and result[0]:
            r, g, b = result[0]
            self._set_bg((int(r), int(g), int(b)))

    def apply_hex(self) -> None:
        try:
            self._set_bg(hex_to_rgb(self.hex_entry.get()))
        except ValueError as exc:
            messagebox.showwarning("色碼錯誤", str(exc))

    def auto_sample_bg(self) -> None:
        if self.source_image is None:
            messagebox.showinfo("提示", "請先開啟圖片")
            return
        self._set_bg(detect_background(self.source_image))
        self.status_label.configure(text=f"已自動識別背景色 {rgb_to_hex(self.bg_rgb)}")

    def toggle_pick_mode(self) -> None:
        if self.source_image is None:
            messagebox.showinfo("提示", "請先開啟圖片")
            return
        self.pick_mode = not self.pick_mode
        if self.pick_mode:
            self.pick_btn.configure(text="吸色中（再點取消）", fg_color="#c45c26")
            self.view_mode.set("原圖")
            self._refresh_preview()
            self.status_label.configure(text="點選預覽圖上的像素以吸取背景色")
        else:
            self.pick_btn.configure(text="點選圖片吸色", fg_color=["#3B8ED0", "#1F6AA5"])
            self.status_label.configure(
                text=f"已載入：{self.source_path.name}" if self.source_path else "請開啟圖片"
            )

    def _on_canvas_click(self, event: tk.Event) -> None:
        if not self.pick_mode or self.source_image is None:
            return
        mapping = getattr(self, "_preview_mapping", None)
        if not mapping:
            return
        scale, ox, oy, disp_w, disp_h = mapping
        x = event.x - ox
        y = event.y - oy
        if x < 0 or y < 0 or x >= disp_w or y >= disp_h:
            return
        src = self.source_image.convert("RGB")
        px = min(src.width - 1, max(0, int(x / scale)))
        py = min(src.height - 1, max(0, int(y / scale)))
        self._set_bg(src.getpixel((px, py)))
        self.pick_mode = False
        self.pick_btn.configure(text="點選圖片吸色", fg_color=["#3B8ED0", "#1F6AA5"])
        self.status_label.configure(text=f"已吸色 {rgb_to_hex(self.bg_rgb)} @ ({px}, {py})")

    # ---- 處理 ----
    def process_current(self) -> None:
        if self.source_image is None:
            messagebox.showinfo("提示", "請先開啟圖片")
            return
        if self._busy:
            return
        self._run_async(self._process_worker, on_done=self._on_process_done)

    def _process_worker(self) -> tuple[Image.Image, Image.Image, tuple[int, int, int], str]:
        margin, is_percent, threshold = self._params()
        assert self.source_image is not None
        used_bg = (
            detect_background(self.source_image)
            if self.auto_bg_var.get()
            else self.bg_rgb
        )
        unit, mode = make_seamless_hard_cut(
            self.source_image,
            bg=used_bg,
            margin=margin,
            threshold=threshold,
            margin_is_percent=is_percent,
        )
        preview, preview_detail, seam = tile_2x2_multi(unit)
        return unit, preview, used_bg, f"{mode}；{preview_detail}", seam

    def _on_process_done(self, result: object, error: BaseException | None) -> None:
        if error:
            messagebox.showerror("處理失敗", str(error))
            return
        unit, preview, used_bg, mode, seam = result  # type: ignore[misc]
        self.unit_image = unit
        self.preview_2x2 = preview
        self._seam_cross = seam
        self._set_bg(used_bg)
        self.view_mode.set("2×2 預覽")
        self.status_label.configure(
            text=f"處理完成 — {mode}；背景 {rgb_to_hex(used_bg)}"
        )
        self._refresh_preview()

    def export_unit(self) -> None:
        if self.unit_image is None:
            messagebox.showinfo("提示", "請先按「處理」產生單元圖")
            return
        default = "seamless_unit.png"
        if self.source_path:
            default = f"{self.source_path.stem}_unit.png"
        path = filedialog.asksaveasfilename(
            title="匯出單元圖",
            defaultextension=".png",
            initialfile=default,
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")],
        )
        if not path:
            return
        self._save_image(self.unit_image, Path(path))

    def export_2x2(self) -> None:
        if self.preview_2x2 is None:
            messagebox.showinfo("提示", "請先按「處理」產生 2×2 預覽")
            return
        default = "seamless_2x2.png"
        if self.source_path:
            default = f"{self.source_path.stem}_2x2.png"
        path = filedialog.asksaveasfilename(
            title="匯出 2×2",
            defaultextension=".png",
            initialfile=default,
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")],
        )
        if not path:
            return
        self._save_image(self.preview_2x2, Path(path))

    def _save_image(self, image: Image.Image, path: Path) -> None:
        try:
            if path.suffix.lower() in {".jpg", ".jpeg"}:
                image.convert("RGB").save(path, quality=95)
            else:
                image.save(path)
            self.status_label.configure(text=f"已匯出：{path.name}")
        except OSError as exc:
            messagebox.showerror("匯出失敗", str(exc))

    def batch_folder(self) -> None:
        folder = filedialog.askdirectory(title="選擇要批次處理的資料夾")
        if not folder:
            return
        src_dir = Path(folder)
        files = sorted(
            p for p in src_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not files:
            messagebox.showinfo("提示", "資料夾內沒有支援的圖片")
            return
        out_dir = src_dir / "output"
        out_dir.mkdir(exist_ok=True)
        margin, is_percent, threshold = self._params()
        auto_bg = self.auto_bg_var.get()
        manual_bg = self.bg_rgb

        def worker() -> int:
            ok = 0
            total = len(files)
            for i, path in enumerate(files):
                try:
                    img = Image.open(path)
                    img.load()
                    use_bg = None if auto_bg else manual_bg
                    unit, _mode = make_seamless_hard_cut(
                        img,
                        bg=use_bg,
                        margin=margin,
                        threshold=threshold,
                        margin_is_percent=is_percent,
                    )
                    preview, _, _ = tile_2x2_multi(unit)
                    unit.save(out_dir / f"{path.stem}_unit.png")
                    preview.save(out_dir / f"{path.stem}_2x2.png")
                    ok += 1
                except OSError:
                    pass
                self.after(0, lambda v=(i + 1) / total: self.progress.set(v))
            return ok

        def done(result: object, error: BaseException | None) -> None:
            if error:
                messagebox.showerror("批次失敗", str(error))
                return
            messagebox.showinfo(
                "批次完成",
                f"成功 {result}/{len(files)} 張\n輸出目錄：{out_dir}",
            )
            self.progress.set(0)

        self._run_async(worker, on_done=done)

    def export_expand(self) -> None:
        if self.unit_image is None:
            messagebox.showinfo("提示", "請先按「處理」產生單元圖")
            return
        try:
            expand = self._expand_settings()
        except ValueError:
            messagebox.showwarning("參數錯誤", "請檢查擴圖數值是否為有效數字")
            return
        try:
            ext = normalize_expand_ext(self.expand_format_var.get())
        except ValueError as exc:
            messagebox.showwarning("參數錯誤", str(exc))
            return
        default = f"seamless_expand{ext}"
        if self.source_path:
            default = f"{self.source_path.stem}{ext}"
        path = filedialog.asksaveasfilename(
            title="匯出擴圖",
            defaultextension=ext,
            initialfile=default,
            filetypes=EXPAND_FILETYPES,
        )
        if not path:
            return
        dest = Path(path)
        if dest.suffix.lower() not in {".tif", ".tiff", ".png", ".jpg", ".jpeg"}:
            dest = dest.with_suffix(ext)
        tmp = dest.with_name(f"{dest.stem}__unit_tmp.png")

        def worker() -> Path:
            assert self.unit_image is not None
            self.unit_image.save(tmp)
            try:
                expand_unit(tmp, dest, expand)
            finally:
                if tmp.exists() and tmp != dest:
                    tmp.unlink(missing_ok=True)
            return dest

        def done(result: object, error: BaseException | None) -> None:
            if error:
                messagebox.showerror("匯出失敗", str(error))
                return
            self.status_label.configure(text=f"已匯出擴圖：{Path(str(result)).name}")

        self._run_async(worker, on_done=done)

    def batch_full_pipeline(self) -> None:
        sku_mode = self.sku_batch_var.get()
        title = (
            "選擇父資料夾（內含 SKU 子資料夾）"
            if sku_mode
            else "選擇要跑完整流水線的資料夾"
        )
        folder = filedialog.askdirectory(title=title)
        if not folder:
            return
        src_dir = Path(folder)

        excel_path: Path | None = None
        if sku_mode:
            excel_raw = self.excel_var.get().strip().strip('"')
            if not excel_raw:
                messagebox.showwarning("提示", "請先選擇 SKU 尺碼 Excel")
                return
            excel_path = Path(excel_raw)
            if not excel_path.is_file():
                messagebox.showerror("錯誤", f"Excel 不存在：\n{excel_path}")
                return
            subdirs = [p for p in src_dir.iterdir() if p.is_dir()]
            if not subdirs:
                messagebox.showinfo("提示", "父資料夾下沒有子資料夾")
                return
        else:
            files = sorted(
                p
                for p in src_dir.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            )
            if not files:
                messagebox.showinfo("提示", "資料夾內沒有支援的圖片")
                return

        try:
            expand = self._expand_settings()
            expand_format = self.expand_format_var.get()
            normalize_expand_ext(expand_format)
        except ValueError as exc:
            messagebox.showwarning("參數錯誤", str(exc) or "請檢查擴圖數值是否為有效數字")
            return

        out_dir = src_dir / "pipeline_out"
        margin, is_percent, threshold = self._params()
        auto_bg = self.auto_bg_var.get()
        manual_bg = self.bg_rgb
        do_expand = self.do_expand_var.get()
        logs: list[str] = []

        def worker() -> tuple[int, int, Path]:
            results = run_full_pipeline(
                src_dir,
                output_dir=out_dir,
                margin=margin,
                margin_is_percent=is_percent,
                threshold=threshold,
                auto_bg=auto_bg,
                manual_bg=manual_bg,
                expand=expand,
                do_expand=do_expand,
                expand_format=expand_format,
                excel_path=excel_path,
                log=logs.append,
                on_progress=lambda v: self.after(0, lambda: self.progress.set(v)),
            )
            ok = sum(1 for r in results if not r.error)
            return ok, len(results), out_dir

        def done(result: object, error: BaseException | None) -> None:
            if error:
                messagebox.showerror("流水線失敗", str(error))
                return
            ok, total, out = result  # type: ignore[misc]
            detail = "\n".join(logs[-10:]) if logs else ""
            messagebox.showinfo(
                "流水線完成",
                f"成功 {ok}/{total}\n輸出：{out}\n\n{detail}",
            )
            self.progress.set(0)

        self._run_async(worker, on_done=done)

    # ---- 預覽 ----
    def _current_display_image(self) -> Image.Image | None:
        mode = self.view_mode.get()
        if mode == "原圖":
            return self.source_image
        if mode == "單元圖":
            return self.unit_image or self.source_image
        return self.preview_2x2 or self.unit_image or self.source_image

    def _refresh_preview(self) -> None:
        image = self._current_display_image()
        self.canvas.delete("all")
        self._preview_mapping = None
        if image is None:
            return
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        if cw < 10 or ch < 10:
            return

        rgb = image.convert("RGB")
        scale = min(cw / rgb.width, ch / rgb.height, 1.0)
        disp_w = max(1, int(rgb.width * scale))
        disp_h = max(1, int(rgb.height * scale))
        resized = rgb.resize((disp_w, disp_h), Image.Resampling.LANCZOS)
        self._photo = ImageTk.PhotoImage(resized)
        ox = (cw - disp_w) // 2
        oy = (ch - disp_h) // 2
        self.canvas.create_image(ox, oy, anchor="nw", image=self._photo)
        self._preview_mapping = (scale, ox, oy, disp_w, disp_h)

        # 2×2 時畫細線標示接縫（錯位拼接用真實縫線座標）
        if self.view_mode.get() == "2×2 預覽" and self.preview_2x2 is not None:
            if self._seam_cross is not None:
                sx, sy = self._seam_cross
                mid_x = ox + int(sx * scale)
                mid_y = oy + int(sy * scale)
            else:
                mid_x = ox + disp_w // 2
                mid_y = oy + disp_h // 2
            self.canvas.create_line(mid_x, oy, mid_x, oy + disp_h, fill="#ff4444", dash=(4, 4))
            self.canvas.create_line(ox, mid_y, ox + disp_w, mid_y, fill="#ff4444", dash=(4, 4))

    # ---- 非同步 ----
    def _run_async(
        self,
        fn: Callable[[], object],
        on_done: Callable[[object, BaseException | None], None],
    ) -> None:
        if self._busy:
            return
        self._busy = True
        self.progress.set(0.15)

        def target() -> None:
            result: object = None
            error: BaseException | None = None
            try:
                result = fn()
            except BaseException as exc:  # noqa: BLE001 — 回傳 GUI 顯示
                error = exc

            def finish() -> None:
                self._busy = False
                self.progress.set(1.0 if error is None else 0)
                on_done(result, error)
                if error is None:
                    self.after(400, lambda: self.progress.set(0))

            self.after(0, finish)

        threading.Thread(target=target, daemon=True).start()


def run_app() -> None:
    app = SeamlessTileApp()
    app.mainloop()
