"""四方連續圖 customtkinter 主視窗。"""

from __future__ import annotations

import tempfile
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import colorchooser, filedialog, messagebox
from typing import Callable

import customtkinter as ctk
from PIL import Image, ImageTk

from app.color_utils import detect_background, hex_to_rgb, rgb_to_hex, to_srgb
from app.pipeline import (
    SKIP_DIR_NAMES,
    ExpandSettings,
    collect_folder_images,
    expand_output_name,
    expand_unit,
    mirror_dest,
    normalize_expand_ext,
    run_full_pipeline,
)
from app.color_io import intermediate_suffix, save_image
from app.processor import make_seamless_hard_cut, tile_2x2_multi
from app.triage import VERDICT_TILEABLE, triage

EXPAND_FILETYPES = [
    ("TIFF", "*.tif;*.tiff"),
    ("PNG", "*.png"),
    ("JPEG", "*.jpg;*.jpeg"),
]


class SeamlessTileApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("四方連續圖工具 · 擴圖匯出")
        self.geometry("1100x800")
        self.minsize(900, 680)

        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")

        self.source_image: Image.Image | None = None
        self.unit_image: Image.Image | None = None
        self.preview_2x2: Image.Image | None = None
        self._seam_cross: tuple[int, int] | None = None
        self.source_path: Path | None = None
        self.folder_files: list[Path] = []
        self.folder_root: Path | None = None
        self.folder_index: int = -1
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

        self.status_label = ctk.CTkLabel(
            left, text="選輸入與輸出資料夾 → 調參／預覽 → 開始批次", anchor="w"
        )
        self.status_label.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 10))

        # ---- 右側控制 ----
        right = ctk.CTkScrollableFrame(main, width=300)
        right.grid(row=0, column=1, sticky="nsew")

        # —— 路徑 ——
        ctk.CTkLabel(right, text="路徑", font=ctk.CTkFont(size=14, weight="bold")).pack(
            anchor="w", padx=8, pady=(8, 4)
        )
        ctk.CTkLabel(
            right,
            text="選輸入與輸出資料夾 → 調參／預覽 → 開始批次。載入只預覽，不會自動出圖。",
            wraplength=260,
            text_color="#666666",
            justify="left",
        ).pack(anchor="w", padx=8, pady=(0, 4))

        self.input_dir_var = ctk.StringVar(value="")
        self.output_dir_var = ctk.StringVar(value="")
        self.excel_var = ctk.StringVar(value="")

        ctk.CTkLabel(right, text="輸入資料夾", anchor="w").pack(anchor="w", padx=8, pady=(4, 2))
        in_row = ctk.CTkFrame(right, fg_color="transparent")
        in_row.pack(fill="x", padx=8, pady=2)
        ctk.CTkEntry(
            in_row,
            textvariable=self.input_dir_var,
            placeholder_text="選擇要處理的資料夾…",
        ).pack(side="left", fill="x", expand=True)
        ctk.CTkButton(in_row, text="選擇…", width=56, command=self._pick_input_dir).pack(
            side="left", padx=(4, 0)
        )

        ctk.CTkLabel(right, text="輸出資料夾", anchor="w").pack(anchor="w", padx=8, pady=(6, 2))
        out_row = ctk.CTkFrame(right, fg_color="transparent")
        out_row.pack(fill="x", padx=8, pady=2)
        ctk.CTkEntry(
            out_row,
            textvariable=self.output_dir_var,
            placeholder_text="選擇輸出根目錄…",
        ).pack(side="left", fill="x", expand=True)
        ctk.CTkButton(
            out_row, text="選擇…", width=56, command=self._pick_output_dir
        ).pack(side="left", padx=(4, 0))
        ctk.CTkLabel(
            right,
            text="會保留 SKU／子資料夾名稱與相對結構。",
            wraplength=260,
            text_color="#666666",
            justify="left",
        ).pack(anchor="w", padx=8, pady=(0, 4))

        self.sku_batch_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            right,
            text="SKU 批次（子資料夾名「-」前為 SKU，尺碼讀 Excel）",
            variable=self.sku_batch_var,
            command=self._on_sku_mode_change,
        ).pack(anchor="w", padx=8, pady=3)

        excel_row = ctk.CTkFrame(right, fg_color="transparent")
        excel_row.pack(fill="x", padx=8, pady=2)
        self.excel_entry = ctk.CTkEntry(
            excel_row, textvariable=self.excel_var, placeholder_text="SKU 尺碼 Excel…"
        )
        self.excel_entry.pack(side="left", fill="x", expand=True)
        self.excel_btn = ctk.CTkButton(
            excel_row, text="選擇…", width=56, command=self._pick_excel
        )
        self.excel_btn.pack(side="left", padx=(4, 0))

        ctk.CTkButton(
            right, text="開啟單張圖片…", command=self.open_image, height=28
        ).pack(fill="x", padx=8, pady=(8, 3))

        nav = ctk.CTkFrame(right, fg_color="transparent")
        nav.pack(fill="x", padx=8, pady=(4, 2))
        self.prev_btn = ctk.CTkButton(
            nav, text="◀ 上一張", width=90, command=self.prev_image, state="disabled"
        )
        self.prev_btn.pack(side="left")
        self.next_btn = ctk.CTkButton(
            nav, text="下一張 ▶", width=90, command=self.next_image, state="disabled"
        )
        self.next_btn.pack(side="right")
        self.folder_pos_label = ctk.CTkLabel(
            right, text="尚未載入資料夾", text_color="#666666"
        )
        self.folder_pos_label.pack(anchor="w", padx=8, pady=(0, 4))

        # —— 背景色 / 預覽參數 ——
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
            text="自動識別背景色（建議開啟）",
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
            text="滿鋪：2×2 錯位後空白緊挨補圖；疏點綴請加大邊緣帶清邊補花。",
            wraplength=260,
            text_color="#666666",
            justify="left",
        ).pack(anchor="w", padx=8, pady=(12, 4))

        # —— 單張預覽 ——
        ctk.CTkLabel(
            right, text="單張預覽", font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor="w", padx=8, pady=(16, 4))
        ctk.CTkLabel(
            right,
            text="用上方參數產生單元圖／2×2 預覽（僅畫面預覽，不另存中間檔）。",
            wraplength=260,
            text_color="#666666",
            justify="left",
        ).pack(anchor="w", padx=8, pady=(0, 4))
        ctk.CTkButton(
            right, text="處理目前圖片", height=36, command=self.process_current
        ).pack(fill="x", padx=8, pady=(4, 3))

        # —— 擴圖參數與批次 ——
        ctk.CTkLabel(
            right, text="擴圖參數", font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor="w", padx=8, pady=(16, 4))
        ctk.CTkLabel(
            right,
            text="批次只輸出最終擴圖檔；單張可另匯出目前這張。",
            wraplength=260,
            text_color="#666666",
            justify="left",
        ).pack(anchor="w", padx=8, pady=(0, 4))

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

        self.crop_hint = ctk.CTkLabel(right, text="", text_color="#666666", anchor="w")
        self.crop_hint.pack(anchor="w", padx=8)

        dpi_row = ctk.CTkFrame(right, fg_color="transparent")
        dpi_row.pack(fill="x", padx=8, pady=2)
        ctk.CTkLabel(dpi_row, text="DPI", width=56, anchor="w").pack(side="left")
        ctk.CTkEntry(dpi_row, textvariable=self.dpi_var, width=64).pack(side="left")

        ctk.CTkButton(
            right, text="匯出目前圖片的擴圖…", command=self.export_expand
        ).pack(fill="x", padx=8, pady=(10, 3))
        ctk.CTkButton(
            right,
            text="開始批次處理",
            height=40,
            fg_color="#0d6e6a",
            hover_color="#085550",
            command=self.start_batch,
        ).pack(fill="x", padx=8, pady=(4, 3))
        ctk.CTkLabel(
            right,
            text="使用上方輸入／輸出資料夾；記憶體跑四方連續，只寫最終檔。",
            wraplength=260,
            text_color="#666666",
            justify="left",
        ).pack(anchor="w", padx=8, pady=(0, 4))

        self.progress = ctk.CTkProgressBar(right)
        self.progress.pack(fill="x", padx=8, pady=(16, 8))
        self.progress.set(0)

        self._on_sku_mode_change()
        self._update_folder_nav()

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
                text="裁切尺寸由 Excel 尺碼決定（僅批次）" if sku else ""
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

    def _pick_input_dir(self) -> None:
        title = (
            "選擇父資料夾（內含 SKU 子資料夾）"
            if self.sku_batch_var.get()
            else "選擇輸入資料夾"
        )
        path = filedialog.askdirectory(title=title)
        if not path:
            return
        self.input_dir_var.set(path)
        self._load_folder_from_path(Path(path))

    def _pick_output_dir(self) -> None:
        path = filedialog.askdirectory(title="選擇輸出資料夾")
        if path:
            self.output_dir_var.set(path)

    def _require_output_dir(self) -> Path | None:
        raw = self.output_dir_var.get().strip().strip('"')
        if not raw:
            messagebox.showwarning("提示", "請先選擇輸出資料夾")
            return None
        out = Path(raw)
        try:
            out.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            messagebox.showerror("錯誤", f"無法建立輸出資料夾：\n{exc}")
            return None
        return out

    def _expand_settings(self) -> ExpandSettings:
        return ExpandSettings(
            dpi=float(self.dpi_var.get()),
            target_w_cm=float(self.target_w_var.get()),
            target_h_cm=float(self.target_h_var.get()),
            crop_w_cm=float(self.crop_w_var.get() or "30"),
            crop_h_cm=float(self.crop_h_var.get() or "30"),
            white_cm=float(self.white_cm_var.get()),
            black_cm=float(self.black_cm_var.get()),
            force_dpi=True,
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
        self.folder_files = []
        self.folder_root = None
        self.folder_index = -1
        self.input_dir_var.set("")
        self._update_folder_nav()
        self._load_image(Path(path))

    def _load_folder_from_path(self, src_dir: Path) -> None:
        files = collect_folder_images(src_dir, recursive=True)
        if not files:
            messagebox.showinfo("提示", "資料夾（含子資料夾）內沒有支援的圖片")
            return
        self.folder_root = src_dir
        self.folder_files = files
        self.folder_index = 0
        self._update_folder_nav()
        self._load_image(files[0])
        self.status_label.configure(
            text=f"已載入資料夾（含遞迴 {len(files)} 張）— 可預覽後按「處理目前圖片」或「開始批次處理」"
        )

    def prev_image(self) -> None:
        if not self.folder_files or self.folder_index <= 0:
            return
        self.folder_index -= 1
        self._update_folder_nav()
        self._load_image(self.folder_files[self.folder_index])

    def next_image(self) -> None:
        if not self.folder_files or self.folder_index >= len(self.folder_files) - 1:
            return
        self.folder_index += 1
        self._update_folder_nav()
        self._load_image(self.folder_files[self.folder_index])

    def _update_folder_nav(self) -> None:
        n = len(self.folder_files)
        if n == 0:
            self.folder_pos_label.configure(text="尚未載入資料夾")
            self.prev_btn.configure(state="disabled")
            self.next_btn.configure(state="disabled")
            return
        i = self.folder_index + 1
        path = self.folder_files[self.folder_index] if 0 <= self.folder_index < n else None
        if path is not None and self.folder_root is not None:
            try:
                name = str(path.relative_to(self.folder_root))
            except ValueError:
                name = path.name
        else:
            name = path.name if path is not None else ""
        self.folder_pos_label.configure(text=f"{i} / {n}  ·  {name}")
        self.prev_btn.configure(state="normal" if self.folder_index > 0 else "disabled")
        self.next_btn.configure(
            state="normal" if self.folder_index < n - 1 else "disabled"
        )

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
        src = to_srgb(self.source_image)
        px = min(src.width - 1, max(0, int(x / scale)))
        py = min(src.height - 1, max(0, int(y / scale)))
        self._set_bg(src.getpixel((px, py)))
        self.pick_mode = False
        self.pick_btn.configure(text="點選圖片吸色", fg_color=["#3B8ED0", "#1F6AA5"])
        self.status_label.configure(text=f"已吸色 {rgb_to_hex(self.bg_rgb)} @ ({px}, {py})")

    # ---- 步驟 1：處理 ----
    def process_current(self) -> None:
        if self.source_image is None:
            messagebox.showinfo("提示", "請先開啟圖片或載入資料夾")
            return
        if self._busy:
            return
        self._run_async(self._process_worker, on_done=self._on_process_done)

    def _process_worker(self) -> tuple[Image.Image | None, Image.Image | None, tuple[int, int, int] | None, str, tuple[int, int] | None]:
        margin, is_percent, threshold = self._params()
        assert self.source_image is not None
        decision = triage(self.source_image)
        if decision.verdict != VERDICT_TILEABLE:
            return None, None, None, decision.describe(), None
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
        if unit is None:
            self.unit_image = None
            self.preview_2x2 = None
            self._seam_cross = None
            self.view_mode.set("原圖")
            self.status_label.configure(text=mode)
            self._refresh_preview()
            return
        self.unit_image = unit
        self.preview_2x2 = preview
        self._seam_cross = seam
        self._set_bg(used_bg)
        self.view_mode.set("2×2 預覽")
        self.status_label.configure(
            text=f"預覽完成 — {mode}；背景 {rgb_to_hex(used_bg)}"
        )
        self._refresh_preview()

    # ---- 單張擴圖 ----
    def export_expand(self) -> None:
        if self.unit_image is None:
            messagebox.showinfo("提示", "請先按「處理目前圖片」產生預覽")
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
        assert self.unit_image is not None
        tmp = dest.with_name(
            f"{dest.stem}__unit_tmp{intermediate_suffix(self.unit_image)}"
        )

        def worker() -> Path:
            assert self.unit_image is not None
            save_image(self.unit_image, tmp)
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
            self.status_label.configure(text=f"擴圖完成：{Path(str(result)).name}")

        self._run_async(worker, on_done=done)

    # ---- 批次：只出最終擴圖 ----
    def start_batch(self) -> None:
        """使用上方輸入／輸出資料夾，只輸出最終擴圖。"""
        in_raw = self.input_dir_var.get().strip().strip('"')
        if not in_raw:
            messagebox.showwarning("提示", "請先選擇輸入資料夾")
            return
        src_dir = Path(in_raw)
        if not src_dir.is_dir():
            messagebox.showerror("錯誤", f"輸入資料夾不存在：\n{src_dir}")
            return

        out_dir = self._require_output_dir()
        if out_dir is None:
            return

        if self.sku_batch_var.get():
            excel_raw = self.excel_var.get().strip().strip('"')
            if not excel_raw:
                messagebox.showwarning("提示", "請先選擇 SKU 尺碼 Excel")
                return
            excel_path = Path(excel_raw)
            if not excel_path.is_file():
                messagebox.showerror("錯誤", f"Excel 不存在：\n{excel_path}")
                return
            if not any(
                p.is_dir() and p.name not in SKIP_DIR_NAMES for p in src_dir.iterdir()
            ):
                messagebox.showinfo("提示", "父資料夾下沒有子資料夾")
                return
            self._run_sku_batch(src_dir, excel_path, out_dir)
            return

        self._run_batch_on_dir(src_dir, out_dir)

    def _run_batch_on_dir(self, src_dir: Path, expand_out: Path) -> None:
        files = collect_folder_images(
            src_dir, recursive=True, exclude_roots=[expand_out]
        )
        if not files:
            messagebox.showinfo("提示", "資料夾（含子資料夾）內沒有支援的圖片")
            return

        try:
            expand = self._expand_settings()
            expand_ext = normalize_expand_ext(self.expand_format_var.get())
        except ValueError as exc:
            messagebox.showwarning(
                "參數錯誤", str(exc) or "請檢查擴圖數值是否為有效數字"
            )
            return

        margin, is_percent, threshold = self._params()
        auto_bg = self.auto_bg_var.get()
        manual_bg = self.bg_rgb

        # 單張僅記錄耗時；不因逾時降接縫品質
        hard_warn_sec = 60.0
        fail_notes: list[str] = []

        def _blog(msg: str) -> None:
            print(msg, flush=True)

        def worker() -> tuple[int, int, Path, list[str]]:
            ok = 0
            skipped = 0
            total = len(files)
            expand_out.mkdir(parents=True, exist_ok=True)
            _blog(
                f"批次開始：共 {total} 張（接縫優先；封面／有框直接跳過；輸出已存在則跳過不覆蓋）"
            )

            for i, path in enumerate(files):
                label = f"[{i + 1}/{total}]"
                dest = mirror_dest(
                    path,
                    src_dir,
                    expand_out,
                    name=expand_output_name(path, expand_ext),
                )
                if dest.exists():
                    _blog(f"{label} 跳過 {path.name}（輸出已存在：{dest.name}）")
                    ok += 1
                    self.after(0, lambda v=(i + 1) / total: self.progress.set(v))
                    continue
                _blog(f"{label} 開始 {path.name}")
                t0 = time.perf_counter()
                try:
                    img = Image.open(path)
                    img.load()
                    decision = triage(img)
                    if decision.verdict != VERDICT_TILEABLE:
                        skipped += 1
                        _blog(f"{label} {decision.describe()} {path.name}")
                        self.after(0, lambda v=(i + 1) / total: self.progress.set(v))
                        continue
                    use_bg = None if auto_bg else manual_bg
                    unit, mode = make_seamless_hard_cut(
                        img,
                        bg=use_bg,
                        margin=margin,
                        threshold=threshold,
                        margin_is_percent=is_percent,
                        log=_blog,
                    )
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with tempfile.NamedTemporaryFile(
                        suffix=intermediate_suffix(unit), delete=False
                    ) as tmp:
                        tmp_path = Path(tmp.name)
                    try:
                        save_image(unit, tmp_path)
                        expand_unit(tmp_path, dest, expand)
                    finally:
                        tmp_path.unlink(missing_ok=True)
                    ok += 1
                    elapsed = time.perf_counter() - t0
                    tag = "難圖" if elapsed >= hard_warn_sec else "完成"
                    mode_short = mode if len(mode) <= 120 else mode[:117] + "…"
                    _blog(f"{label} {tag} {path.name} {elapsed:.1f}s | {mode_short}")
                    if elapsed >= hard_warn_sec:
                        fail_notes.append(
                            f"{path.name} 耗時 {elapsed:.0f}s（演算法切換／週期搜尋偏慢）"
                        )
                except (OSError, ValueError) as exc:
                    elapsed = time.perf_counter() - t0
                    _blog(f"{label} 失敗 {path.name} {elapsed:.1f}s：{exc}")
                    fail_notes.append(f"{path.name}：{exc}")
                except BaseException as exc:  # noqa: BLE001 — 單張失敗不中斷整批
                    elapsed = time.perf_counter() - t0
                    _blog(f"{label} 異常 {path.name} {elapsed:.1f}s：{exc}")
                    fail_notes.append(f"{path.name}：{exc}")
                self.after(0, lambda v=(i + 1) / total: self.progress.set(v))

            _blog(f"批次結束：成功 {ok}/{total}，跳過 {skipped} → {expand_out}")
            return ok, skipped, expand_out, fail_notes

        def done(result: object, error: BaseException | None) -> None:
            if error:
                messagebox.showerror("批次失敗", str(error))
                return
            ok, skipped, primary, notes = result  # type: ignore[misc]
            extra = ""
            if notes:
                extra = "\n\n難圖／失敗摘要：\n" + "\n".join(notes[:12])
                if len(notes) > 12:
                    extra += f"\n…另有 {len(notes) - 12} 條（見命令列）"
            messagebox.showinfo(
                "批次完成",
                f"成功 {ok}/{len(files)}，跳過 {skipped}\n輸出：{primary}{extra}",
            )
            self.progress.set(0)

        self._run_async(worker, on_done=done)

    def _run_sku_batch(
        self,
        src_dir: Path,
        excel_path: Path,
        out_dir: Path,
    ) -> None:
        """SKU 模式：尺碼讀 Excel，只輸出最終擴圖。"""
        try:
            expand = self._expand_settings()
            expand_format = self.expand_format_var.get()
            normalize_expand_ext(expand_format)
        except ValueError as exc:
            messagebox.showwarning(
                "參數錯誤", str(exc) or "請檢查擴圖數值是否為有效數字"
            )
            return

        margin, is_percent, threshold = self._params()
        auto_bg = self.auto_bg_var.get()
        manual_bg = self.bg_rgb
        logs: list[str] = []

        def worker() -> tuple[int, int, int, Path]:
            results = run_full_pipeline(
                src_dir,
                output_dir=out_dir,
                margin=margin,
                margin_is_percent=is_percent,
                threshold=threshold,
                auto_bg=auto_bg,
                manual_bg=manual_bg,
                expand=expand,
                do_expand=True,
                expand_format=expand_format,
                excel_path=excel_path,
                log=logs.append,
                on_progress=lambda v: self.after(0, lambda: self.progress.set(v)),
            )
            ok = sum(1 for r in results if not r.error and not r.skipped)
            skipped = sum(1 for r in results if r.skipped)
            return ok, skipped, len(results), out_dir

        def done(result: object, error: BaseException | None) -> None:
            if error:
                messagebox.showerror("SKU 批次失敗", str(error))
                return
            ok, skipped, total, out = result  # type: ignore[misc]
            detail = "\n".join(logs[-8:]) if logs else ""
            messagebox.showinfo(
                "SKU 批次完成",
                f"成功 {ok}/{total}，跳過 {skipped}\n輸出：{out}\n\n{detail}",
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

        rgb = to_srgb(image)
        scale = min(cw / rgb.width, ch / rgb.height, 1.0)
        disp_w = max(1, int(rgb.width * scale))
        disp_h = max(1, int(rgb.height * scale))
        resized = rgb.resize((disp_w, disp_h), Image.Resampling.LANCZOS)
        self._photo = ImageTk.PhotoImage(resized)
        ox = (cw - disp_w) // 2
        oy = (ch - disp_h) // 2
        self.canvas.create_image(ox, oy, anchor="nw", image=self._photo)
        self._preview_mapping = (scale, ox, oy, disp_w, disp_h)

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
