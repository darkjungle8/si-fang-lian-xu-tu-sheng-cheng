"""四方連續圖 customtkinter 主視窗。"""

from __future__ import annotations

import tempfile
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

        self.status_label = ctk.CTkLabel(left, text="請開啟圖片或載入資料夾", anchor="w")
        self.status_label.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 10))

        # ---- 右側控制 ----
        right = ctk.CTkScrollableFrame(main, width=300)
        right.grid(row=0, column=1, sticky="nsew")

        # —— 檔案 ——
        ctk.CTkLabel(right, text="檔案", font=ctk.CTkFont(size=14, weight="bold")).pack(
            anchor="w", padx=8, pady=(8, 4)
        )
        ctk.CTkLabel(
            right,
            text="載入只負責預覽，不會自動出圖。",
            wraplength=260,
            text_color="#666666",
            justify="left",
        ).pack(anchor="w", padx=8, pady=(0, 4))
        ctk.CTkButton(right, text="開啟圖片…", command=self.open_image).pack(
            fill="x", padx=8, pady=3
        )
        ctk.CTkButton(right, text="載入資料夾…", command=self.load_folder).pack(
            fill="x", padx=8, pady=3
        )

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
        self.folder_pos_label = ctk.CTkLabel(right, text="尚未載入資料夾", text_color="#666666")
        self.folder_pos_label.pack(anchor="w", padx=8, pady=(0, 4))

        # —— 背景色 / 參數（步驟 1 用）——
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

        # —— 步驟 1 ——
        ctk.CTkLabel(
            right, text="步驟 1 · 四方連續", font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor="w", padx=8, pady=(16, 4))
        ctk.CTkLabel(
            right,
            text="使用上方背景色／邊緣帶／閾值。產出單元圖與 2×2，可先預覽再匯出。",
            wraplength=260,
            text_color="#666666",
            justify="left",
        ).pack(anchor="w", padx=8, pady=(0, 4))
        ctk.CTkButton(
            right, text="處理目前圖片", height=36, command=self.process_current
        ).pack(fill="x", padx=8, pady=(4, 3))
        ctk.CTkButton(right, text="匯出單元圖…", command=self.export_unit).pack(
            fill="x", padx=8, pady=3
        )
        ctk.CTkButton(right, text="匯出 2×2…", command=self.export_2x2).pack(
            fill="x", padx=8, pady=3
        )

        # —— 步驟 2 ——
        ctk.CTkLabel(
            right, text="步驟 2 · 擴圖裁切加邊", font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor="w", padx=8, pady=(16, 4))
        ctk.CTkLabel(
            right,
            text="僅此步驟使用下方擴展／裁切／邊框參數。需先完成步驟 1。",
            wraplength=260,
            text_color="#666666",
            justify="left",
        ).pack(anchor="w", padx=8, pady=(0, 4))

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

        self.crop_hint = ctk.CTkLabel(right, text="", text_color="#666666", anchor="w")
        self.crop_hint.pack(anchor="w", padx=8)

        dpi_row = ctk.CTkFrame(right, fg_color="transparent")
        dpi_row.pack(fill="x", padx=8, pady=2)
        ctk.CTkLabel(dpi_row, text="DPI", width=56, anchor="w").pack(side="left")
        ctk.CTkEntry(dpi_row, textvariable=self.dpi_var, width=64).pack(side="left")

        ctk.CTkButton(
            right, text="匯出目前圖片的擴圖…", command=self.export_expand
        ).pack(fill="x", padx=8, pady=(8, 3))
        ctk.CTkButton(
            right,
            text="對資料夾批次擴圖（只出最終檔）…",
            fg_color="#0d6e6a",
            hover_color="#085550",
            command=self.batch_expand_only,
        ).pack(fill="x", padx=8, pady=3)
        ctk.CTkLabel(
            right,
            text="批次擴圖會在記憶體跑步驟 1，只輸出 pipeline_out 最終檔，不存單元圖／2×2。",
            wraplength=260,
            text_color="#666666",
            justify="left",
        ).pack(anchor="w", padx=8, pady=(0, 4))

        # —— 批次：勾選要套用的步驟 ——
        ctk.CTkLabel(
            right, text="批次 · 自訂輸出（進階）", font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor="w", padx=8, pady=(16, 4))
        ctk.CTkLabel(
            right,
            text="需要中間檔時再勾選。可只勾步驟 2，效果同上方「批次擴圖」。",
            wraplength=260,
            text_color="#666666",
            justify="left",
        ).pack(anchor="w", padx=8, pady=(0, 4))

        self.batch_unit_var = ctk.BooleanVar(value=False)
        self.batch_2x2_var = ctk.BooleanVar(value=False)
        self.batch_expand_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            right, text="步驟 1：匯出單元圖（_unit.png）", variable=self.batch_unit_var
        ).pack(anchor="w", padx=8, pady=2)
        ctk.CTkCheckBox(
            right, text="步驟 1：匯出 2×2（_2x2.png）", variable=self.batch_2x2_var
        ).pack(anchor="w", padx=8, pady=2)
        ctk.CTkCheckBox(
            right,
            text="步驟 2：擴圖裁切加邊（最終檔）",
            variable=self.batch_expand_var,
        ).pack(anchor="w", padx=8, pady=2)

        ctk.CTkButton(
            right,
            text="對資料夾全部套用勾選步驟…",
            fg_color="#0d6e6a",
            hover_color="#085550",
            command=self.batch_apply_steps,
        ).pack(fill="x", padx=8, pady=(8, 3))

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
                text="裁切尺寸由 Excel 尺碼決定（僅批次步驟 2）" if sku else ""
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
        self.folder_files = []
        self.folder_index = -1
        self._update_folder_nav()
        self._load_image(Path(path))

    def load_folder(self) -> None:
        folder = filedialog.askdirectory(title="選擇要預覽／處理的資料夾")
        if not folder:
            return
        src_dir = Path(folder)
        files = sorted(
            p for p in src_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not files:
            messagebox.showinfo("提示", "資料夾內沒有支援的圖片")
            return
        self.folder_files = files
        self.folder_index = 0
        self._update_folder_nav()
        self._load_image(files[0])
        self.status_label.configure(
            text=f"已載入資料夾（{len(files)} 張）— 請預覽後按「處理目前圖片」或批次套用步驟"
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
        name = self.folder_files[self.folder_index].name if 0 <= self.folder_index < n else ""
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
        src = self.source_image.convert("RGB")
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

    def _process_worker(self) -> tuple[Image.Image, Image.Image, tuple[int, int, int], str, tuple[int, int]]:
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
            text=f"步驟 1 完成 — {mode}；背景 {rgb_to_hex(used_bg)}"
        )
        self._refresh_preview()

    def export_unit(self) -> None:
        if self.unit_image is None:
            messagebox.showinfo("提示", "請先按「處理目前圖片」產生單元圖")
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
            messagebox.showinfo("提示", "請先按「處理目前圖片」產生 2×2 預覽")
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

    # ---- 步驟 2：擴圖 ----
    def export_expand(self) -> None:
        if self.unit_image is None:
            messagebox.showinfo("提示", "請先完成步驟 1「處理目前圖片」")
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
            title="匯出擴圖（步驟 2）",
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
            self.status_label.configure(text=f"步驟 2 完成：{Path(str(result)).name}")

        self._run_async(worker, on_done=done)

    # ---- 批次：勾選步驟後套用 ----
    def batch_expand_only(self) -> None:
        """對資料夾只跑擴圖：記憶體生成單元圖 → 輸出 pipeline_out，不存中間檔。"""
        self.batch_unit_var.set(False)
        self.batch_2x2_var.set(False)
        self.batch_expand_var.set(True)
        self.batch_apply_steps()

    def batch_apply_steps(self) -> None:
        save_unit = self.batch_unit_var.get()
        save_2x2 = self.batch_2x2_var.get()
        do_expand = self.batch_expand_var.get()
        if not (save_unit or save_2x2 or do_expand):
            messagebox.showinfo("提示", "請至少勾選一個要套用的步驟")
            return

        sku_mode = self.sku_batch_var.get() and do_expand
        if sku_mode:
            title = "選擇父資料夾（內含 SKU 子資料夾）"
        elif self.folder_files:
            # 已載入資料夾：直接用該資料夾
            src_dir = self.folder_files[0].parent
            self._run_batch_on_dir(src_dir, save_unit, save_2x2, do_expand)
            return
        else:
            title = "選擇要批次套用步驟的資料夾"

        folder = filedialog.askdirectory(title=title)
        if not folder:
            return
        src_dir = Path(folder)

        if sku_mode:
            excel_raw = self.excel_var.get().strip().strip('"')
            if not excel_raw:
                messagebox.showwarning("提示", "請先選擇 SKU 尺碼 Excel")
                return
            excel_path = Path(excel_raw)
            if not excel_path.is_file():
                messagebox.showerror("錯誤", f"Excel 不存在：\n{excel_path}")
                return
            if not any(p.is_dir() for p in src_dir.iterdir()):
                messagebox.showinfo("提示", "父資料夾下沒有子資料夾")
                return
            # SKU 模式走完整擴圖流水線（步驟 2）；單元圖／2×2 另存若有勾選
            self._run_sku_batch(src_dir, excel_path, save_unit, save_2x2, do_expand)
            return

        self._run_batch_on_dir(src_dir, save_unit, save_2x2, do_expand)

    def _run_batch_on_dir(
        self,
        src_dir: Path,
        save_unit: bool,
        save_2x2: bool,
        do_expand: bool,
    ) -> None:
        files = sorted(
            p
            for p in src_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not files:
            messagebox.showinfo("提示", "資料夾內沒有支援的圖片")
            return

        expand: ExpandSettings | None = None
        expand_ext = ".tif"
        if do_expand:
            try:
                expand = self._expand_settings()
                expand_ext = normalize_expand_ext(self.expand_format_var.get())
            except ValueError as exc:
                messagebox.showwarning(
                    "參數錯誤", str(exc) or "請檢查擴圖數值是否為有效數字"
                )
                return

        out_dir = src_dir / "output"
        expand_out = src_dir / "pipeline_out"
        margin, is_percent, threshold = self._params()
        auto_bg = self.auto_bg_var.get()
        manual_bg = self.bg_rgb

        def worker() -> tuple[int, Path]:
            ok = 0
            total = len(files)
            if save_unit or save_2x2:
                out_dir.mkdir(exist_ok=True)
            if do_expand:
                expand_out.mkdir(exist_ok=True)

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
                    if save_unit:
                        unit.save(out_dir / f"{path.stem}_unit.png")
                    if save_2x2:
                        preview, _, _ = tile_2x2_multi(unit)
                        preview.save(out_dir / f"{path.stem}_2x2.png")
                    if do_expand:
                        assert expand is not None
                        dest = expand_out / f"{path.stem}{expand_ext}"
                        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                            tmp_path = Path(tmp.name)
                        try:
                            unit.save(tmp_path)
                            expand_unit(tmp_path, dest, expand)
                        finally:
                            tmp_path.unlink(missing_ok=True)
                    ok += 1
                except (OSError, ValueError):
                    pass
                self.after(0, lambda v=(i + 1) / total: self.progress.set(v))

            primary = expand_out if do_expand else out_dir
            return ok, primary

        def done(result: object, error: BaseException | None) -> None:
            if error:
                messagebox.showerror("批次失敗", str(error))
                return
            ok, primary = result  # type: ignore[misc]
            parts = [f"成功 {ok}/{len(files)}"]
            if save_unit or save_2x2:
                parts.append(f"步驟 1 輸出：{out_dir}")
            if do_expand:
                parts.append(f"步驟 2 輸出：{expand_out}")
            messagebox.showinfo("批次完成", "\n".join(parts))
            self.progress.set(0)

        self._run_async(worker, on_done=done)

    def _run_sku_batch(
        self,
        src_dir: Path,
        excel_path: Path,
        save_unit: bool,
        save_2x2: bool,
        do_expand: bool,
    ) -> None:
        """SKU 模式：步驟 2 走 Excel 尺碼；可選另存單元圖／2×2。"""
        if not do_expand and not (save_unit or save_2x2):
            messagebox.showinfo("提示", "請至少勾選一個步驟")
            return
        try:
            expand = self._expand_settings()
            expand_format = self.expand_format_var.get()
            if do_expand:
                normalize_expand_ext(expand_format)
        except ValueError as exc:
            messagebox.showwarning(
                "參數錯誤", str(exc) or "請檢查擴圖數值是否為有效數字"
            )
            return

        out_dir = src_dir / "pipeline_out"
        units_dir = src_dir / "output"
        margin, is_percent, threshold = self._params()
        auto_bg = self.auto_bg_var.get()
        manual_bg = self.bg_rgb
        logs: list[str] = []

        def worker() -> tuple[int, int, Path]:
            # 步驟 2：SKU 擴圖流水線
            if do_expand:
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
                ok = sum(1 for r in results if not r.error)
                total = len(results)
            else:
                ok, total = 0, 0

            # 可選：對各 SKU 子資料夾另存單元圖／2×2
            if save_unit or save_2x2:
                units_dir.mkdir(exist_ok=True)
                subdirs = [p for p in src_dir.iterdir() if p.is_dir() and p.name != "output" and p.name != "pipeline_out"]
                jobs = [
                    f
                    for d in subdirs
                    for f in sorted(d.iterdir())
                    if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
                ]
                total = max(total, len(jobs))
                for i, path in enumerate(jobs):
                    try:
                        img = Image.open(path)
                        img.load()
                        use_bg = None if auto_bg else manual_bg
                        unit, _ = make_seamless_hard_cut(
                            img,
                            bg=use_bg,
                            margin=margin,
                            threshold=threshold,
                            margin_is_percent=is_percent,
                        )
                        sub_out = units_dir / path.parent.name
                        sub_out.mkdir(exist_ok=True)
                        if save_unit:
                            unit.save(sub_out / f"{path.stem}_unit.png")
                        if save_2x2:
                            preview, _, _ = tile_2x2_multi(unit)
                            preview.save(sub_out / f"{path.stem}_2x2.png")
                        if not do_expand:
                            ok += 1
                    except OSError:
                        pass
                    self.after(0, lambda v=(i + 1) / max(1, len(jobs)): self.progress.set(v))

            return ok, total or 1, out_dir if do_expand else units_dir

        def done(result: object, error: BaseException | None) -> None:
            if error:
                messagebox.showerror("SKU 批次失敗", str(error))
                return
            ok, total, out = result  # type: ignore[misc]
            detail = "\n".join(logs[-8:]) if logs else ""
            messagebox.showinfo(
                "SKU 批次完成",
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
