"""四方連續圖 GUI 工具入口。"""

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_local_package() -> None:
    """優先使用本專案的 app/，避免被 ComfyUI 等環境裡的同名套件遮蔽。"""
    root = Path(__file__).resolve().parent
    root_s = str(root)
    while root_s in sys.path:
        sys.path.remove(root_s)
    sys.path.insert(0, root_s)
    for name in list(sys.modules):
        if name == "app" or name.startswith("app."):
            del sys.modules[name]


_ensure_local_package()

from app.gui import run_app  # noqa: E402


def main() -> None:
    run_app()


if __name__ == "__main__":
    main()
