"""擴圖工具（kuotu）路徑解析。"""

from __future__ import annotations

import sys
from pathlib import Path

# seamless_tile/app/paths.py → parents[1] = seamless_tile 專案根
_PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 優先使用專案內打包的 kuotu；開發時可回退到相鄰的 5EDemocache/kuotu
_BUNDLED_KUOTU = _PROJECT_ROOT / "kuotu"
_CACHE_ROOT = Path(__file__).resolve().parents[3]
_SIBLING_KUOTU = _CACHE_ROOT / "kuotu"


def _resolve_kuotu_root() -> Path:
    for candidate in (_BUNDLED_KUOTU, _SIBLING_KUOTU):
        if (candidate / "image_pipeline.py").is_file():
            return candidate.resolve()
    return _BUNDLED_KUOTU.resolve()


KUOTU_ROOT = _resolve_kuotu_root()


def ensure_kuotu_on_path() -> Path:
    root = _resolve_kuotu_root()
    if not (root / "image_pipeline.py").is_file():
        raise FileNotFoundError(
            f"找不到擴圖工具：{root}\n"
            "請確認 seamless_tile/kuotu/image_pipeline.py 存在。"
        )
    s = str(root)
    if s not in sys.path:
        sys.path.insert(0, s)
    return root
