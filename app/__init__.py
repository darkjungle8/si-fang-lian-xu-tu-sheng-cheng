"""四方連續圖處理工具。"""

# 生產圖／印刷級畫布常超過 Pillow 預設 ~89M 像素上限；本工具只處理本機可信檔案。
try:
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
except ImportError:
    pass
