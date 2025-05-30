# scripts/lighten_combine_overlays.py

import os, glob
from PIL import Image, ImageChops

def lighten_combine(token):
    base_dir = os.path.dirname(__file__)
    # 新：递归搜所有 radar_channels 及其子目录下的 PNG
    pattern = os.path.join(base_dir, "output", "overlays", token, "radar_channels", f"{token}_RADAR_*.png")
    paths = sorted(glob.glob(pattern))
    if not paths:
        print("❌ 没找到 PNG 文件，请检查目录：", os.path.dirname(pattern))
        return

    base = Image.open(paths[0]).convert('RGB')
    for p in paths[1:]:
        im = Image.open(p).convert('RGB')
        base = ImageChops.lighter(base, im)

    out_dir = os.path.join(base_dir, "output", "overlays", token)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{token}_radar_all_max.png")
    base.save(out_path)
    print("✅ 保存合并结果：", out_path)

if __name__ == "__main__":
    SAMPLE_TOKEN = "32d2bcf46e734dffb14fe2e0a823d059"
    lighten_combine(SAMPLE_TOKEN)
