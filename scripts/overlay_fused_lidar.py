import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from truckscenes import TruckScenes

# 1. 列出所有 LiDAR 通道
LIDAR_CHANNELS = [
    "LIDAR_TOP_FRONT", "LIDAR_TOP_LEFT", "LIDAR_TOP_RIGHT",
    "LIDAR_LEFT", "LIDAR_RIGHT", "LIDAR_REAR"
]

def overlay_each_lidar(ts, sample_token, camera_channel, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    sample = ts.get('sample', sample_token)
    for chan in LIDAR_CHANNELS:
        if chan not in sample['data']:
            continue
        out_png = os.path.join(output_dir, f"{sample_token}_{chan}.png")
        ts.render_pointcloud_in_image(
            sample_token,
            pointsensor_channel=chan,
            camera_channel=camera_channel,
            render_intensity=False,  # 按高度上色更直观
            dot_size=2,
            out_path=out_png
        )
        print(f"✅ Generated overlay: {out_png}")
        paths.append(out_png)
    return paths

def fuse_overlays(overlay_paths, final_path):
    # 2. 读第一张做底图，再把后续叠加
    base = mpimg.imread(overlay_paths[0])
    fig, ax = plt.subplots(figsize=(12,6))
    ax.imshow(base)
    for p in overlay_paths[1:]:
        img = mpimg.imread(p)
        ax.imshow(img, alpha=0.5)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(final_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"✅ Fused all LiDAR overlays to {final_path}")

if __name__ == "__main__":
    ts = TruckScenes(
        version='v1.0-mini',
        dataroot=r"D:/MyProjects/AV4/data/man-truckscenes",
        verbose=False
    )
    sample_token = ts.sample[0]['token']
    camera_channel = 'CAMERA_LEFT_FRONT'
    tmp_dir = f"output/overlays/{sample_token}/lidar_channels"
    overlay_paths = overlay_each_lidar(ts, sample_token, camera_channel, tmp_dir)

    final_png = f"output/overlays/{sample_token}_all_lidar_fused.png"
    fuse_overlays(overlay_paths, final_png)
