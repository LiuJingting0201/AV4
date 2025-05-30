import os
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from truckscenes import TruckScenes

# 所有Radar通道
RADAR_CHANNELS = [
    "RADAR_LEFT_BACK", "RADAR_LEFT_FRONT", "RADAR_LEFT_SIDE",
    "RADAR_RIGHT_BACK", "RADAR_RIGHT_FRONT", "RADAR_RIGHT_SIDE"
]

# 以全Radar示例为准，LiDAR同理
def overlay_each_radar(ts, sample_token, camera_channel, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    sample = ts.get('sample', sample_token)
    for chan in RADAR_CHANNELS:
        if chan not in sample['data']:
            continue
        out_png = os.path.join(output_dir, f"{sample_token}_{chan}.png")
        ts.render_pointcloud_in_image(
            sample_token,
            pointsensor_channel=chan,
            camera_channel=camera_channel,
            render_intensity=True,  # 用强度上色
            dot_size=5,             # 放大到5~8
            out_path=out_png
        )
        paths.append(out_png)
    return paths

def fuse_overlays(overlay_paths, final_path):
    import matplotlib.pyplot as plt, matplotlib.image as mpimg
    base = mpimg.imread(overlay_paths[0])
    fig, ax = plt.subplots(figsize=(12,6))
    ax.imshow(base, alpha=1.0)  # 底图不透明
    for p in overlay_paths[1:]:
        img = mpimg.imread(p)
        ax.imshow(img, alpha=0.7)  # 只用0.7叠加，不要太多图层
    ax.axis('off')
    fig.savefig(final_path, bbox_inches='tight', pad_inches=0)




if __name__ == "__main__":
    ts = TruckScenes(
        version='v1.0-mini',
        dataroot=r"D:/MyProjects/AV4/data/man-truckscenes",
        verbose=False
    )
    sample_token = ts.sample[0]['token']
    camera_channel = 'CAMERA_LEFT_FRONT'
    temp_dir = f"output/overlays/{sample_token}/radar_channels"
    overlay_paths = overlay_each_radar(ts, sample_token, camera_channel, temp_dir)

    final_png = f"output/overlays/{sample_token}_all_radar_fused.png"
    fuse_overlays(overlay_paths, final_png)
