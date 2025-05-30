import os
from truckscenes import TruckScenes

def overlay_fusion(sample_token, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    ts = TruckScenes(
        version='v1.0-mini',
        dataroot=r"D:/MyProjects/AV4/data/man-truckscenes",
        verbose=False
    )

    # LiDAR ➕ Camera
    out_lidar = os.path.join(output_dir, f"{sample_token}_cam_lidar.png")
    ts.render_pointcloud_in_image(
        sample_token,
        pointsensor_channel='LIDAR_TOP_FRONT',
        camera_channel='CAMERA_LEFT_FRONT',
        render_intensity=False,    # 不用强度上色，用高度上色
        dot_size=2,                # 点的大小
        color_by='z',              # 按高度 z 值上色
        alpha=0.6,                 # 点透明度，便于看图像细节
        out_path=out_lidar
    )
    print(f"✅ LiDAR+Camera overlay saved to {out_lidar}")

    # Radar ➕ Camera
    out_radar = os.path.join(output_dir, f"{sample_token}_cam_radar.png")
    ts.render_pointcloud_in_image(
        sample_token,
        pointsensor_channel='RADAR_LEFT_FRONT',
        camera_channel='CAMERA_LEFT_FRONT',
        render_intensity=False,
        dot_size=3,
        color_by='velocity',       # 如果 Radar 有速度通道，可按速度上色
        alpha=0.8,
        out_path=out_radar
    )
    print(f"✅ Radar+Camera overlay saved to {out_radar}")

    # （可选）三模态合并可视化：把上面两张图用matplotlib叠加
    try:
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg

        img_cam = mpimg.imread(out_lidar)
        img_radar = mpimg.imread(out_radar)

        fig, ax = plt.subplots(figsize=(12,6))
        ax.imshow(img_cam)
        ax.imshow(img_radar, alpha=0.5)  # 半透明叠加
        ax.axis('off')
        tri_out = os.path.join(output_dir, f"{sample_token}_cam_lidar_radar.png")
        plt.savefig(tri_out, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        print(f"✅ 三模态叠加图 saved to {tri_out}")
    except ImportError:
        print("⚠️ matplotlib 未安装，跳过三模态叠加步骤")

if __name__ == "__main__":
    ts = TruckScenes(
        version='v1.0-mini',
        dataroot=r"D:/MyProjects/AV4/data/man-truckscenes",
        verbose=False
    )
    # 如果想测试第一个 sample，可以直接拿第一个 token
    sample_token = ts.sample[0]['token']
    overlay_fusion(sample_token, output_dir="output/overlays")
