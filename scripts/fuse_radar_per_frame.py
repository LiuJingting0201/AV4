import os
import numpy as np
import open3d as o3d
from truckscenes import TruckScenes

# Radar通道名，按你的数据实际补全
RADAR_CHANNELS = [
    "RADAR_LEFT_BACK", "RADAR_LEFT_FRONT", "RADAR_LEFT_SIDE",
    "RADAR_RIGHT_BACK", "RADAR_RIGHT_FRONT", "RADAR_RIGHT_SIDE"
]

def load_pointcloud(filepath):
    # 适配Radar点云格式（通常为.pcd）
    if filepath.endswith('.pcd'):
        try:
            pcd = o3d.io.read_point_cloud(filepath)
            return np.asarray(pcd.points)
        except Exception as e:
            print(f"读取 pcd 文件失败: {filepath}，错误: {e}")
            return None
    elif filepath.endswith('.bin'):
        try:
            # 如果你的Radar是bin格式
            points = np.fromfile(filepath, dtype=np.float32).reshape(-1, 5)[:, :3]
            return points
        except Exception as e:
            print(f"读取 Radar bin 文件失败: {filepath}，错误: {e}")
            return None
    else:
        print(f"不支持的Radar点云格式: {filepath}")
        return None

def fuse_radar_for_sample(truckscenes, sample_token, output_dir):
    all_points = []
    sample = truckscenes.get('sample', sample_token)
    for channel in RADAR_CHANNELS:
        if channel in sample['data']:
            sample_data_token = sample['data'][channel]
            file_path = truckscenes.get_sample_data_path(sample_data_token)
            points = load_pointcloud(file_path)
            if points is not None and points.size > 0:
                all_points.append(points)
    if not all_points:
        print(f"⚠️ 没有找到可用Radar数据: {sample_token}")
        return
    merged_points = np.vstack(all_points)
    print(f"合并后Radar点数: {merged_points.shape[0]}")

    # 可视化
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(merged_points)
    o3d.visualization.draw_geometries([pcd], window_name="Merged Radar Fusion")

    # 保存
    os.makedirs(output_dir, exist_ok=True)
    ply_path = os.path.join(output_dir, f"{sample_token}_merged_radar.ply")
    o3d.io.write_point_cloud(ply_path, pcd)
    print(f"已保存为: {ply_path}")

if __name__ == "__main__":
    dataroot = r"D:/MyProjects/AV4/data/man-truckscenes"
    truckscenes = TruckScenes(version='v1.0-mini', dataroot=dataroot)
    sample_token = truckscenes.sample[0]['token']
    fuse_radar_for_sample(truckscenes, sample_token, output_dir="output/merged_radar")
