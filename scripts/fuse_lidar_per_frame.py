import os
import numpy as np
import open3d as o3d
from truckscenes import TruckScenes

# 所有LiDAR通道名（按你数据实际补全）
LIDAR_CHANNELS = [
    "LIDAR_TOP_FRONT", "LIDAR_TOP_LEFT", "LIDAR_TOP_RIGHT",
    "LIDAR_LEFT", "LIDAR_RIGHT", "LIDAR_REAR"
]

def load_pointcloud(filepath):
    if filepath.endswith('.bin'):
        try:
            points = np.fromfile(filepath, dtype=np.float32).reshape(-1, 5)[:, :3]
            return points
        except Exception as e:
            print(f"读取 LiDAR bin 文件失败: {filepath}，错误: {e}")
            return None
    elif filepath.endswith('.pcd'):
        try:
            pcd = o3d.io.read_point_cloud(filepath)
            return np.asarray(pcd.points)
        except Exception as e:
            print(f"读取 pcd 文件失败: {filepath}，错误: {e}")
            return None
    else:
        print(f"不支持的点云格式: {filepath}")
        return None

def fuse_lidar_for_sample(truckscenes, sample_token, output_dir):
    all_points = []
    sample = truckscenes.get('sample', sample_token)
    for channel in LIDAR_CHANNELS:
        if channel in sample['data']:
            sample_data_token = sample['data'][channel]
            file_path = truckscenes.get_sample_data_path(sample_data_token)
            points = load_pointcloud(file_path)
            if points is not None and points.size > 0:
                all_points.append(points)
    if not all_points:
        print(f"⚠️ 没有找到可用LiDAR数据: {sample_token}")
        return
    merged_points = np.vstack(all_points)
    print(f"合并后点数: {merged_points.shape[0]}")

    # Open3D可视化
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(merged_points)
    o3d.visualization.draw_geometries([pcd], window_name="Merged LiDAR Fusion")

    # 保存
    os.makedirs(output_dir, exist_ok=True)
    ply_path = os.path.join(output_dir, f"{sample_token}_merged_lidar.ply")
    o3d.io.write_point_cloud(ply_path, pcd)
    print(f"已保存为: {ply_path}")

if __name__ == "__main__":
    dataroot = r"D:/MyProjects/AV4/data/man-truckscenes"
    truckscenes = TruckScenes(version='v1.0-mini', dataroot=dataroot)
    # 你可以更改sample_token为你要分析的那一帧
    sample_token = truckscenes.sample[0]['token']
    fuse_lidar_for_sample(truckscenes, sample_token, output_dir="output/merged_lidar")
