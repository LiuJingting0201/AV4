import os
import numpy as np
import open3d as o3d
from truckscenes import TruckScenes

def load_pointcloud(filepath):
    """
    自动识别点云格式并加载
    LiDAR 使用 .bin 文件，5通道 float32，取前三维
    Radar 使用 .pcd 文件，Open3D读取
    """
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
            print(f"读取 Radar pcd 文件失败: {filepath}，错误: {e}")
            return None
    else:
        print(f"不支持的点云格式: {filepath}")
        return None

def export_pointcloud(truckscenes, sample_token, sensor_channel, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    sample_data_token = truckscenes.get('sample', sample_token)['data'][sensor_channel]
    pointcloud_path = truckscenes.get_sample_data_path(sample_data_token)
    points = load_pointcloud(pointcloud_path)
    if points is None:
        print(f"跳过 {sensor_channel} 的点云导出，路径: {pointcloud_path}")
        return None

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    ply_filename = f"{sample_token}_{sensor_channel}.ply"
    ply_filepath = os.path.join(output_dir, ply_filename)
    o3d.io.write_point_cloud(ply_filepath, pcd)
    print(f"导出点云成功: {ply_filepath}")
    return ply_filepath

def batch_export(truckscenes, sensor_channels, max_samples=10):
    base_output_dir = "output/pointclouds"
    os.makedirs(base_output_dir, exist_ok=True)

    for sample in truckscenes.sample[:max_samples]:
        sample_token = sample['token']
        for sensor in sensor_channels:
            export_pointcloud(
                truckscenes,
                sample_token,
                sensor,
                os.path.join(base_output_dir, sensor)
            )

def view_pointcloud(ply_path):
    pcd = o3d.io.read_point_cloud(ply_path)
    o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    dataroot = r"D:/MyProjects/AV4/data/man-truckscenes"
    truckscenes = TruckScenes(version='v1.0-mini', dataroot=dataroot)

    sensors = ['LIDAR_TOP_FRONT', 'RADAR_LEFT_FRONT']
    batch_export(truckscenes, sensors, max_samples=20)

    # 示例：查看导出的第一条点云
    import glob
    ply_files = glob.glob("output/pointclouds/**/*.ply", recursive=True)
    if ply_files:
        print(f"展示点云文件: {ply_files[0]}")
        view_pointcloud(ply_files[0])
