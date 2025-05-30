#!/usr/bin/env python3
"""
TruckScenes 传感器融合可视化工具
基于官方 truckscenes-devkit 的精简版本
解决点云和3D框不在同一坐标系的问题
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import os
from scipy.spatial.transform import Rotation as R

# 使用官方devkit（推荐）
try:
    from truckscenes.truckscenes import TruckScenes
    from truckscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
    from truckscenes.utils.geometry_utils import transform_matrix
    DEVKIT_AVAILABLE = True
    print("使用官方 TruckScenes devkit")
except ImportError:
    DEVKIT_AVAILABLE = False
    print("未找到官方devkit，使用简化实现")

class TruckScenesVisualizer:
    def __init__(self, dataroot: str, version: str = 'v1.0-mini'):
        """
        初始化可视化工具
        
        Args:
            dataroot: TruckScenes数据根目录
            version: 数据版本 ('v1.0-mini', 'v1.0-trainval', 'v1.0-test')
        """
        self.dataroot = dataroot
        self.version = version
        
        if DEVKIT_AVAILABLE:
            self.ts = TruckScenes(version=version, dataroot=dataroot, verbose=True)
        else:
            # 简化实现：直接加载JSON文件
            self.ts = self._load_simple_ts()
    
    def _load_simple_ts(self):
        """简化的TruckScenes数据加载"""
        class SimpleTruckScenes:
            def __init__(self, dataroot, version):
                self.dataroot = dataroot
                self.version = version
                
                # 加载基本表
                tables_path = os.path.join(dataroot, version)
                self.sample = self._load_table(os.path.join(tables_path, 'sample.json'))
                self.sample_data = self._load_table(os.path.join(tables_path, 'sample_data.json'))
                self.sample_annotation = self._load_table(os.path.join(tables_path, 'sample_annotation.json'))
                self.calibrated_sensor = self._load_table(os.path.join(tables_path, 'calibrated_sensor.json'))
                self.ego_pose = self._load_table(os.path.join(tables_path, 'ego_pose.json'))
                
            def _load_table(self, path):
                """加载JSON表并创建token映射"""
                with open(path, 'r') as f:
                    data = json.load(f)
                return {record['token']: record for record in data}
            
            def get(self, table_name, token):
                """获取记录"""
                table = getattr(self, table_name)
                return table[token]
        
        return SimpleTruckScenes(self.dataroot, self.version)
    
    def visualize_sample(self, sample_token: str, max_distance: float = 50.0):
        """
        可视化指定样本的传感器融合数据
        
        Args:
            sample_token: 样本token
            max_distance: 最大显示距离
        """
        print(f"可视化样本: {sample_token}")
        
        # 获取样本信息
        sample = self.ts.get('sample', sample_token)
        
        # 检查可用的传感器
        print("可用传感器:", list(sample['data'].keys()))
        
        # 获取所有传感器数据
        lidar_data = []
        radar_data = []
        
        # 查找LiDAR数据 - 尝试不同的可能命名
        lidar_keys = [key for key in sample['data'].keys() if 'LIDAR' in key.upper()]
        if not lidar_keys:
            # 如果没有LIDAR，尝试其他可能的命名
            lidar_keys = [key for key in sample['data'].keys() if any(x in key.upper() for x in ['VELODYNE', 'OUSTER', 'HESAI'])]
        
        print(f"找到LiDAR传感器: {lidar_keys}")
        
        for lidar_key in lidar_keys:
            lidar_token = sample['data'][lidar_key]
            lidar_points, lidar_pose = self._get_sensor_data(lidar_token)
            if lidar_points is not None:
                lidar_data.append((lidar_points, lidar_pose, lidar_key))
        
        # Radar数据
        radar_keys = [key for key in sample['data'].keys() if 'RADAR' in key.upper()]
        print(f"找到Radar传感器: {radar_keys}")
        
        for sensor_key in radar_keys:
            radar_token = sample['data'][sensor_key]
            radar_points, radar_pose = self._get_sensor_data(radar_token)
            if radar_points is not None:
                radar_data.append((radar_points, radar_pose, sensor_key))
        
        # 如果没有找到LiDAR，使用第一个点云传感器作为参考
        if not lidar_data:
            print("警告: 未找到LiDAR传感器，使用第一个可用传感器作为参考")
            all_sensors = list(sample['data'].keys())
            if all_sensors:
                ref_token = sample['data'][all_sensors[0]]
                ref_points, ref_pose = self._get_sensor_data(ref_token)
                if ref_points is not None:
                    lidar_data.append((ref_points, ref_pose, all_sensors[0]))
        
        # 获取3D标注框
        annotations = self._get_sample_annotations(sample_token)
        
        # 可视化
        self._create_visualization(
            lidar_data, radar_data, annotations, 
            sample_token, max_distance
        )
    
    def _get_sensor_data(self, sample_data_token: str):
        """获取传感器数据和姿态信息"""
        sample_data_record = self.ts.get('sample_data', sample_data_token)
        
        # 获取传感器校准和自车姿态
        cs_record = self.ts.get('calibrated_sensor', sample_data_record['calibrated_sensor_token'])
        ego_pose_record = self.ts.get('ego_pose', sample_data_record['ego_pose_token'])
        
        # 构建从传感器到全局坐标的变换矩阵
        sensor_to_ego = self._pose_to_matrix(cs_record)
        ego_to_global = self._pose_to_matrix(ego_pose_record)
        sensor_to_global = ego_to_global @ sensor_to_ego
        
        # 加载点云数据
        data_path = os.path.join(self.dataroot, sample_data_record['filename'])
        
        if not os.path.exists(data_path):
            print(f"数据文件不存在: {data_path}")
            return None, None
        
        if DEVKIT_AVAILABLE:
            # 使用官方devkit
            if 'LIDAR' in sample_data_record['channel']:
                pc = LidarPointCloud.from_file(data_path)
            else:
                pc = RadarPointCloud.from_file(data_path)
            points = pc.points[:3, :].T  # 转换为 (N, 3) 格式
        else:
            # 简化加载（假设是.bin文件）
            points = self._load_points_simple(data_path)
        
        if points is None or len(points) == 0:
            return None, None
        
        # 变换到全局坐标（以ego为参考）
        points_ego = self._transform_points_to_ego(points, sensor_to_global, ego_to_global)
        
        return points_ego, ego_pose_record
    
    def _load_points_simple(self, data_path: str):
        """简化的点云加载（支持.bin和.ply格式）"""
        if data_path.endswith('.bin'):
            # 加载二进制点云文件
            points = np.fromfile(data_path, dtype=np.float32)
            if len(points) % 4 == 0:  # LiDAR: x,y,z,intensity
                points = points.reshape(-1, 4)[:, :3]
            elif len(points) % 5 == 0:  # Radar: x,y,z,rcs,v_comp
                points = points.reshape(-1, 5)[:, :3]
            else:
                print(f"未知的点云格式: {data_path}")
                return None
        elif data_path.endswith('.ply'):
            # 加载PLY文件
            try:
                import open3d as o3d
                pcd = o3d.io.read_point_cloud(data_path)
                points = np.asarray(pcd.points)
            except ImportError:
                print("需要安装 open3d 来加载PLY文件: pip install open3d")
                return None
        else:
            print(f"不支持的文件格式: {data_path}")
            return None
        
        return points
    
    def _pose_to_matrix(self, pose_record):
        """将姿态记录转换为4x4变换矩阵"""
        translation = np.array(pose_record['translation'])
        rotation = np.array(pose_record['rotation'])  # [w, x, y, z]
        
        # 创建旋转矩阵
        r = R.from_quat([rotation[1], rotation[2], rotation[3], rotation[0]])  # [x,y,z,w]
        
        # 创建4x4变换矩阵
        transform = np.eye(4)
        transform[:3, :3] = r.as_matrix()
        transform[:3, 3] = translation
        
        return transform
    
    def _transform_points_to_ego(self, points, sensor_to_global, ego_to_global):
        """将点云从传感器坐标系变换到ego坐标系"""
        # 计算从全局到ego的逆变换
        global_to_ego = np.linalg.inv(ego_to_global)
        
        # 完整变换：sensor -> global -> ego
        sensor_to_ego = global_to_ego @ sensor_to_global
        
        # 变换点云
        points_homo = np.hstack([points, np.ones((len(points), 1))])
        points_ego = (sensor_to_ego @ points_homo.T).T[:, :3]
        
        return points_ego
    
    def _get_sample_annotations(self, sample_token: str):
        """获取样本的3D标注框"""
        # 获取该样本的所有标注
        sample_annotations = []
        
        if DEVKIT_AVAILABLE:
            # 使用官方devkit - 通过sample获取annotations
            sample_record = self.ts.get('sample', sample_token)
            
            # 获取与该sample关联的所有annotations
            ann_tokens = sample_record.get('anns', [])
            for ann_token in ann_tokens:
                ann_record = self.ts.get('sample_annotation', ann_token)
                sample_annotations.append(ann_record)
        else:
            # 简化实现 - 遍历所有annotations找到匹配的
            for ann_token, ann_record in self.ts.sample_annotation.items():
                if ann_record['sample_token'] == sample_token:
                    sample_annotations.append(ann_record)
        
        # 转换标注框到ego坐标系
        ego_annotations = []
        if sample_annotations:
            # 获取参考ego姿态（使用第一个传感器的时间戳）
            sample = self.ts.get('sample', sample_token)
            
            # 找到第一个可用的传感器数据作为参考
            first_sensor_key = list(sample['data'].keys())[0]
            first_sensor_data = self.ts.get('sample_data', sample['data'][first_sensor_key])
            ego_pose = self.ts.get('ego_pose', first_sensor_data['ego_pose_token'])
            ego_to_global = self._pose_to_matrix(ego_pose)
            global_to_ego = np.linalg.inv(ego_to_global)
            
            for ann in sample_annotations:
                # 变换中心点
                center_global = np.array(ann['translation'])
                center_ego = (global_to_ego @ np.append(center_global, 1))[:3]
                
                # 变换旋转
                r_global = R.from_quat([ann['rotation'][1], ann['rotation'][2], 
                                      ann['rotation'][3], ann['rotation'][0]])
                r_ego_transform = R.from_matrix(global_to_ego[:3, :3])
                r_ego = r_ego_transform * r_global
                
                ego_ann = {
                    'translation': center_ego,
                    'size': ann['size'],
                    'rotation': r_ego,
                    'category_name': ann['category_name']
                }
                ego_annotations.append(ego_ann)
        
        return ego_annotations
    
    def _create_visualization(self, lidar_data, radar_data, annotations, sample_token, max_distance):
        """创建3D可视化"""
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制LiDAR点云
        total_lidar_points = 0
        for lidar_points, pose, sensor_name in lidar_data:
            # 距离过滤
            distances = np.linalg.norm(lidar_points[:, :2], axis=1)
            mask = distances <= max_distance
            filtered_points = lidar_points[mask]
            
            if len(filtered_points) > 0:
                # 下采样以提高性能
                step = max(1, len(filtered_points) // 20000)
                viz_points = filtered_points[::step]
                
                # 按高度着色
                colors = viz_points[:, 2]
                scatter = ax.scatter(viz_points[:, 0], viz_points[:, 1], viz_points[:, 2],
                                   s=1, c=colors, cmap='viridis', alpha=0.6, 
                                   label=f'LiDAR ({sensor_name})' if total_lidar_points == 0 else "")
                total_lidar_points += len(viz_points)
        
        # 绘制Radar点云
        total_radar_points = 0
        for radar_points, pose, sensor_name in radar_data:
            distances = np.linalg.norm(radar_points[:, :2], axis=1)
            mask = distances <= max_distance
            filtered_points = radar_points[mask]
            
            if len(filtered_points) > 0:
                ax.scatter(filtered_points[:, 0], filtered_points[:, 1], filtered_points[:, 2],
                          s=20, c='red', alpha=0.8, marker='^', 
                          label=f'Radar ({sensor_name})' if total_radar_points == 0 else "")
                total_radar_points += len(filtered_points)
        
        # 绘制3D标注框
        box_count = 0
        for ann in annotations:
            center = ann['translation']
            distance = np.linalg.norm(center[:2])
            
            if distance <= max_distance * 1.2:
                corners = self._get_box_corners(
                    center, ann['size'], ann['rotation']
                )
                self._draw_3d_box(ax, corners)
                
                # 添加类别标签
                ax.text(center[0], center[1], center[2] + ann['size'][2]/2 + 1,
                       ann['category_name'], fontsize=8, ha='center')
                box_count += 1
        
        # 绘制ego车辆
        ax.scatter([0], [0], [0], s=200, c='blue', marker='o', 
                  label='Ego Vehicle', alpha=1.0)
        
        # 设置图形属性
        ax.set_xlabel('X (前进方向) [m]')
        ax.set_ylabel('Y (左侧方向) [m]')
        ax.set_zlabel('Z (向上方向) [m]')
        ax.set_title(f'TruckScenes 传感器融合可视化\n'
                    f'样本: {sample_token[:16]}...\n'
                    f'LiDAR: {total_lidar_points}, Radar: {total_radar_points}, 标注框: {box_count}')
        
        # 设置坐标轴范围
        ax.set_xlim([-max_distance, max_distance])
        ax.set_ylim([-max_distance, max_distance])
        ax.set_zlim([-5, 15])
        
        # 添加颜色条
        if total_lidar_points > 0:
            plt.colorbar(scatter, ax=ax, label='高度 [m]', shrink=0.5)
        
        # 添加图例
        ax.legend(loc='upper right')
        
        # 设置视角
        ax.view_init(elev=20, azim=45)
        
        plt.tight_layout()
        plt.show()
        
        print(f"可视化完成！显示了 {total_lidar_points} 个LiDAR点, "
              f"{total_radar_points} 个Radar点, {box_count} 个3D标注框")
    
    def _get_box_corners(self, center, size, rotation):
        """计算3D标注框的8个角点"""
        w, l, h = size
        
        # 在物体坐标系中的8个角点
        corners = np.array([
            [-l/2, -w/2, -h/2], [+l/2, -w/2, -h/2],
            [+l/2, +w/2, -h/2], [-l/2, +w/2, -h/2],
            [-l/2, -w/2, +h/2], [+l/2, -w/2, +h/2],
            [+l/2, +w/2, +h/2], [-l/2, +w/2, +h/2]
        ])
        
        # 应用旋转和平移
        if hasattr(rotation, 'as_matrix'):
            # scipy.spatial.transform.Rotation对象
            rotation_matrix = rotation.as_matrix()
        else:
            # 假设是四元数数组 [w, x, y, z]
            r = R.from_quat([rotation[1], rotation[2], rotation[3], rotation[0]])
            rotation_matrix = r.as_matrix()
        
        corners_world = corners @ rotation_matrix.T + np.array(center)
        return corners_world
    
    def _draw_3d_box(self, ax, corners, color='orange', linewidth=2):
        """绘制3D标注框"""
        # 定义12条边
        edges = [
            [0,1], [1,2], [2,3], [3,0],  # 底面
            [4,5], [5,6], [6,7], [7,4],  # 顶面
            [0,4], [1,5], [2,6], [3,7]   # 垂直边
        ]
        
        for edge in edges:
            start, end = corners[edge[0]], corners[edge[1]]
            ax.plot3D([start[0], end[0]], [start[1], end[1]], [start[2], end[2]],
                     color=color, linewidth=linewidth)


def main():
    """主函数 - 使用示例"""
    # 配置参数
    sample_token = "32d2bcf46e734dffb14fe2e0a823d059"
    dataroot = "../data/man-truckscenes"  # 根据实际路径调整
    version = "v1.0-mini"
    max_distance = 50.0
    
    try:
        # 创建可视化工具
        print("初始化 TruckScenes 可视化工具...")
        visualizer = TruckScenesVisualizer(dataroot, version)
        
        # 可视化指定样本
        visualizer.visualize_sample(sample_token, max_distance)
        
    except Exception as e:
        print(f"错误: {e}")
        print("\n请检查:")
        print("1. 数据路径是否正确")
        print("2. 是否安装了必要的依赖包: pip install truckscenes-devkit[all]")
        print("3. 样本token是否存在于数据集中")


if __name__ == "__main__":
    main()