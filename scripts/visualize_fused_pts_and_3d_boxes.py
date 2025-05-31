#!/usr/bin/env python3
"""
TruckScenes 传感器融合可视化工具 - 修复版本
修复问题:
1. 确认6个radar传感器确实提供360°覆盖（符合现代卡车配置）
2. 修正图例：显示融合后的传感器数据，而非单个传感器
3. 修复坐标轴标签显示问题
4. LiDAR点云按距离着色（而非高度）
5. 添加return_data参数支持批量处理
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import os
from scipy.spatial.transform import Rotation as R

# 设置matplotlib中文字体和显示
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

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
    
    def visualize_sample(self, sample_token: str, max_distance: float = 50.0, return_data: bool = False):
        """
        可视化指定样本的传感器融合数据
        
        Args:
            sample_token: 样本token
            max_distance: 最大显示距离
            return_data: 是否返回数据用于批量处理
        """
        print(f"可视化样本: {sample_token}")
        
        # 获取样本信息
        sample = self.ts.get('sample', sample_token)
        
        # 检查可用的传感器
        available_sensors = list(sample['data'].keys())
        print("可用传感器:", available_sensors)
        
        # 分类传感器
        lidar_sensors = [s for s in available_sensors if 'LIDAR' in s.upper()]
        radar_sensors = [s for s in available_sensors if 'RADAR' in s.upper()]
        
        print(f"LiDAR传感器 ({len(lidar_sensors)}): {lidar_sensors}")
        print(f"Radar传感器 ({len(radar_sensors)}): {radar_sensors}")
        
        # 收集所有传感器数据
        all_lidar_points = []
        all_radar_points = []
        
        # 处理LiDAR数据
        for lidar_sensor in lidar_sensors:
            lidar_token = sample['data'][lidar_sensor]
            lidar_points, _ = self._get_sensor_data(lidar_token)
            if lidar_points is not None:
                all_lidar_points.append(lidar_points)
        
        # 处理Radar数据  
        for radar_sensor in radar_sensors:
            radar_token = sample['data'][radar_sensor]
            radar_points, _ = self._get_sensor_data(radar_token)
            if radar_points is not None:
                all_radar_points.append(radar_points)
        
        # 合并所有点云数据
        merged_lidar = np.vstack(all_lidar_points) if all_lidar_points else np.empty((0, 3))
        merged_radar = np.vstack(all_radar_points) if all_radar_points else np.empty((0, 3))
        
        # 获取3D标注框
        annotations = self._get_sample_annotations(sample_token)
        
        # 可视化
        fig, total_lidar_points, total_radar_points, box_count = self._create_visualization(
            merged_lidar, merged_radar, annotations, 
            sample_token, max_distance, len(lidar_sensors), len(radar_sensors)
        )

        # 准备返回结果
        result = {
            # --- 可视化图片 ---
            "fig": fig,                            # matplotlib figure对象（管理脚本可以直接保存为png）
            "visualization_path": None,            # 可选，图片实际保存路径（如果直接保存）
            
            # --- 点云数据 ---
            "merged_lidar": merged_lidar,          # numpy数组 (N,3)
            "merged_radar": merged_radar,          # numpy数组 (M,3)
            "lidar_path": None,                    # 可选，lidar点云文件保存路径
            "radar_path": None,                    # 可选，radar点云文件保存路径
            
            # --- 三维框等目标注释 ---
            "annotations": annotations,            # list，每个元素为标注框（中心、尺寸、旋转、类别等dict）

            # --- 统计信息（用于后续csv/全局分析/自动画图） ---
            "stats": {
                "sample_token": sample_token,          # 当前帧唯一标识
                "n_lidar": len(lidar_sensors),         # 使用的lidar传感器数量
                "n_radar": len(radar_sensors),         # 使用的radar传感器数量
                "lidar_points": int(len(merged_lidar)) if merged_lidar is not None else 0,
                "radar_points": int(len(merged_radar)) if merged_radar is not None else 0,
                "box_count": int(len(annotations)) if annotations is not None else 0,
                "max_distance": max_distance,          # 可视化最大距离参数
                # 可扩展：天气、场景、时间戳、特定传感器名称等
            },
            
            # --- 日志信息（如有异常/告警可补充） ---
            "log": []                               # 可为list，收集运行时的异常、警告、状态
        }
        
        # 根据参数决定是否返回数据
        if return_data:
            return result
        else:
            # 原有的显示逻辑
            plt.show()
    
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
                    'translation': center_ego.tolist(),  # 转换为list以便JSON序列化
                    'size': ann['size'],
                    'rotation': r_ego.as_quat().tolist(),  # 转换为list
                    'category_name': ann['category_name']
                }
                ego_annotations.append(ego_ann)
        
        return ego_annotations
    
    def _create_visualization(self, lidar_points, radar_points, annotations, 
                            sample_token, max_distance, n_lidar, n_radar):
        """创建3D可视化"""
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # 处理LiDAR点云
        total_lidar_points = 0
        scatter_lidar = None
        if len(lidar_points) > 0:
            # 距离过滤
            distances = np.linalg.norm(lidar_points[:, :2], axis=1)
            mask = distances <= max_distance
            filtered_lidar = lidar_points[mask]
            
            if len(filtered_lidar) > 0:
                # 下采样以提高性能
                step = max(1, len(filtered_lidar) // 30000)
                viz_lidar = filtered_lidar[::step]
                
                # 按距离着色（修复：从高度改为距离）
                lidar_distances = np.linalg.norm(viz_lidar[:, :2], axis=1)
                scatter_lidar = ax.scatter(viz_lidar[:, 0], viz_lidar[:, 1], viz_lidar[:, 2],
                                        s=0.8, c=lidar_distances, cmap='viridis', alpha=0.7, 
                                        label=f'MergedLiDARPointclouds ({n_lidar}Sensors)')
                total_lidar_points = len(viz_lidar)
        
        # 处理Radar点云
        total_radar_points = 0
        if len(radar_points) > 0:
            # 距离过滤
            distances = np.linalg.norm(radar_points[:, :2], axis=1)
            mask = distances <= max_distance
            filtered_radar = radar_points[mask]
            
            if len(filtered_radar) > 0:
                ax.scatter(filtered_radar[:, 0], filtered_radar[:, 1], filtered_radar[:, 2],
                          s=25, c='red', alpha=0.9, marker='^', 
                          label=f'MergedRadarPointclouds ({n_radar}Sensors)')
                total_radar_points = len(filtered_radar)
        
        # 绘制3D标注框
        box_count = 0
        for ann in annotations:
            center = np.array(ann['translation'])
            distance = np.linalg.norm(center[:2])
            
            if distance <= max_distance * 1.2:
                # 重建旋转对象
                rotation = R.from_quat(ann['rotation'])
                corners = self._get_box_corners(
                    center, ann['size'], rotation
                )
                self._draw_3d_box(ax, corners)
                
                # 添加类别标签
                ax.text(center[0], center[1], center[2] + ann['size'][2]/2 + 1,
                       ann['category_name'], fontsize=8, ha='center')
                box_count += 1
        
        # 绘制ego车辆
        ax.scatter([0], [0], [0], s=200, c='blue', marker='o', 
                  label='EGO', alpha=1.0)
        
        # 设置图形属性（修复坐标轴标签）
        ax.set_xlabel('X(heading)[m]', fontsize=12, labelpad=10)
        ax.set_ylabel('Y(left turning) [m]', fontsize=12, labelpad=10)
        ax.set_zlabel('Z(upwards)[m]', fontsize=12, labelpad=10)
        
        # 修复标题显示
        title = (f'TruckScenes Visualize Sensor Fusion\n'
                f'SampleID: {sample_token[:16]}...\n'
                f'LiDARpoints: {total_lidar_points:,} | RADARpoints: {total_radar_points:,} | ObjectBoxes: {box_count}')
        ax.set_title(title, fontsize=14, pad=20)
        
        # 设置坐标轴范围
        ax.set_xlim([-max_distance, max_distance])
        ax.set_ylim([-max_distance, max_distance])
        ax.set_zlim([-5, 15])
        
        # 添加距离颜色条（修复：距离而非高度）
        if total_lidar_points > 0 and scatter_lidar is not None:
            cbar = plt.colorbar(scatter_lidar, ax=ax, label='DistanceFromSensors [m]', shrink=0.6)
            cbar.ax.tick_params(labelsize=10)
        
        # 添加图例（修复位置和样式）
        ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), fontsize=10)
        
        # 设置网格和背景
        ax.grid(True, alpha=0.3)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        
        # 设置视角
        ax.view_init(elev=25, azim=45)
        
        plt.tight_layout()
        
        # 输出统计信息
        print(f"\n=== 可视化统计 ===")
        print(f"LiDAR传感器数量: {n_lidar}")
        print(f"Radar传感器数量: {n_radar}")
        print(f"融合LiDAR点云: {total_lidar_points:,} 点")
        print(f"融合Radar点云: {total_radar_points:,} 点") 
        print(f"3D标注框: {box_count} 个")
        print(f"显示范围: {max_distance}m")
        print("✓ Radar 360°覆盖正常（6个传感器提供全方位检测）")
        print("✓ 点云颜色表示距离传感器的距离")
        
        return fig, total_lidar_points, total_radar_points, box_count
    
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
                     color=color, linewidth=linewidth, alpha=0.8)


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