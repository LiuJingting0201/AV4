#!/usr/bin/env python3
"""
TruckScenes 输出管理模块
功能：
1. 批量处理选定的样本
2. 保存可视化图片、点云数据、统计信息
3. 生成汇总报告和CSV文件
4. 错误处理和日志记录
"""

import os
import json
import traceback
import pandas as pd
import numpy as np
import inspect
from datetime import datetime
import matplotlib.pyplot as plt

# 尝试导入open3d（用于保存点云）
try:
    import open3d as o3d
    O3D_AVAILABLE = True
    print("✓ Open3D available for point cloud saving")
except ImportError:
    O3D_AVAILABLE = False
    print("⚠️ Open3D not available - point clouds will not be saved")

# 导入主可视化类
try:
    from visualize_fused_pts_and_3d_boxes import TruckScenesVisualizer
    print("✓ TruckScenesVisualizer imported successfully")
except ImportError as e:
    print(f"❌ Failed to import TruckScenesVisualizer: {e}")
    print("请确保 visualize_fused_pts_and_3d_boxes.py 在同一目录下")


# ==== 1. 路径与配置 ====
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "man-truckscenes")
CONFIGS_DIR = os.path.join(BASE_DIR, "configs")
OUTPUT_ROOT = os.path.join(BASE_DIR, "output")
SELECTED_JSON = os.path.join(CONFIGS_DIR, "selected_samples.json")
SUMMARY_DIR = os.path.join(OUTPUT_ROOT, "summary")

# 确保目录存在
os.makedirs(SUMMARY_DIR, exist_ok=True)
os.makedirs(CONFIGS_DIR, exist_ok=True)

print(f"📁 配置路径:")
print(f"  - 数据目录: {DATA_DIR}")
print(f"  - 配置目录: {CONFIGS_DIR}")
print(f"  - 输出目录: {OUTPUT_ROOT}")
print(f"  - 选择样本: {SELECTED_JSON}")


# ==== 2. 工具函数 ====
def ensure_dir(path):
    """确保目录存在"""
    os.makedirs(path, exist_ok=True)


def save_point_cloud(points, filename):
    """保存点云数据为PLY格式"""
    if points is None or len(points) == 0:
        print(f"⚠️ Warning: No points to save for {filename}")
        return False
    
    if not O3D_AVAILABLE:
        print(f"⚠️ Warning: Cannot save {filename} - open3d not available")
        return False
        
    try:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        o3d.io.write_point_cloud(filename, pcd)
        print(f"✓ Saved point cloud: {os.path.basename(filename)} ({len(points):,} points)")
        return True
    except Exception as e:
        print(f"❌ Error saving point cloud {filename}: {e}")
        return False


def save_stats(stats, filename):
    """保存统计信息为JSON格式"""
    try:
        # 确保所有numpy数组都转换为Python原生类型
        cleaned_stats = {}
        for key, value in stats.items():
            if isinstance(value, np.ndarray):
                cleaned_stats[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                cleaned_stats[key] = value.item()
            else:
                cleaned_stats[key] = value
        
        # 添加处理时间戳
        cleaned_stats['processed_at'] = datetime.now().isoformat()
                
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(cleaned_stats, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved stats: {os.path.basename(filename)}")
        return True
    except Exception as e:
        print(f"❌ Error saving stats {filename}: {e}")
        return False


def save_annotations(annotations, filename):
    """保存标注信息为JSON格式"""
    try:
        # 清理annotations中的numpy数组
        cleaned_annotations = []
        for ann in annotations:
            cleaned_ann = {}
            for key, value in ann.items():
                if isinstance(value, np.ndarray):
                    cleaned_ann[key] = value.tolist()
                elif isinstance(value, (np.integer, np.floating)):
                    cleaned_ann[key] = value.item()
                else:
                    cleaned_ann[key] = value
            cleaned_annotations.append(cleaned_ann)
            
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(cleaned_annotations, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved annotations: {os.path.basename(filename)} ({len(annotations)} boxes)")
        return True
    except Exception as e:
        print(f"❌ Error saving annotations {filename}: {e}")
        return False


def save_figure(fig, filename, dpi=150):
    """保存matplotlib图片"""
    try:
        fig.savefig(filename, dpi=dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close(fig)  # 关闭图形以释放内存
        print(f"✓ Saved visualization: {os.path.basename(filename)}")
        return True
    except Exception as e:
        print(f"❌ Error saving figure {filename}: {e}")
        return False


def log_error(error_msg, log_path):
    """记录错误日志"""
    try:
        ensure_dir(os.path.dirname(log_path))
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*50}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Error: {error_msg}\n")
            f.write(f"{'='*50}\n")
    except Exception as e:
        print(f"❌ Failed to write error log: {e}")


# ==== 3. 主批量处理函数 ====
def process_sample(visualizer, scene_name, sample_token, output_root, max_distance=50.0):
    """
    处理单个样本
    
    Args:
        visualizer: TruckScenesVisualizer实例
        scene_name: 场景名称
        sample_token: 样本token
        output_root: 输出根目录
        max_distance: 最大显示距离
    
    Returns:
        dict: 统计信息，失败时返回None
    """
    try:
        print(f"\n🔄 Processing: {scene_name} | {sample_token[:16]}...")
        
        # 检查 visualize_sample 方法是否支持 return_data 参数
        import inspect
        sig = inspect.signature(visualizer.visualize_sample)
        
        if 'return_data' in sig.parameters:
            # 新版本：支持 return_data 参数
            result = visualizer.visualize_sample(
                sample_token, 
                max_distance=max_distance, 
                return_data=True
            )
        else:
            # 旧版本：手动构建结果数据
            print("⚠️ 使用兼容模式处理样本...")
            
            # 先获取样本信息
            sample = visualizer.ts.get('sample', sample_token)
            
            # 收集传感器数据
            available_sensors = list(sample['data'].keys())
            lidar_sensors = [s for s in available_sensors if 'LIDAR' in s.upper()]
            radar_sensors = [s for s in available_sensors if 'RADAR' in s.upper()]
            
            # 收集点云数据
            all_lidar_points = []
            all_radar_points = []
            
            # 处理LiDAR数据
            for lidar_sensor in lidar_sensors:
                lidar_token = sample['data'][lidar_sensor]
                lidar_points, _ = visualizer._get_sensor_data(lidar_token)
                if lidar_points is not None:
                    all_lidar_points.append(lidar_points)
            
            # 处理Radar数据  
            for radar_sensor in radar_sensors:
                radar_token = sample['data'][radar_sensor]
                radar_points, _ = visualizer._get_sensor_data(radar_token)
                if radar_points is not None:
                    all_radar_points.append(radar_points)
            
            # 合并点云数据
            merged_lidar = np.vstack(all_lidar_points) if all_lidar_points else np.empty((0, 3))
            merged_radar = np.vstack(all_radar_points) if all_radar_points else np.empty((0, 3))
            
            # 获取标注
            annotations = visualizer._get_sample_annotations(sample_token)
            
            # 创建可视化（但不显示）
            plt.ioff()  # 关闭交互模式
            fig, lidar_count, radar_count, box_count = visualizer._create_visualization(
                merged_lidar, merged_radar, annotations, 
                sample_token, max_distance, len(lidar_sensors), len(radar_sensors)
            )
            plt.ion()  # 重新开启交互模式
            
            # 手动构建结果
            result = {
                "fig": fig,
                "merged_lidar": merged_lidar,
                "merged_radar": merged_radar,
                "annotations": annotations,
                "stats": {
                    "sample_token": sample_token,
                    "n_lidar": len(lidar_sensors),
                    "n_radar": len(radar_sensors),
                    "lidar_points": int(len(merged_lidar)),
                    "radar_points": int(len(merged_radar)),
                    "box_count": int(len(annotations)) if annotations else 0,
                    "max_distance": max_distance
                },
                "log": []
            }
        
        # 创建输出目录
        out_dir = os.path.join(output_root, scene_name, sample_token)
        ensure_dir(out_dir)
        
        # 保存可视化图片
        if result.get("fig"):
            img_path = os.path.join(out_dir, "visualization.png")
            save_figure(result["fig"], img_path)
        
        # 保存点云数据
        if result.get("merged_lidar") is not None:
            lidar_path = os.path.join(out_dir, "merged_lidar.ply")
            if save_point_cloud(result["merged_lidar"], lidar_path):
                result["stats"]["lidar_path"] = lidar_path
        
        if result.get("merged_radar") is not None:
            radar_path = os.path.join(out_dir, "merged_radar.ply")
            if save_point_cloud(result["merged_radar"], radar_path):
                result["stats"]["radar_path"] = radar_path
        
        # 保存统计信息
        stats_path = os.path.join(out_dir, "stats.json")
        save_stats(result["stats"], stats_path)
        
        # 保存标注信息
        if result.get("annotations"):
            ann_path = os.path.join(out_dir, "annotations.json")
            save_annotations(result["annotations"], ann_path)
        
        # 保存处理日志
        if result.get("log"):
            log_path = os.path.join(out_dir, "process_log.txt")
            with open(log_path, "w", encoding="utf-8") as f:
                for log_entry in result["log"]:
                    f.write(f"{log_entry}\n")
        
        print(f"✅ Successfully processed: {scene_name} | {sample_token[:16]}")
        return result["stats"]
        
    except Exception as e:
        error_msg = f"Error processing {scene_name} | {sample_token}: {str(e)}"
        print(f"❌ {error_msg}")
        
        # 记录详细错误信息
        error_log_path = os.path.join(output_root, scene_name, sample_token, "error_log.txt")
        ensure_dir(os.path.dirname(error_log_path))
        with open(error_log_path, "w", encoding="utf-8") as f:
            f.write(f"Error processing sample: {sample_token}\n")
            f.write(f"Scene: {scene_name}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Error message: {str(e)}\n\n")
            f.write("Full traceback:\n")
            f.write(traceback.format_exc())
        
        return None


def create_sample_config():
    """创建示例配置文件"""
    sample_config = {
        "scene-0001": [
            "32d2bcf46e734dffb14fe2e0a823d059",
            "e3b2c1d4a5f6789012345678901234ab"
        ],
        "scene-0002": [
            "f4c3b2a1d5e6789012345678901234cd",
            "a1b2c3d4e5f6789012345678901234ef"
        ]
    }
    
    with open(SELECTED_JSON, "w", encoding="utf-8") as f:
        json.dump(sample_config, f, indent=2)
    print(f"✓ Created sample config: {SELECTED_JSON}")


def generate_summary_report(all_stats, summary_dir):
    """生成汇总报告"""
    if not all_stats:
        print("⚠️ No data available for summary report")
        return
    
    df = pd.DataFrame(all_stats)
    
    # 保存完整统计CSV
    csv_path = os.path.join(summary_dir, "all_stats.csv")
    df.to_csv(csv_path, index=False)
    print(f"✓ Saved complete stats: {csv_path}")
    
    # 生成汇总统计
    summary_stats = {
        "total_samples": len(df),
        "total_scenes": df['scene_name'].nunique() if 'scene_name' in df.columns else 0,
        "total_lidar_points": int(df['lidar_points'].sum()) if 'lidar_points' in df.columns else 0,
        "total_radar_points": int(df['radar_points'].sum()) if 'radar_points' in df.columns else 0,
        "total_boxes": int(df['box_count'].sum()) if 'box_count' in df.columns else 0,
        "avg_lidar_points": float(df['lidar_points'].mean()) if 'lidar_points' in df.columns else 0,
        "avg_radar_points": float(df['radar_points'].mean()) if 'radar_points' in df.columns else 0,
        "avg_boxes": float(df['box_count'].mean()) if 'box_count' in df.columns else 0,
        "processing_date": datetime.now().isoformat()
    }
    
    # 保存汇总统计
    summary_path = os.path.join(summary_dir, "summary_stats.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_stats, f, indent=2)
    print(f"✓ Saved summary stats: {summary_path}")
    
    # 按场景分组统计
    if 'scene_name' in df.columns:
        scene_stats = df.groupby('scene_name').agg({
            'lidar_points': ['count', 'sum', 'mean'] if 'lidar_points' in df.columns else ['count'],
            'radar_points': ['sum', 'mean'] if 'radar_points' in df.columns else [],
            'box_count': ['sum', 'mean'] if 'box_count' in df.columns else []
        }).round(2)
        
        scene_csv_path = os.path.join(summary_dir, "scene_summary.csv")
        scene_stats.to_csv(scene_csv_path)
        print(f"✓ Saved scene summary: {scene_csv_path}")
    
    print(f"\n📊 处理汇总:")
    print(f"  - 总样本数: {summary_stats['total_samples']}")
    print(f"  - 总场景数: {summary_stats['total_scenes']}")
    print(f"  - 总LiDAR点: {summary_stats['total_lidar_points']:,}")
    print(f"  - 总Radar点: {summary_stats['total_radar_points']:,}")
    print(f"  - 总标注框: {summary_stats['total_boxes']}")


def main():
    """主函数"""
    print("🚛 TruckScenes 批量输出管理系统")
    print("=" * 50)
    
    # ==== 4. 检查配置文件 ====
    if not os.path.exists(SELECTED_JSON):
        print(f"⚠️ 配置文件不存在: {SELECTED_JSON}")
        print("🔧 创建示例配置文件...")
        create_sample_config()
        print("📝 请编辑配置文件，添加要处理的场景和样本token")
        return
    
    # 加载配置
    try:
        with open(SELECTED_JSON, "r", encoding="utf-8") as f:
            scene_samples = json.load(f)
        print(f"✓ 加载配置文件: {len(scene_samples)} 个场景")
    except Exception as e:
        print(f"❌ 无法加载配置文件: {e}")
        return
    
    # ==== 5. 初始化可视化工具 ====
    try:
        print("🔧 初始化 TruckScenes 可视化工具...")
        visualizer = TruckScenesVisualizer(DATA_DIR, version="v1.0-mini")
        print("✓ 可视化工具初始化成功")
    except Exception as e:
        print(f"❌ 无法初始化可视化工具: {e}")
        return
    
    # ==== 6. 批量处理 ====
    all_stats = []
    total_samples = sum(len(samples) for samples in scene_samples.values())
    processed_count = 0
    
    print(f"\n🚀 开始批量处理 {total_samples} 个样本...")
    
    for scene_name, sample_list in scene_samples.items():
        print(f"\n📂 处理场景: {scene_name} ({len(sample_list)} 个样本)")
        
        for sample_token in sample_list:
            processed_count += 1
            print(f"[{processed_count}/{total_samples}] ", end="")
            
            stats = process_sample(
                visualizer, scene_name, sample_token, OUTPUT_ROOT
            )
            
            if stats is not None:
                stats["scene_name"] = scene_name
                stats["sample_token"] = sample_token
                all_stats.append(stats)
    
    # ==== 7. 生成汇总报告 ====
    print(f"\n📈 生成汇总报告...")
    generate_summary_report(all_stats, SUMMARY_DIR)
    
    # ==== 8. 完成信息 ====
    success_count = len(all_stats)
    failure_count = total_samples - success_count
    
    print(f"\n🎉 批量处理完毕!")
    print(f"  ✅ 成功: {success_count}/{total_samples}")
    if failure_count > 0:
        print(f"  ❌ 失败: {failure_count}/{total_samples}")
    print(f"  📁 输出目录: {OUTPUT_ROOT}")
    print(f"  📊 汇总报告: {SUMMARY_DIR}")
    
    if success_count == 0:
        print("\n⚠️ 没有成功处理的样本，请检查:")
        print("  1. 数据路径是否正确")
        print("  2. 样本token是否有效")
        print("  3. 依赖包是否正确安装")


if __name__ == "__main__":
    main()