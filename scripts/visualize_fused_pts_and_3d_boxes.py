import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

def plot_3d_bbox(ax, corners, color='orange', linewidth=2):
    # corners: (8,3) array, 按标准顺序，详见下方说明
    edges = [
        [0,1],[1,2],[2,3],[3,0],  # 上面
        [4,5],[5,6],[6,7],[7,4],  # 下面
        [0,4],[1,5],[2,6],[3,7]   # 侧面
    ]
    for e in edges:
        ax.plot(*zip(corners[e[0]], corners[e[1]]), c=color, linewidth=linewidth)

# -------- 读取 LiDAR 和 Radar 融合点云 --------
lidar_path = 'output/merged_lidar/32d2bcf46e734dffb14fe2e0a823d059_merged_lidar.ply'
radar_path = 'output/merged_radar/32d2bcf46e734dffb14fe2e0a823d059_merged_radar.ply'

lidar_pcd = o3d.io.read_point_cloud(lidar_path)
radar_pcd = o3d.io.read_point_cloud(radar_path)

lidar_pts = np.asarray(lidar_pcd.points)
radar_pts = np.asarray(radar_pcd.points)

# ----------- 读取 annotation boxes -----------
# 假设已获得 boxes_list，每个元素是 shape=(8,3) 的 numpy array
# 你需要根据 TruckScenes API 读取 annotation，或者用你前面能打印出3D框corners的脚本（见下方注释）

boxes_list = []
# 例如：
# boxes_list.append(corners1)  # corners1.shape = (8,3)
# boxes_list.append(corners2)  # ...

# 示例：如果你已经能打印3D box corners，建议先人工粘贴2-3个corners测试效果（报告只要视觉展示即可）

# ----------- 可视化 3D -------------
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
ax.set_title('Fusion of LiDAR + Radar + 3D Boxes\nSample: 32d2bcf4')

# LiDAR点
if len(lidar_pts) > 0:
    ax.scatter(lidar_pts[:,0], lidar_pts[:,1], lidar_pts[:,2], s=0.05, c='deepskyblue', label='LiDAR', alpha=0.7)
# Radar点
if len(radar_pts) > 0:
    ax.scatter(radar_pts[:,0], radar_pts[:,1], radar_pts[:,2], s=0.8, c='orange', label='Radar', alpha=0.5)

# 画 annotation box
for corners in boxes_list:
    plot_3d_bbox(ax, corners, color='orangered', linewidth=2)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.tight_layout()
plt.show()
