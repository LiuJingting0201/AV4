import numpy as np
import matplotlib.pyplot as plt
from truckscenes import TruckScenes
from scipy.spatial.transform import Rotation as R

# --- 配置 ---
sample_token = "32d2bcf46e734dffb14fe2e0a823d059"
ts = TruckScenes('v1.0-mini', dataroot='D:/MyProjects/AV4/data/man-truckscenes', verbose=False)

# --- 获取所有LiDAR和Radar点 ---
def get_points(ts, sample_token, sensor_type='LIDAR'):
    pts = []
    for sd in ts.sample_data:
        if sd["sample_token"] == sample_token and sensor_type in sd["channel"]:
            path = ts.get_sample_data_path(sd["token"])
            pts.append(np.load(path, allow_pickle=True))   # 加上 allow_pickle=True
    if len(pts):
        return np.vstack(pts)
    return np.zeros((0, 3))


lidar_pts = get_points(ts, sample_token, 'LIDAR')
radar_pts = get_points(ts, sample_token, 'RADAR')

# --- 读取所有3D框 ---
sample = [s for s in ts.sample if s["token"] == sample_token][0]
anns = [ts.get('sample_annotation', t) for t in sample['anns']]
edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]

def box_corners(ann):
    dx, dy, dz = ann['size']
    hx, hy, hz = dx/2, dy/2, dz/2
    corners = np.array([
        [ hx,  hy,  hz], [-hx,  hy,  hz], [-hx, -hy,  hz], [ hx, -hy,  hz],
        [ hx,  hy, -hz], [-hx,  hy, -hz], [-hx, -hy, -hz], [ hx, -hy, -hz]
    ])
    w,x,y,z = ann['rotation']
    rotm = R.from_quat([x,y,z,w]).as_matrix()
    center = np.array(ann['translation'])
    return (rotm @ corners.T).T + center

# --- 可视化 ---
fig = plt.figure(figsize=(12,10))
ax = fig.add_subplot(111, projection='3d')
ax.set_title(f"Fusion of LiDAR + Radar + 3D Boxes\nSample: {sample_token[:8]}")

if lidar_pts.shape[0] > 0:
    ax.scatter(lidar_pts[:,0], lidar_pts[:,1], lidar_pts[:,2], s=0.03, c='dodgerblue', alpha=0.3, label="LiDAR")
if radar_pts.shape[0] > 0:
    ax.scatter(radar_pts[:,0], radar_pts[:,1], radar_pts[:,2], s=2, c='red', alpha=0.7, label="Radar")

for ann in anns:
    cs = box_corners(ann)
    for i,j in edges:
        ax.plot([cs[i,0],cs[j,0]], [cs[i,1],cs[j,1]], [cs[i,2],cs[j,2]], c='orange', linewidth=1)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.tight_layout()
plt.show()
