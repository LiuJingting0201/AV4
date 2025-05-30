import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from truckscenes import TruckScenes
from truckscenes.utils.visualization_utils import view_points

# ——— 1. 要合并的所有 Radar 通道 ———
RADAR_CHANNELS = [
    "RADAR_LEFT_BACK","RADAR_LEFT_FRONT","RADAR_LEFT_SIDE",
    "RADAR_RIGHT_BACK","RADAR_RIGHT_FRONT","RADAR_RIGHT_SIDE"
]

def make_transform(translation, quat_wxyz):
    """
    TruckScenes 中 pose['rotation'] & cs['rotation'] 都是 [w, x, y, z] 格式，
    而 scipy.from_quat 要 [x, y, z, w]，所以我们要重新排列：
    """
    # 1. 重新排列四元数
    w, x, y, z = quat_wxyz
    q = [x, y, z, w]
    # 2. 生成旋转矩阵
    from scipy.spatial.transform import Rotation as R
    rotm = R.from_quat(q).as_matrix()
    # 3. 拼 4x4 变换
    T = np.eye(4)
    T[:3,:3] = rotm
    T[:3, 3] = translation
    return T


def load_and_merge_points(ts, sample_token, channels):
    """读取并合并所有指定通道的点云到一个 (N,3) 数组"""
    pts_list = []
    sample = ts.get('sample', sample_token)
    for ch in channels:
        if ch not in sample['data']:
            continue
        sd_token = sample['data'][ch]
        fp = ts.get_sample_data_path(sd_token)
        if fp.endswith('.bin'):
            # LiDAR.bin 二进制
            arr = np.fromfile(fp, dtype=np.float32).reshape(-1,5)
            xyz = arr[:, :3]
        else:
            # Radar .pcd 或其他
            pcd = o3d.io.read_point_cloud(fp)
            xyz = np.asarray(pcd.points)
        pts_list.append(xyz)
    if not pts_list:
        return np.zeros((0,3))
    return np.vstack(pts_list)

def project_to_image(ts, sample_token, cam_chan, points):
    """
    手动把全局 (x,y,z) 点投影到相机像素平面上
    返回：
      uv: (2,N) 像素坐标
      depth: (N,) 深度值
    """
    # 获取相机 sample_data
    sd_token = ts.get('sample', sample_token)['data'][cam_chan]
    sd = ts.get('sample_data', sd_token)
    cs = ts.get('calibrated_sensor', sd['calibrated_sensor_token'])
    pose = ts.get('ego_pose', sd['ego_pose_token'])

    # 构建变换矩阵：global->ego->sensor
    T_ego   = make_transform(pose['translation'], pose['rotation'])
    T_sensor= make_transform(cs['translation'], cs['rotation'])
    # global->sensor: 先 global->ego，再 ego->sensor
    T = np.linalg.inv(T_sensor) @ np.linalg.inv(T_ego)

    # 齐次坐标投影
    pts_h = np.hstack([points, np.ones((len(points),1))]).T  # (4,N)
    cam_pts = T @ pts_h                                    # (4,N)
    pts3 = cam_pts[:3, :]                                  # (3,N)

    # 用官方的 view_points 做内参投影
    uv = view_points(pts3, np.array(cs['camera_intrinsic']), normalize=True)  # (2,N)
    depth = pts3[2, :]  # Z 轴深度
    return uv, depth

def overlay_radar_all_at_once(sample_index=0):
    # 初始化 DevKit
    ts = TruckScenes(
        version='v1.0-mini',
        dataroot=r"D:/MyProjects/AV4/data/man-truckscenes",
        verbose=False
    )
    sample_token = ts.sample[sample_index]['token']
    cam_chan = 'CAMERA_LEFT_FRONT'

    # 读取相机图像
    sd_token = ts.get('sample', sample_token)['data'][cam_chan]
    img_fp = ts.get_sample_data_path(sd_token)
    img = np.array(Image.open(img_fp))
    H, W = img.shape[:2]

    # 合并所有 Radar 通道点云
    pts = load_and_merge_points(ts, sample_token, RADAR_CHANNELS)

    # 投影到图像平面
    uv, depth = project_to_image(ts, sample_token, cam_chan, pts)

    # 过滤在图像范围内的点
    mask = (
        (depth > 0) &
        (uv[0] >= 0) & (uv[0] < W) &
        (uv[1] >= 0) & (uv[1] < H)
    )
    print("Total points:", len(pts), "  In image:", mask.sum())

    # 绘制
    fig, ax = plt.subplots(figsize=(12,6))
    ax.imshow(img)
    # 第一步：红点测试
    ax.scatter(uv[0,mask], uv[1,mask], c='r', s=12, alpha=0.7)
    # 第二步：按深度上色
    sc = ax.scatter(
        uv[0,mask], uv[1,mask],
        c=depth[mask],
        cmap='turbo',
        s=8,
        alpha=0.8
    )
    plt.colorbar(sc, ax=ax, label='Depth (m)')
    ax.axis('off')

    # 保存与展示
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output', 'overlays'))
    os.makedirs(out_dir, exist_ok=True)
    out_fp = os.path.join(out_dir, f"{sample_token}_radar_all_once.png")
    plt.savefig(out_fp, bbox_inches='tight', pad_inches=0)
    plt.show()
    print("✅ Saved all-at-once radar overlay:", out_fp)

if __name__ == "__main__":
    overlay_radar_all_at_once(sample_index=0)
