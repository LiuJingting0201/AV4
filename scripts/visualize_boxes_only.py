import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

def draw_3d_box(ax, center, size, rot, color='orange'):
    # size: [w, l, h]    rot: quaternion [w,x,y,z]
    w, l, h = size
    x_c, y_c, z_c = center
    # 8 corners (中心为原点)
    corners = np.array([
        [ w/2,  l/2,  h/2],
        [-w/2,  l/2,  h/2],
        [-w/2, -l/2,  h/2],
        [ w/2, -l/2,  h/2],
        [ w/2,  l/2, -h/2],
        [-w/2,  l/2, -h/2],
        [-w/2, -l/2, -h/2],
        [ w/2, -l/2, -h/2]
    ])
    rotm = R.from_quat([rot[1], rot[2], rot[3], rot[0]]).as_matrix()
    corners = corners @ rotm.T + np.array([x_c, y_c, z_c])
    # 画线
    edges = [
        (0,1),(1,2),(2,3),(3,0),
        (4,5),(5,6),(6,7),(7,4),
        (0,4),(1,5),(2,6),(3,7)
    ]
    for i,j in edges:
        ax.plot([corners[i,0], corners[j,0]],
                [corners[i,1], corners[j,1]],
                [corners[i,2], corners[j,2]], c=color, linewidth=2)

# =========== 加载 annotation json =============
with open("../data/man-truckscenes/v1.0-mini/sample_annotation.json", "r") as f:
    all_boxes = json.load(f)  # list of dict

# 过滤当前sample的boxes
target_sample_token = "32d2bcf46e734dffb14fe2e0a823d059"
boxes = [b for b in all_boxes if b["sample_token"] == target_sample_token]

# =========== 可视化 ===========
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

for ann in boxes:
    center = ann['translation']
    size = ann['size']
    rot = ann['rotation']
    draw_3d_box(ax, center, size, rot, color='orange')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.tight_layout()
plt.show()
