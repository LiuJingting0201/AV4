from truckscenes import TruckScenes
import matplotlib.pyplot as plt
def main():
    # 1. 初始化 TruckScenes 数据集路径
    dataroot = r"D:/MyProjects/AV4/data/man-truckscenes"  # 修改为你自己的路径
    truckscenes = TruckScenes(version='v1.0-mini', dataroot=dataroot)

    # 2. 获取第一个样本 token
    sample_token = truckscenes.sample[0]['token']

    # 3. 读取 sample 详细信息
    sample = truckscenes.get('sample', sample_token)
    print(f"Sample timestamp: {sample['timestamp']}")

    # 4. 获取各传感器 sample_data token
    cam_token = sample['data']['CAMERA_LEFT_FRONT']
    lidar_token = sample['data']['LIDAR_TOP_FRONT']
    radar_token = sample['data']['RADAR_LEFT_FRONT']

    # 5. 可视化各传感器数据
    print("Rendering Camera image with annotations...")
    truckscenes.render_sample_data(cam_token, with_anns=True)

    print("Rendering LiDAR point cloud...")
    truckscenes.render_sample_data(lidar_token)

    print("Rendering Radar point cloud...")
    truckscenes.render_sample_data(radar_token)

if __name__ == "__main__":
    main()
plt.show()