from truckscenes import TruckScenes

def main():
    # 初始化
    dataroot = r"D:/MyProjects/AV4/data/man-truckscenes"
    truckscenes = TruckScenes(version='v1.0-mini', dataroot=dataroot)

    sample_token = truckscenes.sample[0]['token']
    sample = truckscenes.get('sample', sample_token)

    # 其他代码...

    ann_tokens = sample['anns']
    anns = [truckscenes.get('sample_annotation', t) for t in ann_tokens]
    print(f"Total annotations in sample: {len(anns)}")
    for ann in anns[:5]:
        category_name = ann.get('category_name')
        if not category_name:
            category_token = ann.get('category_token')
            if category_token:
                category_name = truckscenes.get('category', category_token)['name']
            else:
                category_name = 'Unknown'
        sensor_modality = ann.get('sensor_modality', 'Unknown')
        print(f"Category: {category_name}, Detected by: {sensor_modality}")

if __name__ == "__main__":
    main()
