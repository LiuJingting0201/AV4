# scripts/parse_sample.py

import os
import pandas as pd
from config import DATA_DIR, OUTPUT_DIR

def list_all_files(data_dir):
    """列出数据文件夹下的所有文件（简单示例）"""
    all_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            full_path = os.path.join(root, file)
            all_files.append(full_path)
    return all_files

def main():
    print(f"[INFO] 解析数据路径: {DATA_DIR}")

    # 示例：列出数据文件
    all_files = list_all_files(DATA_DIR)
    print(f"[INFO] 总共有 {len(all_files)} 个文件/样本。")

    # 示例：生成 DataFrame（这里只是示范，可以后续改为解析 TruckScenes 数据）
    df = pd.DataFrame({
        "FilePath": all_files
    })

    # 保存 CSV
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file = os.path.join(OUTPUT_DIR, "sample_files.csv")
    df.to_csv(output_file, index=False)
    print(f"[INFO] 文件列表已保存到 {output_file}")

if __name__ == "__main__":
    main()
