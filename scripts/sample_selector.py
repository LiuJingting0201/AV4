import json
import os

# ==== 1. 路径配置 ====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "data", "man-truckscenes", "v1.0-mini"))
CONFIGS_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "configs"))
OUTPUT_PATH = os.path.join(CONFIGS_DIR, "selected_samples.json")

SCENE_JSON = os.path.join(DATA_DIR, "scene.json")
SAMPLE_JSON = os.path.join(DATA_DIR, "sample.json")

# ==== 2. 修正后的目标场景标签（基于实际数据中的标签） ====
TARGET_SCENES = {
    # 夜晚/暗光环境 + 高速公路
    "scene_night_highway":   ["lighting.dark", "area.highway"],
    
    # 雨天 + 高速公路
    "scene_rainy_highway":   ["weather.rain", "area.highway"],
    
    # 阴天 + 城市（模拟复杂城市环境）
    "scene_overcast_city":   ["weather.overcast", "area.city"],
    
    # 雪天 + 城市
    "scene_snow_city":       ["weather.snow", "area.city"],
    
    # 终端区域（物流区域）
    "scene_terminal_area":   ["area.terminal"],
    
    # 晴天城市（作为基准对比）
    "scene_clear_city":      ["weather.clear", "area.city"],
    
    # 桥梁结构 + 城市（复杂结构场景）
    "scene_bridge_city":     ["structure.bridge", "area.city"],
    
    # 施工路段（动态障碍场景）
    "scene_construction":    ["construction.roadworks"],
    
    # 黄昏时间（能见度挑战）
    "scene_twilight":        ["lighting.twilight"]
}

# ==== 3. 加载原始数据 ====
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_scene_tokens_by_tags(scenes, required_tags):
    """
    返回所有包含全部 required_tags 的 scene
    """
    result = []
    for s in scenes:
        tags = s["description"].split(";")
        # 检查是否所有必需的标签都存在
        if all(any(req_tag in tag for tag in tags) for req_tag in required_tags):
            result.append(s)
    return result

def get_sample_tokens_for_scene(scene_token, samples):
    """
    返回该 scene 下的所有 sample_token（顺序）
    """
    return [s["token"] for s in samples if s["scene_token"] == scene_token]

def analyze_available_tags(scenes):
    """
    分析所有可用的标签，帮助调试
    """
    all_tags = set()
    for scene in scenes:
        tags = scene["description"].split(";")
        all_tags.update(tags)
    
    # 按类别分组
    weather_tags = [tag for tag in all_tags if tag.startswith("weather.")]
    area_tags = [tag for tag in all_tags if tag.startswith("area.")]
    lighting_tags = [tag for tag in all_tags if tag.startswith("lighting.")]
    structure_tags = [tag for tag in all_tags if tag.startswith("structure.")]
    construction_tags = [tag for tag in all_tags if tag.startswith("construction.")]
    
    return {
        "weather": sorted(weather_tags),
        "area": sorted(area_tags),
        "lighting": sorted(lighting_tags),
        "structure": sorted(structure_tags),
        "construction": sorted(construction_tags),
        "all": sorted(all_tags)
    }

def main():
    scenes = load_json(SCENE_JSON)
    samples = load_json(SAMPLE_JSON)
    output = {}

    print("\n=== 数据集标签分析 ===")
    available_tags = analyze_available_tags(scenes)
    for category, tags in available_tags.items():
        if category != "all":
            print(f"{category.upper()}: {tags}")

    print(f"\n=== 所有场景及描述字段（共{len(scenes)}个场景） ===")
    for i, s in enumerate(scenes):
        print(f"{i:2d} | {s['name']}")
        print(f"    desc: {s['description']}")
        print(f"    samples: {s['nbr_samples']}\n")

    print("\n=== 自动标签筛选结果 ===")
    # ==== 4. 按标签自动筛选 ====
    for tag_name, tag_list in TARGET_SCENES.items():
        selected_scenes = get_scene_tokens_by_tags(scenes, tag_list)
        if not selected_scenes:
            print(f"⚠️  未找到符合 {tag_list} 的场景！")
            continue
        
        output[tag_name] = []
        for scene in selected_scenes:
            print(f"✅ 场景 {tag_name}")
            print(f"   名称: {scene['name']}")
            print(f"   描述: {scene['description']}")
            print(f"   帧数: {scene['nbr_samples']}帧")
            print()
            
            sample_tokens = get_sample_tokens_for_scene(scene["token"], samples)
            # 可选：只取前几帧或关键帧
            output[tag_name].extend(sample_tokens[:5])  # 每个场景取前5帧作为代表

    # ==== 5. 统计结果 ====
    print(f"\n=== 筛选统计 ===")
    total_samples = 0
    for scene_type, sample_list in output.items():
        print(f"{scene_type}: {len(sample_list)} 个样本")
        total_samples += len(sample_list)
    print(f"总共选中: {total_samples} 个样本")

    # ==== 6. 导出到 json 文件 ====
    os.makedirs(CONFIGS_DIR, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\n已生成典型场景样本列表: {OUTPUT_PATH}")


#############精简样本，只选前两个
    with open("configs/selected_samples.json", "r") as f:
        all_samples = json.load(f)

    reduced = {scene: tokens[:2] for scene, tokens in all_samples.items()}

    with open("configs/selected_samples.json", "w") as f:
        json.dump(reduced, f, indent=2)
    print("每类已精简为2个样本！")



    # ==== 7. 建议 ====
    print("\n🟩 建议：")
    print("1. 可以手动编辑 selected_samples.json 调整每类场景的样本数量")
    print("2. 如需要特定帧，可以根据时间戳或其他条件进一步筛选")
    print("3. 当前每个场景类型取前5帧，可根据需要调整")

if __name__ == "__main__":
    main()