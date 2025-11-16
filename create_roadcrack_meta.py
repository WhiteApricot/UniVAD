import os
import json
from glob import glob
import sys

# --- 配置 ---
DATASET_NAME = 'RoadCrack_Crop'
CLASS_NAME = 'road_texture'


# --- 结束配置 ---

def main():
    # 'data_dir' 是数据集的根目录, e.g., 'data/RoadCrack_Crop'
    data_dir = os.path.join('data', DATASET_NAME)

    if not os.path.isdir(data_dir):
        print(f"错误: 目录 '{data_dir}' 未找到。")
        print("请确保您已经成功运行了 'prepare_crack_crop.py' 脚本。")
        sys.exit(1)

    # ---
    # --- 最终修复：使用 'mvtec_loco_solver.py' 要求的"字典列表"结构 ---
    # ---
    meta_data = {
        "train": {CLASS_NAME: []},  # 变成一个列表
        "test": {CLASS_NAME: []}  # 变成一个列表
    }

    # 'class_dir' 是类别子目录, e.g., 'data/RoadCrack_Crop/road_texture'
    class_dir = os.path.join(data_dir, CLASS_NAME)

    if not os.path.isdir(class_dir):
        print(f"错误: 类别目录 '{class_dir}' 未找到。")
        sys.exit(1)

    print(f"正在为 '{CLASS_NAME}' (在 {DATASET_NAME} 中) 生成 'meta.json'...")

    # 1. 查找训练图像 (train/good)
    train_imgs = glob(os.path.join(class_dir, 'train', 'good', '*.jpg'))
    if not train_imgs:
        print(f"  警告: 在 {os.path.join(class_dir, 'train', 'good')} 中未找到训练图像。")

    for img_path in sorted(train_imgs):
        # 路径修复：mvtec.py 会自动添加 self.root。我们只保存相对路径。
        # e.g., 'road_texture/train/good/001_crop.jpg'
        img_path_relative = img_path.replace(data_dir + os.sep, '').replace(os.sep, '/')

        info = {
            "img_path": img_path_relative,
            "mask_path": "",  # 训练集没有掩码
            "cls_name": CLASS_NAME,
            "specie_name": "good",
            "anomaly": 0  # 0 = 正常 (使用 'anomaly' 键)
        }
        meta_data["train"][CLASS_NAME].append(info)

    # 2. 查找测试图像 (test/good 和 test/crack)
    test_img_paths = glob(os.path.join(class_dir, 'test', '*', '*.jpg'))

    if not test_img_paths:
        print(f"  警告: 在 {os.path.join(class_dir, 'test')} 中未找到测试图像。")

    test_good_count = 0
    test_crack_count = 0

    for img_path in sorted(test_img_paths):
        # e.g., 'road_texture/test/crack/001.jpg'
        img_path_relative = img_path.replace(data_dir + os.sep, '').replace(os.sep, '/')

        parts = img_path_relative.split('/')
        img_basename = os.path.splitext(parts[-1])[0]
        anomaly_type = parts[-2]  # 'crack' or 'good'

        info = {
            "img_path": img_path_relative,
            "cls_name": CLASS_NAME,
            "specie_name": anomaly_type
        }

        if anomaly_type == 'good':
            info["mask_path"] = ""
            info["anomaly"] = 0  # 0 = 正常
            test_good_count += 1
        else:
            test_crack_count += 1
            mask_path_raw = os.path.join(class_dir, 'ground_truth', anomaly_type, img_basename + '.png')
            # e.g., 'road_texture/ground_truth/crack/001.png'
            if not os.path.exists(mask_path_raw):
                print(f"    警告: 未找到 {img_path} 对应的掩码: {mask_path_raw}")
                info["mask_path"] = ""  # 掩码为空
            else:
                info["mask_path"] = mask_path_raw.replace(data_dir + os.sep, '').replace(os.sep, '/')

            info["anomaly"] = 1  # 1 = 异常

        meta_data["test"][CLASS_NAME].append(info)

    # 4. 写入 meta.json 文件
    output_path = os.path.join(data_dir, 'meta.json')
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(meta_data, f, indent=4, ensure_ascii=False)

        print(f"\n成功创建 '{output_path}' (最终修复的 '字典列表' 结构)。")
        print(f"  - {len(meta_data['train'][CLASS_NAME])} 个训练样本")
        print(f"  - {len(meta_data['test'][CLASS_NAME])} 个测试样本 ({test_good_count} good, {test_crack_count} crack)")

    except IOError as e:
        print(f"写入 {output_path} 文件时出错: {e}")


if __name__ == "__main__":
    main()