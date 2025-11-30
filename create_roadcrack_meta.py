import os
import json
from glob import glob
import sys

# --- 配置 ---
DATASET_NAME = 'RoadCrack_Crop'
CLASS_NAME = 'road_texture'


# --- 结束配置 ---

def main():
    # 数据集根目录
    data_dir = os.path.join('data', DATASET_NAME)

    if not os.path.isdir(data_dir):
        print(f"错误: 目录 '{data_dir}' 未找到。请确保在 UniVAD 根目录下运行此脚本。")
        sys.exit(1)

    # 初始化 meta_data 结构
    meta_data = {
        "train": {CLASS_NAME: []},
        "test": {CLASS_NAME: []}
    }

    class_dir = os.path.join(data_dir, CLASS_NAME)
    if not os.path.isdir(class_dir):
        print(f"错误: 类别目录 '{class_dir}' 未找到。")
        sys.exit(1)

    print(f"正在为 '{CLASS_NAME}' 生成 'meta.json'...")

    # =========================================================
    # 1. 处理训练集 (train/good) - 主要是 .png (裁剪图)
    # =========================================================
    train_dir = os.path.join(class_dir, 'train', 'good')
    # 同时查找 png 和 jpg 以防万一
    train_imgs = sorted(glob(os.path.join(train_dir, '*.png')) +
                        glob(os.path.join(train_dir, '*.jpg')))

    if not train_imgs:
        print(f"  警告: 在 {train_dir} 中未找到训练图像 (png/jpg)。")

    for img_path in train_imgs:
        # 使用 relpath 计算相对路径，避免字符串替换错误
        rel_path = os.path.relpath(img_path, data_dir)
        # 统一转为 Linux 风格斜杠
        rel_path = rel_path.replace(os.sep, '/')

        info = {
            "img_path": rel_path,
            "mask_path": "",
            "cls_name": CLASS_NAME,
            "specie_name": "good",
            "anomaly": 0
        }
        meta_data["train"][CLASS_NAME].append(info)

    # =========================================================
    # 2. 处理测试集 (test/good 和 test/crack)
    # =========================================================
    # 同时查找 png 和 jpg
    test_search_path = os.path.join(class_dir, 'test', '*', '*')
    all_test_files = glob(test_search_path)
    # 过滤出图片文件
    test_img_paths = sorted([f for f in all_test_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    test_good_count = 0
    test_crack_count = 0

    for img_path in test_img_paths:
        rel_path = os.path.relpath(img_path, data_dir)
        rel_path = rel_path.replace(os.sep, '/')

        # 解析类别: road_texture/test/crack/xxx.jpg -> crack
        parts = rel_path.split('/')
        anomaly_type = parts[-2]
        img_basename = os.path.splitext(parts[-1])[0]

        info = {
            "img_path": rel_path,
            "cls_name": CLASS_NAME,
            "specie_name": anomaly_type
        }

        if anomaly_type == 'good':
            info["mask_path"] = ""
            info["anomaly"] = 0
            test_good_count += 1
        else:
            test_crack_count += 1
            # 寻找对应的掩码文件 (通常也是 png)
            mask_path_raw = os.path.join(class_dir, 'ground_truth', anomaly_type, img_basename + '.png')

            # 如果找不到 png 掩码，尝试找 jpg 掩码 (虽然掩码一般是png)
            if not os.path.exists(mask_path_raw):
                mask_path_raw = os.path.join(class_dir, 'ground_truth', anomaly_type, img_basename + '.jpg')

            if os.path.exists(mask_path_raw):
                mask_rel = os.path.relpath(mask_path_raw, data_dir)
                info["mask_path"] = mask_rel.replace(os.sep, '/')
                info["anomaly"] = 1
            else:
                # 只有当确实找不到时才警告
                print(f"警告: 找不到掩码文件 -> {mask_path_raw}")
                info["mask_path"] = ""
                info["anomaly"] = 1

        meta_data["test"][CLASS_NAME].append(info)

    # =========================================================
    # 3. 保存
    # =========================================================
    output_path = os.path.join(data_dir, 'meta.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(meta_data, f, indent=4, ensure_ascii=False)

    print(f"\n成功生成: {output_path}")
    print(f"训练集数量: {len(meta_data['train'][CLASS_NAME])}")
    print(f"测试集数量: {len(meta_data['test'][CLASS_NAME])} (Good: {test_good_count}, Crack: {test_crack_count})")


if __name__ == "__main__":
    main()