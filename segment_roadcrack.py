from models.component_segmentaion import grounding_segmentation
import os
import glob
import yaml


# 1. 定义一个辅助函数 (保持不变)
def read_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    return config


# 2. --- 这是我们要运行的唯一代码块 ---

# 2a. 定义您的数据集信息 (保持不变)
mask_path = "./masks/RoadCrack_Crop"
dataset_name = "RoadCrack_Crop"
categories = ["road_texture"]

# 2b. 借用一个相似的纹理配置文件 (保持不变)
config_file_to_use = "carpet.yaml"
config = read_config(f"./configs/class_histogram/{config_file_to_use}")

print(f"--- 正在为自定义数据集 '{dataset_name}' 运行组件分割 ---")

for category in categories:
    # 2c. 创建输出目录 (保持不变)
    output_dir = f"{mask_path}/{category}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"  类别: {category}")
    print(f"  输出掩码到: {output_dir}")

    # -----------------------------------------------------------------
    # --- 修复 1 (START): 正确查找 test 目录中的所有文件 ---
    # -----------------------------------------------------------------
    # test/crack 是 .jpg, test/good 是 .png
    # 我们必须同时查找这两种
    test_image_paths_jpg = sorted(glob.glob(f"./data/{dataset_name}/{category}/test/*/*.jpg"))
    test_image_paths_png = sorted(glob.glob(f"./data/{dataset_name}/{category}/test/*/*.png"))
    test_image_paths = test_image_paths_jpg + test_image_paths_png  # 合并两个列表
    # -----------------------------------------------------------------
    # --- 修复 1 (END) ---
    # -----------------------------------------------------------------

    print(f"  找到 {len(test_image_paths)} 张 'test' 图像 (来自 good 和 crack)...")
    if test_image_paths:
        grounding_segmentation(
            test_image_paths, output_dir, config["grounding_config"]
        )

    # -----------------------------------------------------------------
    # --- 修复 2 (START): 正确查找 train 目录中的文件 ---
    # -----------------------------------------------------------------
    # train/good 目录中现在只包含 .png 文件
    train_image_paths = sorted(glob.glob(f"./data/{dataset_name}/{category}/train/*/*.png"))
    # -----------------------------------------------------------------
    # --- 修复 2 (END) ---
    # -----------------------------------------------------------------

    print(f"  找到 {len(train_image_paths)} 张 'train' 图像...")
    if train_image_paths:
        grounding_segmentation(
            train_image_paths, output_dir, config["grounding_config"]
        )

print(f"--- '{dataset_name}' 处理完成 ---")