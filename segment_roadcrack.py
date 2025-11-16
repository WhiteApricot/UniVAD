from models.component_segmentaion import grounding_segmentation
import os
import glob
import yaml


# 1. 定义一个辅助函数 (从您的原始脚本中复制)
def read_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    return config


# 2. --- 这是我们要运行的唯一代码块 ---

# 2a. 定义您的数据集信息
# (这些必须与您的 data/create_roadcrack_meta.py 脚本中的设置匹配)
mask_path = "./masks/RoadCrack_Crop"
dataset_name = "RoadCrack_Crop"
categories = ["road_texture"]  # 这是您在 prepare...py 中设置的 "CLASS_NAME"

# 2b. 借用一个相似的纹理配置文件 (如我们之前讨论的 carpet)
# 确保这个 .yaml 文件存在于 'configs/class_histogram/' 中
config_file_to_use = "carpet.yaml"
config = read_config(f"./configs/class_histogram/{config_file_to_use}")

print(f"--- 正在为自定义数据集 '{dataset_name}' 运行组件分割 ---")

for category in categories:
    # 2c. 创建输出目录
    output_dir = f"{mask_path}/{category}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"  类别: {category}")
    print(f"  输出掩码到: {output_dir}")

    # 2d. 处理 test 图像 (来自 'test/crack' 和 'test/good')
    # *** 关键: 确保这里的 .jpg 扩展名与您的文件名匹配 ***
    test_image_paths = sorted(glob.glob(f"./data/{dataset_name}/{category}/test/*/*.jpg"))
    print(f"  找到 {len(test_image_paths)} 张 'test' 图像...")
    if test_image_paths:
        grounding_segmentation(
            test_image_paths, output_dir, config["grounding_config"]
        )

    # 2e. 处理 train 图像 (来自 'train/good')
    # *** 关键: 确保这里的 .jpg 扩展名与您的文件名匹配 ***
    train_image_paths = sorted(glob.glob(f"./data/{dataset_name}/{category}/train/*/*.jpg"))
    print(f"  找到 {len(train_image_paths)} 张 'train' 图像...")
    if train_image_paths:
        grounding_segmentation(
            train_image_paths, output_dir, config["grounding_config"]
        )

print(f"--- '{dataset_name}' 处理完成 ---")