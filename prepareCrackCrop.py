import os
import cv2
import numpy as np
import shutil
from glob import glob
from tqdm import tqdm
import sys
import random  # 导入 random 库用于采样

# --- 配置 ---
SOURCE_ROOT = 'CRACK500FINAL'  # 您的原始数据存放的根目录
OUTPUT_DATASET_NAME = 'RoadCrack_Crop'  # 新数据集的名称
CLASS_NAME = 'road_texture'  # 您数据唯一的类别名

# 最终裁切出的“纯净矩形”的面积，必须大于原图总面积的这个比例
MIN_NORMAL_AREA_RATIO = 0.4

# --- !!! 新增：小规模测试的采样比例 !!! ---
# 1.0 = 使用 100% 的数据 (完整测试)
# 0.1 = 使用 10% 的数据 (快速测试)
SAMPLE_RATIO = 0.1
# --- 结束新增 ---

# 原始数据路径
SOURCE_IMAGE_DIR = os.path.join(SOURCE_ROOT, 'JPEGImages')
SOURCE_MASK_DIR = os.path.join(SOURCE_ROOT, 'SegmentationClass')

# UniVAD 格式的目标路径
BASE_OUTPUT_DIR = os.path.join('data', OUTPUT_DATASET_NAME)
CLASS_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, CLASS_NAME)

TRAIN_GOOD_DIR = os.path.join(CLASS_OUTPUT_DIR, 'train', 'good')
TEST_CRACK_DIR = os.path.join(CLASS_OUTPUT_DIR, 'test', 'crack')
GT_CRACK_DIR = os.path.join(CLASS_OUTPUT_DIR, 'ground_truth', 'crack')
TEST_GOOD_DIR = os.path.join(CLASS_OUTPUT_DIR, 'test', 'good')


def find_largest_pure_rectangle(image, mask):
    """
    在图像中找到最大的、100%纯净的矩形。
    掩码中：0=正常, 255=裂缝
    """

    # 1. 反转掩码，使 255=正常, 0=裂缝
    normal_mask = cv2.bitwise_not(mask)

    # 2. 找到所有“正常”区域的轮廓
    contours, _ = cv2.findContours(normal_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None  # 没有找到正常区域

    # 3. 找到面积最大的“正常”轮廓
    try:
        max_contour = max(contours, key=cv2.contourArea)
    except ValueError:
        return None  # 轮廓可能为空或无效

    # 4. 获取这个最大轮廓的“脏”边界框 (x, y, w, h)
    x, y, w, h = cv2.boundingRect(max_contour)

    # 5. 创建一个只包含最大正常轮廓的“查询掩码”
    query_mask = np.zeros_like(normal_mask)
    cv2.drawContours(query_mask, [max_contour], -1, 255, -1)  # 填充轮廓

    # 6. 开始收缩
    x1, y1 = x, y
    x2, y2 = x + w, y + h

    # 检查上边界 (y1)
    while y1 < y2:
        if np.all(query_mask[y1, x1:x2] == 255): break
        y1 += 1
    if y1 >= y2: return None  # 无法找到纯净区域

    # 检查下边界 (y2)
    while y2 > y1:
        if np.all(query_mask[y2 - 1, x1:x2] == 255): break
        y2 -= 1
    if y1 >= y2: return None

    # 检查左边界 (x1)
    while x1 < x2:
        if np.all(query_mask[y1:y2, x1] == 255): break
        x1 += 1
    if x1 >= x2: return None

    # 检查右边界 (x2)
    while x2 > x1:
        if np.all(query_mask[y1:y2, x2 - 1] == 255): break
        x2 -= 1
    if x1 >= x2: return None

    # 7. 裁切原始图像
    cropped_pure_image = image[y1:y2, x1:x2]

    return cropped_pure_image


def main():
    print(f"--- 裁切方案 (采样率: {SAMPLE_RATIO * 100}%) ---")
    print(f"正在创建MVTec格式目录: {BASE_OUTPUT_DIR}")

    # 清理旧的数据目录（如果存在），以防数据混淆
    if os.path.exists(BASE_OUTPUT_DIR):
        print(f"清理旧的数据集目录: {BASE_OUTPUT_DIR}")
        shutil.rmtree(BASE_OUTPUT_DIR)

    os.makedirs(TRAIN_GOOD_DIR, exist_ok=True)
    os.makedirs(TEST_CRACK_DIR, exist_ok=True)
    os.makedirs(GT_CRACK_DIR, exist_ok=True)
    os.makedirs(TEST_GOOD_DIR, exist_ok=True)  # 创建一个空的 'test/good'

    all_image_paths = sorted(glob(os.path.join(SOURCE_IMAGE_DIR, '*.jpg')))
    if not all_image_paths:
        print(f"错误: 在 {SOURCE_IMAGE_DIR} 中未找到 .jpg 图像。")
        print(f"请确保您的 '{SOURCE_ROOT}' 文件夹与此脚本位于同一目录。")
        sys.exit(1)

    # --- !!! 新增：采样逻辑 !!! ---
    if SAMPLE_RATIO < 1.0:
        print(f"--- 采样模式: 仅使用 {SAMPLE_RATIO * 100:.0f}% 的图像进行小规模测试 ---")
        num_to_sample = int(len(all_image_paths) * SAMPLE_RATIO)
        if num_to_sample == 0 and len(all_image_paths) > 0:  # 确保至少有1张
            num_to_sample = 1
        # 使用 random.sample 进行随机采样
        image_paths = random.sample(all_image_paths, num_to_sample)
        print(f"将从 {len(all_image_paths)} 张图像中随机采样 {len(image_paths)} 张进行处理。")
    else:
        image_paths = all_image_paths  # 使用所有图像
        print(f"将处理全部 {len(image_paths)} 张图像。")
    # --- 结束新增 ---

    print(f"开始处理 {len(image_paths)} 张图像...")

    train_count = 0
    test_count = 0
    skipped_count = 0

    for img_path in tqdm(image_paths, desc="处理图像"):
        base_name = os.path.basename(img_path)
        name_without_ext = os.path.splitext(base_name)[0]
        mask_path = os.path.join(SOURCE_MASK_DIR, name_without_ext + '.png')

        if not os.path.exists(mask_path):
            continue

        try:
            image = cv2.imread(img_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if image is None or mask is None:
                continue

            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            img_h, img_w = image.shape[:2]
            original_total_area = img_h * img_w

            # 1. 查找并裁切最大的纯净矩形 (用于 train/good)
            pure_crop = find_largest_pure_rectangle(image, mask)

            if pure_crop is not None:
                crop_h, crop_w = pure_crop.shape[:2]
                crop_area = crop_h * crop_w

                if (crop_area / original_total_area) >= MIN_NORMAL_AREA_RATIO:
                    save_path = os.path.join(TRAIN_GOOD_DIR, f"{name_without_ext}_crop.jpg")
                    cv2.imwrite(save_path, pure_crop)
                    train_count += 1
                else:
                    skipped_count += 1
            else:
                skipped_count += 1

            # 2. 复制原始图像和掩码到 test/crack 和 ground_truth/crack
            shutil.copy(img_path, os.path.join(TEST_CRACK_DIR, base_name))
            shutil.copy(mask_path, os.path.join(GT_CRACK_DIR, name_without_ext + '.png'))
            test_count += 1

        except Exception as e:
            print(f"处理 {base_name} 时发生严重错误: {e}")

    print("\n--- 处理完成 ---")
    print(f"总共处理了 {len(image_paths)} 张采样的图像。")
    print(f"生成了 {train_count} 张'大于40%面积'的纯净裁切样本 (在 {TRAIN_GOOD_DIR})")
    print(f"有 {skipped_count} 张图像因裁切后面积不足40%或形状不佳而被跳过。")
    print(f"复制了 {test_count} 张图像和掩码到 'test/crack' 目录。")
    print(f"您的小规模数据集已准备就绪: {BASE_OUTPUT_DIR}")


if __name__ == "__main__":
    if not os.path.exists(SOURCE_ROOT):
        print(f"错误: 原始数据目录 '{SOURCE_ROOT}' 未找到。")
        print("请确保 CRACK500FINAL 文件夹与此脚本在同一目录中。")
        sys.exit(1)
    main()