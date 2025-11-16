import os
import cv2
import numpy as np
import shutil
import random
from tqdm import tqdm


def find_largest_pure_rectangle(mask_path):
    """
    (最终修正版)
    加载掩码 (0=正常, 255=裂缝)，并使用 "最大矩形直方图" 算法
    找到全局最大的、100%纯净（全0）的矩形。

    该区域面积必须不小于原图的40%。
    """

    # 1. 以灰度模式加载掩码
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"警告: 无法加载掩码 {mask_path}。跳过此文件。")
        return None

    # 2. 获取原图总面积和40%阈值
    height, width = mask.shape
    total_area = height * width
    min_required_area = total_area * 0.4

    # 3. 反转掩码，使 0=裂缝, 1=正常
    # 我们寻找 "0" (正常) 区域，所以将 0 变为 1
    normal_mask = (mask == 0).astype(np.uint8)

    # 4. 基于 "最大矩形直方图" 的动态规划

    # hist (histogram) 存储从 (y, x) 向上连续为 1 (正常) 的高度
    hist = np.zeros(width, dtype=int)
    max_area = 0
    best_rect = (0, 0, 0, 0)  # x, y, w, h

    for y in range(height):
        # 4.1 更新当前行的直方图
        for x in range(width):
            # 如果当前 (y, x) 是正常的 (1)，则高度+1；否则重置为0
            hist[x] = hist[x] + 1 if normal_mask[y, x] == 1 else 0

        # 4.2 在当前直方图 (hist) 中寻找最大矩形
        # O(N) 栈 (stack) 解决方案
        stack = [-1]
        for i, h in enumerate(hist):
            # 当遇到比栈顶更矮的柱子时，开始计算
            while stack[-1] != -1 and hist[stack[-1]] >= h:
                h_pop = hist[stack.pop()]  # 弹出的柱子高度
                w_pop = i - stack[-1] - 1  # 宽度
                area = h_pop * w_pop

                if area > max_area:
                    max_area = area
                    # (y - h_pop + 1) 是矩形的起始 y 坐标
                    best_rect = (stack[-1] + 1, y - h_pop + 1, w_pop, h_pop)
            stack.append(i)

        # 4.3 处理栈中剩余的柱子 (它们可以一直延伸到最右边)
        while stack[-1] != -1:
            h_pop = hist[stack.pop()]
            w_pop = width - stack[-1] - 1  # 宽度延伸到尽头
            area = h_pop * w_pop

            if area > max_area:
                max_area = area
                best_rect = (stack[-1] + 1, y - h_pop + 1, w_pop, h_pop)

    # 5. 检查面积是否达到40% (已删除 print 提示)
    if max_area < min_required_area:
        return None

    # 安全检查，防止找到 0 面积
    if best_rect[2] == 0 or best_rect[3] == 0:
        return None

    return best_rect


def process_and_save(file_list, source_img_dir, source_mask_dir, dest_img_dir, dest_mask_dir, crop_normal_region,
                     set_name=""):
    """
    处理文件列表并根据指令保存它们。
    """

    os.makedirs(dest_img_dir, exist_ok=True)
    if dest_mask_dir:
        os.makedirs(dest_mask_dir, exist_ok=True)

    for filename_jpg in tqdm(file_list, desc=f"处理 {set_name}"):

        # 确定源文件路径
        base_name = os.path.splitext(filename_jpg)[0]
        filename_png = base_name + '.png'

        src_img_path = os.path.join(source_img_dir, filename_jpg)
        src_mask_path = os.path.join(source_mask_dir, filename_png)

        if not os.path.exists(src_img_path) or not os.path.exists(src_mask_path):
            print(f"警告: 找不到 {filename_jpg} 或 {filename_png} 的文件对。跳过。")
            continue

        if crop_normal_region:
            # --- 逻辑 1: 裁剪最大纯净区域 (用于 train/good 和 test/good) ---

            # 找到最大纯净区域 (使用新的稳健算法)
            rect = find_largest_pure_rectangle(src_mask_path)
            if rect is None:
                # rect 为 None 意味着找不到，或找到的区域太小 (<40%)
                continue

            x, y, w, h = rect

            # 加载原图并裁剪
            image = cv2.imread(src_img_path)
            if image is None:
                print(f"警告: 无法加载图片 {src_img_path}。跳过。")
                continue

            cropped_image = image[y:y + h, x:x + w]

            # 定义目标路径
            dest_img_path = os.path.join(dest_img_dir, base_name + f"_crop_{x}_{y}.png")
            cv2.imwrite(dest_img_path, cropped_image)

            # 如果是为 test/good 创建，我们需要一个全黑的 ground_truth 掩码
            if dest_mask_dir:
                # 创建一个全黑的掩码 (与裁剪的图片大小相同)
                blank_mask = np.zeros((h, w), dtype=np.uint8)
                dest_mask_path = os.path.join(dest_mask_dir, base_name + f"_crop_{x}_{y}.png")
                cv2.imwrite(dest_mask_path, blank_mask)

        else:
            # --- 逻辑 2: 复制异常文件 (用于 test/crack) ---

            # 复制原图
            dest_img_path = os.path.join(dest_img_dir, filename_jpg)
            shutil.copy(src_img_path, dest_img_path)

            # 复制对应的真值掩码
            if dest_mask_dir:
                dest_mask_path = os.path.join(dest_mask_dir, filename_png)
                shutil.copy(src_mask_path, dest_mask_path)


def main():
    # 1. 定义源路径 (CRACK500FINAL)
    source_img_dir = './CRACK500FINAL/JPEGImages'
    source_mask_dir = './CRACK500FINAL/SegmentationClass'

    # 2. 定义目标 MVTec 格式的基路径
    output_base = './data/RoadCrack_Crop/road_texture'

    # 清理旧目录
    if os.path.exists(output_base):
        print(f"正在清理旧目录: {output_base}")
        shutil.rmtree(output_base)
    print("创建新目录结构...")

    # 3. 定义目标路径
    dest_train_good_img = os.path.join(output_base, 'train', 'good')

    dest_test_good_img = os.path.join(output_base, 'test', 'good')
    dest_test_crack_img = os.path.join(output_base, 'test', 'crack')

    dest_gt_good_mask = os.path.join(output_base, 'ground_truth', 'good')
    dest_gt_crack_mask = os.path.join(output_base, 'ground_truth', 'crack')

    # 4. 收集并拆分文件 (10%/10%/10% 非重叠逻辑)
    try:
        all_files = [f for f in os.listdir(source_img_dir) if f.endswith('.jpg')]
        random.shuffle(all_files)
    except FileNotFoundError:
        print(f"错误: 找不到源目录 {source_img_dir}。请检查路径。")
        return

    total_count = len(all_files)
    if total_count == 0:
        print(f"错误: 在 {source_img_dir} 中找不到 .jpg 文件。")
        return

    # 按照 10% / 10% / 10% 拆分
    train_count = int(total_count * 0.1)
    test_crack_count = int(total_count * 0.1)
    test_good_count = int(total_count * 0.1)

    if total_count < 3 or train_count == 0 or test_crack_count == 0 or test_good_count == 0:
        print(f"错误: 数据集太小 (总共 {total_count} 张图)。无法按 10% 拆分。")
        print(f"至少需要约 30 张图才能分别为 train/test-crack/test-good 分配至少1张图。")
        return

    # 确保三组不重叠
    train_files = all_files[0: train_count]

    test_crack_files = all_files[train_count: train_count + test_crack_count]

    test_good_files = all_files[train_count + test_crack_count: train_count + test_crack_count + test_good_count]

    print(f"数据集拆分完毕:")
    print(f" - 总文件数: {total_count}")
    print(f" - train/good 来源文件数: {len(train_files)}")
    print(f" - test/crack 来源文件数: {len(test_crack_files)}")
    print(f" - test/good 来源文件数: {len(test_good_files)}")
    print(f" - 未使用文件数: {total_count - len(train_files) - len(test_crack_files) - len(test_good_files)}")
    print("-" * 30)

    # 5. 处理文件

    # --- A. 创建训练集 (train/good) ---
    process_and_save(train_files, source_img_dir, source_mask_dir,
                     dest_train_good_img,
                     None,  # train/good 不需要 ground_truth
                     crop_normal_region=True,
                     set_name="train/good")

    # --- B. 创建测试集 (test/crack) ---
    process_and_save(test_crack_files, source_img_dir, source_mask_dir,
                     dest_test_crack_img,
                     dest_gt_crack_mask,
                     crop_normal_region=False,
                     set_name="test/crack")

    # --- C. 创建测试集 (test/good) ---
    process_and_save(test_good_files, source_img_dir, source_mask_dir,
                     dest_test_good_img,
                     dest_gt_good_mask,
                     crop_normal_region=True,
                     set_name="test/good")

    print("-" * 30)
    print("数据准备完成！")


if __name__ == "__main__":
    main()