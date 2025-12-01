import argparse
import logging
import os
import numpy as np
import torch
import torchvision
import threading
import torchvision.transforms as transforms
from tabulate import tabulate
# --- 修复 4 (START): 导入 average_precision_score ---
from sklearn.metrics import roc_auc_score, average_precision_score
# --- 修复 4 (END) ---
from tqdm import tqdm
import math
from PIL import Image
from prefetch_generator import BackgroundGenerator
import matplotlib.pyplot as plt  # <--- 新增：导入 matplotlib 用于绘图
from UniVAD import UniVAD
import gc  # 引入垃圾回收模块

from datasets.mvtec import MVTecDataset
from datasets.visa import VisaDataset
from datasets.mvtec_loco import MVTecLocoDataset
from datasets.brainmri import BrainMRIDataset
from datasets.his import HISDataset
from datasets.resc import RESCDataset
from datasets.liverct import LiverCTDataset
from datasets.chestxray import ChestXrayDataset
from datasets.oct17 import OCT17Dataset


class DataLoaderX(torch.utils.data.DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def resize_tokens(x):
    B, N, C = x.shape
    x = x.view(B, int(math.sqrt(N)), int(math.sqrt(N)), C)
    return x


# --- 将计算逻辑封装为独立函数，以便定期调用 ---
def compute_and_print_metrics(results, obj_list, logger, is_final=False):
    table_ls = []
    auroc_sp_ls = []
    auroc_px_ls = []
    auprc_px_ls = []

    # 定义单个对象的计算逻辑
    def cal_score_task(obj, results_dict, output_list):
        # 筛选当前 obj 的数据
        gt_px = []
        pr_px = []
        gt_sp = []
        pr_sp = []

        has_data = False
        for idxes in range(len(results_dict["cls_names"])):
            if results_dict["cls_names"][idxes] == obj:
                gt_px.append(results_dict["imgs_masks"][idxes].squeeze(1).numpy())
                pr_px.append(results_dict["anomaly_maps"][idxes])
                gt_sp.append(results_dict["gt_sp"][idxes])
                pr_sp.append(results_dict["pr_sp"][idxes])
                has_data = True

        if not has_data:
            return

        gt_px = np.array(gt_px)
        gt_sp = np.array(gt_sp)
        pr_px = np.array(pr_px)
        pr_sp = np.array(pr_sp)

        # -----------------------------------------------------------------
        # --- 方案二修复: 捕获样本级 AUROC 计算错误 ---
        # -----------------------------------------------------------------
        try:
            auroc_sp = roc_auc_score(gt_sp, pr_sp)
        except ValueError:
            auroc_sp = np.nan

        # 像素级 AUROC
        try:
            auroc_px = roc_auc_score(gt_px.ravel(), pr_px.ravel())
        except ValueError:
            auroc_px = np.nan

        # --- 修复 4: 新增 AUPRC-PX 计算 ---
        try:
            auprc_px = average_precision_score(gt_px.ravel(), pr_px.ravel())
        except ValueError:
            auprc_px = np.nan

        # 将结果存入列表 (线程安全的方式通常建议用 append 到各自的 list，这里简化处理)
        row = [
            obj,
            str(np.round(auroc_sp * 100, decimals=1)),
            str(np.round(auroc_px * 100, decimals=1)),
            str(np.round(auprc_px * 100, decimals=1))
        ]

        output_list.append({
            "row": row,
            "auroc_sp": auroc_sp,
            "auroc_px": auroc_px,
            "auprc_px": auprc_px
        })

    # 多线程执行
    temp_results = []
    threads = []
    for obj in obj_list:
        t = threading.Thread(target=cal_score_task, args=(obj, results, temp_results))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    # 整理结果
    for res in temp_results:
        table_ls.append(res["row"])
        auroc_sp_ls.append(res["auroc_sp"])
        auroc_px_ls.append(res["auroc_px"])
        auprc_px_ls.append(res["auprc_px"])

    # 计算均值 (忽略 nan)
    if len(auroc_sp_ls) > 0:
        table_ls.append(
            [
                "mean",
                str(np.round(np.nanmean(auroc_sp_ls) * 100, decimals=1)),
                str(np.round(np.nanmean(auroc_px_ls) * 100, decimals=1)),
                str(np.round(np.nanmean(auprc_px_ls) * 100, decimals=1)),
            ]
        )

    # 生成表格
    headers = ["objects", "auroc_sp", "auroc_px", "auprc_px"]
    results_table = tabulate(table_ls, headers=headers, tablefmt="pipe")

    prefix = "[Final Result]" if is_final else "[Intermediate Result]"
    logger.info(f"\n{prefix}\n{results_table}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Test", add_help=True)
    parser.add_argument("--image_size", type=int, default=448, help="image size")
    parser.add_argument("--k_shot", type=int, default=1, help="k-shot")
    parser.add_argument(
        "--dataset", type=str, default="mvtec", help="train dataset name"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data/mvtec",
        help="path to test dataset",
    )
    parser.add_argument(
        "--save_path", type=str, default=f"./results/", help="path to save results"
    )
    parser.add_argument(
        "--round", type=int, default=3, help="round"
    )
    parser.add_argument("--class_name", type=str, default="None", help="device")
    parser.add_argument("--device", type=str, default="cuda", help="device")

    # --- 新增功能: 定期保存参数 ---
    parser.add_argument("--save_interval", type=int, default=0, help="每N张图片输出一次结果 (0表示不输出)")

    args = parser.parse_args()

    dataset_name = args.dataset
    dataset_dir = args.data_path
    device = args.device
    k_shot = args.k_shot

    image_size = args.image_size
    # 注意：这里的 save_path 仅用于日志，后面的可视化路径是独立构建的
    save_path = args.save_path + "/" + dataset_name + "/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    txt_path = os.path.join(save_path, "log.txt")

    # logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    root_logger.setLevel(logging.WARNING)
    logger = logging.getLogger("test")
    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s",
        datefmt="%y-%m-%d %H:%M:%S",
    )
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(txt_path, mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # record parameters
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")

    UniVAD_model = UniVAD(image_size=args.image_size).to(device)

    # dataset
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )

    gaussion_filter = torchvision.transforms.GaussianBlur(3, 4.0)

    if dataset_name == "mvtec":
        test_data = MVTecDataset(
            root=dataset_dir,
            transform=transform,
            target_transform=transform,
            aug_rate=-1,
            mode="test",
        )
    elif dataset_name == "visa":
        test_data = VisaDataset(
            root=dataset_dir,
            transform=transform,
            target_transform=transform,
            mode="test",
        )
    elif dataset_name == "mvtec_loco":
        test_data = MVTecLocoDataset(
            root=dataset_dir,
            transform=transform,
            target_transform=transform,
            aug_rate=-1,
            mode="test",
        )
    elif dataset_name == "brainmri":
        test_data = BrainMRIDataset(
            root="./data/BrainMRI",
            transform=transform,
            target_transform=transform,
            aug_rate=-1,
            mode="test",
        )
    elif dataset_name == "his":
        test_data = HISDataset(
            root="./data/HIS",
            transform=transform,
            target_transform=transform,
            aug_rate=-1,
            mode="test",
        )
    elif dataset_name == "resc":
        test_data = RESCDataset(
            root="./data/RESC",
            transform=transform,
            target_transform=transform,
            aug_rate=-1,
            mode="test",
        )
    elif dataset_name == "chestxray":
        test_data = ChestXrayDataset(
            root="./data/ChestXray",
            transform=transform,
            target_transform=transform,
            aug_rate=-1,
            mode="test",
        )
    elif dataset_name == "oct17":
        test_data = OCT17Dataset(
            root="./data/OCT17",
            transform=transform,
            target_transform=transform,
            aug_rate=-1,
            mode="test",
        )
    elif dataset_name == "liverct":
        test_data = LiverCTDataset(
            root="./data/LiverCT",
            transform=transform,
            target_transform=transform,
            aug_rate=-1,
            mode="test",
        )
    # -----------------------------------------------------------------
    # --- 修复 1 (START): 兼容 RoadCrack_Crop (修复 NotImplementedError) ---
    # -----------------------------------------------------------------
    elif dataset_name == "RoadCrack_Crop":
        # 我们重用 MVTecDataset，因为它符合 MVTec 格式
        # 关键: 确保 'root' 指向您的数据集路径
        dataset_dir = f'./data/{dataset_name}'  # 即 './data/RoadCrack_Crop'
        logger.info(f"加载自定义数据集: {dataset_dir}")
        test_data = MVTecDataset(
            root=dataset_dir,
            transform=transform,
            target_transform=transform,
            aug_rate=-1,
            mode="test",
        )
    # -----------------------------------------------------------------
    # --- 修复 1 (END) ---
    # -----------------------------------------------------------------
    else:
        raise NotImplementedError("Dataset not supported")

    test_dataloader = DataLoaderX(
        test_data, batch_size=1, shuffle=False, num_workers=8, pin_memory=True
    )

    with torch.no_grad():
        obj_list = [x.replace("_", " ") for x in test_data.get_cls_names()]

    results = {}
    results["cls_names"] = []
    results["imgs_masks"] = []
    results["anomaly_maps"] = []
    results["gt_sp"] = []
    results["pr_sp"] = []

    cls_last = None

    image_transform = transforms.Compose(
        [transforms.Resize((image_size, image_size)), transforms.ToTensor()]
    )

    # 计数器用于 save_interval
    count = 0

    for items in tqdm(test_dataloader):
        count += 1

        image = items["img"].to(device)
        image_pil = items["img_pil"]
        image_path = items["img_path"][0]

        # -----------------------------------------------------------------
        # --- 修复 3 (START): 修复 NameError ---
        # -----------------------------------------------------------------
        # 必须先从 'items' 中定义 'cls_name'
        cls_name = items["cls_name"][0]
        # -----------------------------------------------------------------
        # --- 修复 3 (END) ---
        # -----------------------------------------------------------------

        if args.class_name != "None":
            # 检查现在可以安全运行
            if args.class_name.replace("_", " ") != cls_name:
                continue

        results["cls_names"].append(cls_name)
        gt_mask = items["img_mask"]
        gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
        results["imgs_masks"].append(gt_mask)  # px
        results["gt_sp"].append(items["anomaly"].item())

        if cls_name != cls_last:
            if dataset_name == "mvtec":
                normal_image_paths = [
                    "./data/mvtec/"
                    + cls_name.replace(" ", "_")
                    + "/train/good/"
                    + str(i).zfill(3)
                    + ".png"
                    for i in range(args.round, args.round + k_shot)
                ]
            elif dataset_name == "mvtec_loco":
                normal_image_paths = [
                    "./data/mvtec_loco_caption/"
                    + cls_name.replace(" ", "_")
                    + "/train/good/"
                    + str(i).zfill(3)
                    + ".png"
                    for i in range(args.round, args.round + k_shot)
                ]
            elif dataset_name == "visa":
                if cls_name.replace(" ", "_") in [
                    "capsules",
                    "cashew",
                    "chewinggum",
                    "fryum",
                    "pipe_fryum",
                ]:
                    normal_image_paths = [
                        "./data/VisA_pytorch/1cls/"
                        + cls_name.replace(" ", "_")
                        + "/train/good/"
                        + str(i).zfill(3)
                        + ".JPG"
                        for i in range(args.round, args.round + k_shot)
                    ]
                else:
                    normal_image_paths = [
                        "./data/VisA_pytorch/1cls/"
                        + cls_name.replace(" ", "_")
                        + "/train/good/"
                        + str(i).zfill(4)
                        + ".JPG"
                        for i in range(args.round, args.round + k_shot)
                    ]
            elif dataset_name in [
                "his",
                "oct17",
                "chestxray",
                "brainmri",
                "liverct",
                "resc",
            ]:
                dir = (
                        "./data/"
                        + cls_name.replace(" ", "_")
                        + "/train/good"
                )
                files = sorted(os.listdir(dir))[:k_shot]
                normal_image_paths = [os.path.join(dir, file) for file in files]
            # -----------------------------------------------------------------
            # --- 修复 2 (START): 兼容 RoadCrack_Crop (修复 FileNotFoundError) ---
            # -----------------------------------------------------------------
            elif dataset_name == "RoadCrack_Crop":
                # 关键修复：确保路径指向 "./data/..." 而不是 "./masks/..."
                dir = (
                        f"./data/{dataset_name}/"  # <-- 修复了这里的路径
                        + cls_name.replace(" ", "_")
                        + "/train/good"
                )

                files = os.listdir(dir)
                # 修复：确保在 k_shot 大于可用文件数时不会崩溃
                if len(files) < k_shot:
                    logger.warning(f"警告: 期望 {k_shot} 个参考样本，但只找到 {len(files)} 个。")
                    k_shot_actual = len(files)
                else:
                    k_shot_actual = k_shot

                # 确保 k_shot_actual > 0
                if k_shot_actual == 0:
                    logger.error(f"错误: 在 {dir} 中找不到任何参考样本。")
                    # 在这种情况下我们无法继续
                    raise FileNotFoundError(f"No reference images found in {dir}")

                selected_files = np.random.choice(files, k_shot_actual, replace=False)

                normal_image_paths = [os.path.join(dir, file) for file in selected_files]
                logger.info(f"为 {cls_name} 加载 {len(normal_image_paths)} 个参考样本...")
            # -----------------------------------------------------------------
            # --- 修复 2 (END) ---
            # -----------------------------------------------------------------

            # normal_image_path = normal_image_paths[:k_shot]
            normal_images = torch.cat(
                [
                    image_transform(Image.open(x).convert("RGB")).unsqueeze(0)
                    for x in normal_image_paths
                ],
                dim=0,
            ).to(device)

            setup_data = {
                "few_shot_samples": normal_images,
                "dataset_category": cls_name.replace(" ", "_"),
                "image_path": normal_image_paths,
            }
            UniVAD_model.setup(setup_data)
            cls_last = cls_name

        with torch.no_grad():

            pred_value = UniVAD_model(image, image_path, image_pil)
            anomaly_score, anomaly_map = (
                pred_value["pred_score"],
                pred_value["pred_mask"],
            )
            results["anomaly_maps"].append(anomaly_map.detach().cpu().numpy())
            overall_anomaly_score = anomaly_score.item()
            results["pr_sp"].append(overall_anomaly_score)

            # =================================================================
            # --- 新增功能: 逐像素可视化保存 (支持 k_shot 动态命名) ---
            # =================================================================
            # 1. 动态构建保存路径
            vis_dir_name = f"{k_shot}shot_small_test"
            vis_save_path = os.path.join(
                args.save_path,  # results/
                dataset_name,  # RoadCrack_Crop
                cls_name.replace(" ", "_"),  # road_texture
                vis_dir_name,  # 4shot_small_test
                dataset_name  # RoadCrack_Crop (保留原有目录结构习惯)
            )

            if not os.path.exists(vis_save_path):
                os.makedirs(vis_save_path, exist_ok=True)

            # 2. 准备数据
            img_vis = image[0].permute(1, 2, 0).cpu().numpy()
            gt_vis = gt_mask[0].squeeze().cpu().numpy()
            score_vis = anomaly_map.detach().cpu().numpy().squeeze()

            # 3. 绘图 (三联图)
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))

            axes[0].imshow(img_vis)
            axes[0].set_title("Original Image")
            axes[0].axis("off")

            axes[1].imshow(gt_vis, cmap="gray")
            axes[1].set_title("Ground Truth")
            axes[1].axis("off")

            im_score = axes[2].imshow(score_vis, cmap="jet")
            axes[2].set_title("Anomaly Prediction")
            axes[2].axis("off")
            plt.colorbar(im_score, ax=axes[2], fraction=0.046, pad=0.04)

            # 4. 保存
            file_name = os.path.basename(image_path)
            save_full_path = os.path.join(vis_save_path, file_name)
            plt.savefig(save_full_path, bbox_inches='tight', pad_inches=0.1)
            plt.close(fig)
            # =================================================================

            # -----------------------------------------------------------------
            # --- 内存管理修复 (关键修改) ---
            # -----------------------------------------------------------------
            # 手动删除不再需要的变量，并清理显存
            del pred_value, anomaly_score, anomaly_map
            # 同样删除可视化变量
            del img_vis, gt_vis, score_vis

            # 清理 CUDA 缓存
            torch.cuda.empty_cache()
            # 如果内存依然紧张，可以取消下面这行的注释进行强制垃圾回收（会略微影响速度）
            # gc.collect()
            # -----------------------------------------------------------------

        # --- 定期输出结果 ---
        if args.save_interval > 0 and count % args.save_interval == 0:
            logger.info(f"\n[Process Log] Processed {count} images.")
            compute_and_print_metrics(results, obj_list, logger, is_final=False)

    # 最终结果
    compute_and_print_metrics(results, obj_list, logger, is_final=True)