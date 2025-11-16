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
from UniVAD import UniVAD

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


def cal_score(obj):
    table = []
    gt_px = []
    pr_px = []
    gt_sp = []
    pr_sp = []

    table.append(obj)
    for idxes in range(len(results["cls_names"])):
        if results["cls_names"][idxes] == obj:
            gt_px.append(results["imgs_masks"][idxes].squeeze(1).numpy())
            pr_px.append(results["anomaly_maps"][idxes])
            gt_sp.append(results["gt_sp"][idxes])
            pr_sp.append(results["pr_sp"][idxes])
    gt_px = np.array(gt_px)
    gt_sp = np.array(gt_sp)
    pr_px = np.array(pr_px)
    pr_sp = np.array(pr_sp)

    # -----------------------------------------------------------------
    # --- 方案二修复 (START): 捕获样本级 AUROC 计算错误 ---
    # -----------------------------------------------------------------
    try:
        auroc_sp = roc_auc_score(gt_sp, pr_sp)
    except ValueError as e:
        # logger 是在主线程中定义的，在线程中访问它
        logger.warning(f"无法计算 {obj} 的 sample-level AUROC: {e}")
        auroc_sp = np.nan  # 将无法计算的分数记为 nan
    # -----------------------------------------------------------------
    # --- 方案二修复 (END) ---
    # -----------------------------------------------------------------

    # 像素级 AUROC
    auroc_px = roc_auc_score(gt_px.ravel(), pr_px.ravel())

    # --- 修复 4 (START): 新增 AUPRC-PX 计算 ---
    # 鉴于像素不平衡，AUPRC 是一个很好的补充指标
    try:
        auprc_px = average_precision_score(gt_px.ravel(), pr_px.ravel())
    except ValueError as e:
        logger.warning(f"无法计算 {obj} 的 pixel-level AUPRC: {e}")
        auprc_px = np.nan
    # --- 修复 4 (END) ---

    table.append(str(np.round(auroc_sp * 100, decimals=1)))
    table.append(str(np.round(auroc_px * 100, decimals=1)))
    # --- 修复 4 (START): 添加 AUPRC-PX 到表格 ---
    table.append(str(np.round(auprc_px * 100, decimals=1)))
    # --- 修复 4 (END) ---

    table_ls.append(table)
    auroc_sp_ls.append(auroc_sp)  # 添加计算出的值 (或 nan)
    auroc_px_ls.append(auroc_px)
    # --- 修复 4 (START): 添加 AUPRC-PX 到列表 ---
    auprc_px_ls.append(auprc_px)
    # --- 修复 4 (END) ---


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
    args = parser.parse_args()

    dataset_name = args.dataset
    dataset_dir = args.data_path
    device = args.device
    k_shot = args.k_shot

    image_size = args.image_size
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

    for items in tqdm(test_dataloader):
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

    # metrics
    table_ls = []
    auroc_sp_ls = []
    auroc_px_ls = []
    # --- 修复 4 (START): 初始化 AUPRC-PX 列表 ---
    auprc_px_ls = []
    # --- 修复 4 (END) ---

    threads = [None] * 20
    idx = 0
    for obj in tqdm(obj_list):
        threads[idx] = threading.Thread(target=cal_score, args=(obj,))
        threads[idx].start()
        idx += 1

    for i in range(idx):
        threads[i].join()

    # logger
    # -----------------------------------------------------------------
    # --- 方案二修复 (START): 使用 np.nanmean 忽略 nan 值计算均值 ---
    # -----------------------------------------------------------------
    table_ls.append(
        [
            "mean",
            str(np.round(np.nanmean(auroc_sp_ls) * 100, decimals=1)),  # 使用 nanmean
            str(np.round(np.nanmean(auroc_px_ls) * 100, decimals=1)),  # 使用 nanmean
            # --- 修复 4 (START): 添加 AUPRC-PX 均值 ---
            str(np.round(np.nanmean(auprc_px_ls) * 100, decimals=1)),  # 使用 nanmean
            # --- 修复 4 (END) ---
        ]
    )
    # -----------------------------------------------------------------
    # --- 方案二修复 (END) ---
    # -----------------------------------------------------------------

    results = tabulate(
        table_ls,
        headers=[
            "objects",
            "auroc_sp",
            "auroc_px",
            # --- 修复 4 (START): 添加 AUPRC-PX 标题 ---
            "auprc_px",
            # --- 修复 4 (END) ---
        ],
        tablefmt="pipe",
    )
    logger.info("\n%s", results)