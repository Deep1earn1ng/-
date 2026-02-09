import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from .bases import ImageDataset
from .sampler import RandomIdentitySampler
from . import init_dataset
import numpy as np
import random
# [保持] 同时导入 LSEA 和 自定义的 RandomErasing
from utils.transforms import LSEA, RandomErasing


def train_collate_fn(batch):
    """
    自定义 collate_fn，处理 batch 数据
    """
    imgs, pids, camids, _ = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids


def val_collate_fn(batch):
    imgs, pids, camids, img_paths = zip(*batch)
    camids_batch = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, camids_batch, img_paths


def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_dataloader(cfg):
    # ImageNet 标准化参数
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # [修改] 针对 MSMT17 优化的增强流水线
    train_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3), # Bicubic Interpolation

        # 1. [新增] AutoAugment: 自动搜索的最佳几何/色彩变换策略
        # 极大地丰富 MSMT17 的数据多样性，提分利器
        T.AutoAugment(policy=T.AutoAugmentPolicy.IMAGENET),

        # 2. LSEA: 局部灰度化
        # 增强对颜色变化的鲁棒性，迫使模型关注结构信息
        LSEA(p=0.5),

        T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
        T.Pad(cfg.INPUT.PADDING),
        T.RandomCrop(cfg.INPUT.SIZE_TRAIN),

        T.ToTensor(),
        T.Normalize(mean=mean, std=std),

        # 3. Custom RandomErasing: 随机擦除
        # 使用 [0.0, 0.0, 0.0] 填充 (归一化后的均值)，模拟平均背景遮挡
        RandomErasing(probability=cfg.INPUT.RE_PROB, mean=[0.0, 0.0, 0.0])
    ])

    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST, interpolation=3),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])

    num_workers = cfg.DATALOADER.NUM_WORKERS

    dataset = init_dataset(cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR)

    train_set = ImageDataset(dataset.train, train_transforms)

    if cfg.DATALOADER.SAMPLER == 'softmax':
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True,
            num_workers=num_workers, collate_fn=train_collate_fn,
            worker_init_fn=worker_init_fn
        )
    else:
        # 使用 RandomIdentitySampler (PK Sampler) 保证每个 Batch 包含 K 个 ID 的 N 张图片
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
            sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
            num_workers=num_workers, collate_fn=train_collate_fn,
            worker_init_fn=worker_init_fn
        )

    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)
    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False,
        num_workers=num_workers, collate_fn=val_collate_fn
    )

    num_classes = dataset.num_train_pids
    num_cameras = dataset.num_train_cams

    return train_loader, val_loader, len(dataset.query), num_classes, num_cameras