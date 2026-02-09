import argparse
import os
import glob
import torch
import torch.nn as nn
from utils.logger import setup_logger
from utils.iotools import load_config, check_cfg_keys, save_config
from datasets.loader import make_dataloader
from modeling.modeling_builder import make_model
from losses.losses_builder import make_loss
from solver.optimizer import make_optimizer
from solver.scheduler import make_scheduler
from engine.trainer import do_train


def get_available_configs(config_dir="configs"):
    """列出目录下所有可用的 yml 配置文件"""
    if not os.path.exists(config_dir):
        return []
    return glob.glob(os.path.join(config_dir, "*.yml"))


def train():
    # 1. 解析命令行参数
    parser = argparse.ArgumentParser(description="HBF-ReID Training")
    parser.add_argument(
        "--config_file",
        default="configs/msmt_hbf_vit.yml",  # 修改配置文件可以修改运行的数据集，
        help="配置文件路径",
        type=str
    )
    args = parser.parse_args()

    # 智能路径修正
    if not os.path.exists(args.config_file):
        potential_path = os.path.join("configs", os.path.basename(args.config_file))
        if os.path.exists(potential_path):
            print(f"提示: '{args.config_file}' 未找到，已自动定位到 '{potential_path}'")
            args.config_file = potential_path
        else:
            available_configs = get_available_configs()
            error_msg = f"\n❌ 错误: 找不到配置文件: {args.config_file}\n"
            error_msg += f"📂 configs/ 目录下发现以下可用配置:\n"
            for cfg_path in available_configs:
                error_msg += f"   - {cfg_path}\n"
            raise FileNotFoundError(error_msg)

    # 2. 加载配置
    cfg = load_config(args.config_file)
    check_cfg_keys(cfg)

    # 初始化日志
    save_config(cfg, cfg.OUTPUT_DIR)  # OUTPUT_DIR是输出地址，在yml文件中
    logger = setup_logger("reid", cfg.OUTPUT_DIR)
    logger.info(">>> 成功加载配置文件: {}".format(args.config_file))
    logger.info(">>> 实验结果将保存至: {}".format(cfg.OUTPUT_DIR))

    # 设置随机种子
    torch.manual_seed(cfg.SOLVER.SEED)  # SEED是设置随机种子，在yml文件中
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.SOLVER.SEED)
        torch.backends.cudnn.deterministic = True

    # 3. 准备数据
    train_loader, val_loader, num_query, num_classes, num_cameras = make_dataloader(cfg)
    logger.info(f"数据加载完成: {num_classes} IDs, {num_cameras} Cameras")

    # 4. 构建模型
    model = make_model(cfg, num_classes=num_classes, camera_num=num_cameras, view_num=0)

    # =================================================================================
    # [核心修复] 必须在构建优化器之前，将模型转到 GPU
    # =================================================================================
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)
    logger.info(f"模型已移动至设备: {device}")

    # 5. 配置检查 (Center Loss 维度校验)
    if cfg.MODEL.IF_WITH_CENTER == 'yes':
        config_feat_dim = getattr(cfg.MODEL, 'FEAT_DIM', 768)

        # [核心修复] 兼容 ViT (embed_dim) 和 ResNet (in_planes) 的属性名
        if hasattr(model, 'in_planes'):
            model_feat_dim = model.in_planes
        elif hasattr(model, 'embed_dim'):
            model_feat_dim = model.embed_dim
        else:
            # 最后的保底，ViT-Base 通常是 768
            model_feat_dim = 768
            logger.warning(f"无法自动获取模型特征维度，默认假设为 {model_feat_dim}")

        if config_feat_dim != model_feat_dim:
            logger.error(f"Config Mismatch! cfg.FEAT_DIM({config_feat_dim}) != Model({model_feat_dim})")
            raise ValueError("维度配置错误")

    # 6. 构建损失函数
    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)

    # [核心修复] Center Loss 包含可学习参数 (centers)，必须转到 GPU
    if center_criterion is not None:
        center_criterion.to(device)
        logger.info("Center Loss 模块已移动至 GPU")

    # 7. 构建优化器 (此时传入的 model 和 center_criterion 都在 GPU 上，非常关键！)
    optimizer_results = make_optimizer(cfg, model, center_criterion)
    if isinstance(optimizer_results, tuple):
        optimizer, optimizer_center = optimizer_results
    else:
        optimizer = optimizer_results
        optimizer_center = None

    # 8. 构建学习率调度器
    scheduler = make_scheduler(cfg, optimizer)

    # 9. 启动训练
    logger.info(">>> 准备就绪，正式启动训练流程...")
    do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_func,
        num_query
    )


if __name__ == "__main__":
    train()