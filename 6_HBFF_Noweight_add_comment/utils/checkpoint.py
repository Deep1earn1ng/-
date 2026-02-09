import os
import torch
import logging
import glob
import re


def save_checkpoint(cfg, epoch, model, optimizer, scheduler, is_best_map=False, is_best_rank1=False):
    """
    完善后的模型保存函数：
    1. 状态字典包含随机种子以支持实验复现。
    2. 支持多指标（mAP, Rank-1）最优权重分别保存。
    3. 自动清理策略：仅保留最新的一个周期性权重，以及两个最优权重。
    """
    logger = logging.getLogger("reid.checkpoint")

    # 建立保存目录
    output_dir = cfg.OUTPUT_DIR
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 准备保存的状态字典
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'cfg': cfg,
        # [优化] 使用 getattr 增强鲁棒性，防止配置中缺少 SEED 导致崩溃
        'seed': getattr(cfg.SOLVER, 'SEED', 1234)
    }

    # 1. 保存当前周期权重 (例如 transformer_epoch_10.pth)
    # 这是用于断点续训的基础文件
    filename = os.path.join(output_dir, f"{cfg.MODEL.NAME}_epoch_{epoch}.pth")
    torch.save(state, filename)

    # 2. 如果是 mAP 最佳模型，额外存一份 (覆盖旧的 best_mAP.pth)
    if is_best_map:
        best_map_path = os.path.join(output_dir, f"{cfg.MODEL.NAME}_best_mAP.pth")
        torch.save(state, best_map_path)
        logger.info(f"🏆 保存当前 mAP 最高模型至: {best_map_path}")

    # 3. 如果是 Rank-1 最佳模型，额外存一份 (覆盖旧的 best_rank1.pth)
    if is_best_rank1:
        best_r1_path = os.path.join(output_dir, f"{cfg.MODEL.NAME}_best_rank1.pth")
        torch.save(state, best_r1_path)
        logger.info(f"🚀 保存当前 Rank-1 最高模型至: {best_r1_path}")

    # 4. 权重清理策略：仅保留最新的 1 个周期性模型
    # 匹配模式: transformer_epoch_*.pth
    # 注意: best_mAP 和 best_rank1 不包含 "_epoch_"，因此不会被误删
    pattern = os.path.join(output_dir, f"{cfg.MODEL.NAME}_epoch_*.pth")
    checkpoint_files = glob.glob(pattern)

    # 解析文件名中的 epoch 数字并进行排序
    file_list = []
    for f in checkpoint_files:
        match = re.search(r'epoch_(\d+)\.pth', f)
        if match:
            file_list.append((int(match.group(1)), f))

    # 按照 epoch 编号降序排列（最新的排在前面）
    file_list.sort(key=lambda x: x[0], reverse=True)

    # 如果文件总数超过 1 个，则删除所有更早的周期性权重
    if len(file_list) > 1:
        for _, old_file_path in file_list[1:]:
            try:
                os.remove(old_file_path)
                # 不再输出清理日志，保持控制台整洁
            except OSError as e:
                logger.warning(f"⚠️ 删除文件 {old_file_path} 失败: {e}")


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    """
    加载模型权重。支持推理加载或断点续训加载。
    """
    logger = logging.getLogger("reid.checkpoint")
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ 未找到权重文件: {path}")

    logger.info(f"=> 正在加载权重: {path}")
    checkpoint = torch.load(path, map_location='cpu')

    # 兼容性处理：判断是存的完整 state 还是仅 state_dict
    if 'state_dict' in checkpoint:
        model_state = checkpoint['state_dict']
    else:
        model_state = checkpoint

    # 加载到模型 (strict=False 允许部分权重不匹配，例如微调时)
    msg = model.load_state_dict(model_state, strict=False)
    logger.info(f"模型加载结果: {msg}")

    # 如果提供了优化器和调度器，且 checkpoint 中包含它们，则恢复状态 (用于断点续训)
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("=> 成功恢复优化器状态")

    if scheduler is not None and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])
        logger.info("=> 成功恢复调度器状态")

    return checkpoint.get('epoch', 0)


def load_pretrain_vit(model, pretrain_path):
    """
    专门用于加载 ImageNet 预训练的 ViT 权重 (通常仅包含 model state_dict)。
    """
    logger = logging.getLogger("reid.checkpoint")
    if not os.path.exists(pretrain_path):
        logger.warning(f"⚠️ 未找到预训练权重: {pretrain_path}，将从随机初始化开始。")
        return

    checkpoint = torch.load(pretrain_path, map_location='cpu')
    logger.info(f"=> 加载预训练 ViT: {pretrain_path}")

    # 兼容性处理：如果预训练文件也是完整 checkpoint
    if 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']

    model_dict = model.state_dict()

    # 仅加载形状匹配的键 (过滤掉分类头等不匹配的层)
    pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict and v.shape == model_dict[k].shape}

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    missing_keys = set(model.state_dict().keys()) - set(pretrained_dict.keys())
    # logger.info(f"预训练匹配成功。未匹配到的键: {list(missing_keys)[:5]}...")