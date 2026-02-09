import torch


def make_optimizer(cfg, model, center_criterion=None):
    """
    构建优化器。
    核心逻辑：对 Bias 和 BatchNorm 参数不进行权重衰减 (Weight Decay)。
    """
    params = []

    # 遍历模型的所有命名参数
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue

        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY

        # 1. 针对 Bias 项的特殊处理
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS  # 通常设为 0

        # 2. 针对 BatchNorm 层参数不进行衰减
        # 在 Transformer 中，LayerNorm 同样适用此规则
        if "norm" in key or "bn" in key:
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS  # 强制为 0

        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    # 3. 选择优化器算法
    if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
        # [改进] 使用 getattr 设置默认值，防止配置缺失导致崩溃
        momentum = getattr(cfg.SOLVER, 'MOMENTUM', 0.9)
        optimizer = torch.optim.SGD(params, momentum=momentum)
    elif cfg.SOLVER.OPTIMIZER_NAME == 'Adam':
        optimizer = torch.optim.Adam(params, lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    else:
        # [专家建议] 对于 ViT，AdamW 通常比 Adam 表现更稳健
        optimizer = torch.optim.AdamW(params, lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    # 4. 如果有 Center Loss，将其参数加入另一个（或同一个）优化器
    # 注意：Center Loss 通常使用单独的 SGD 优化器以保持稳定，但这里简化处理
    if center_criterion is not None:
        optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=cfg.SOLVER.CENTER_LR)
        return optimizer, optimizer_center

    return optimizer