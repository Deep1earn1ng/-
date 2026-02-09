import torch
import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth
from .triplet_loss import TripletLoss
from .center_loss import CenterLoss
from .circle_loss import CircleLoss


def make_loss(cfg, num_classes):
    """
    损失函数构建工厂。
    """
    # 1. ID Loss (Cross Entropy)
    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        id_loss_func = CrossEntropyLabelSmooth(num_classes=num_classes)
    else:
        id_loss_func = F.cross_entropy

    # 2. Metric Loss (支持 Triplet 或 Circle)
    # 默认回退到 'triplet'，以匹配大多数稳健的 Baseline
    metric_loss_type = getattr(cfg.MODEL, 'METRIC_LOSS_TYPE', 'triplet')

    if metric_loss_type == 'circle':
        # Circle Loss 参数: m=0.25, gamma=64 是 ReID 的黄金组合
        metric_loss_func = CircleLoss(m=0.25, gamma=64)
    else:
        # Triplet Loss: 带有 Hard Mining
        metric_loss_func = TripletLoss(margin=cfg.SOLVER.MARGIN)

    # 3. Center Loss 初始化
    center_criterion = None
    if cfg.MODEL.IF_WITH_CENTER == 'yes':
        # ViT-Base 默认维度为 768
        feat_dim = getattr(cfg.MODEL, 'FEAT_DIM', 768)
        center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim)

    def loss_func(score, feat, target):
        """
        Args:
            score: 预测分类分数 (logits)
            feat:  归一化后的特征向量
            target: 真实标签
        """
        # 1. 特征转 Float32 (AMP 稳定性保障)
        if feat.dtype == torch.float16:
            feat = feat.float()

        # 2. ID Loss 计算
        if isinstance(score, list):
            # JPM 多分支模式：计算所有分支 Loss 的加权和
            weights = getattr(cfg.MODEL, 'JPM_LOSS_WEIGHTS', [1.0] * len(score))
            # 确保 score 也是 float32
            id_loss = sum([w * id_loss_func(s.float(), target) for s, w in zip(score, weights)]) / sum(weights)
        else:
            id_loss = id_loss_func(score.float(), target)

        # 3. Metric Loss 计算
        metric_loss = metric_loss_func(feat, target)

        # 组合基础 Loss
        # 注意：这里复用了 TRIPLET_LOSS_WEIGHT 参数名，即使使用的是 Circle Loss
        total_loss = cfg.MODEL.ID_LOSS_WEIGHT * id_loss + \
                     cfg.MODEL.TRIPLET_LOSS_WEIGHT * metric_loss

        # 4. Center Loss 计算 [关键修正点]
        if center_criterion is not None:
            # 优先读取 Config 中的 CENTER_LOSS_WEIGHT (建议 0.005)
            # 如果 Config 未定义，则使用默认值 0.0005
            weight = getattr(cfg.MODEL, 'CENTER_LOSS_WEIGHT', 0.0005)
            total_loss += weight * center_criterion(feat, target)

        return total_loss

    return loss_func, center_criterion