import torch
from torch import nn


def normalize(x, axis=-1):
    """特征归一化"""
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y):
    """
    计算欧氏距离矩阵
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    # (x-y)^2 = x^2 + y^2 - 2xy
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(x, y.t(), beta=1, alpha=-2)

    # [关键修复] 限制最小值为 1e-12，防止 sqrt(0) 导致梯度爆炸 (NaN)
    dist = dist.clamp(min=1e-12).sqrt()
    return dist


class TripletLoss(nn.Module):
    """
    带难样本挖掘的三元组损失 (Triplet Loss with Hard Mining)。
    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification.
    """

    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        # margin > 0 使用软间隔，triplet loss = max(d_p - d_n + margin, 0)
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: 特征向量 [batch_size, feat_dim]
            targets: 真实标签 [batch_size]
        """
        n = inputs.size(0)

        # 1. 计算成对欧氏距离
        dist = euclidean_dist(inputs, inputs)

        # 2. 挖掘难样本 (Hard Example Mining)
        # mask[i][j] = 1 if targets[i] == targets[j] (Positive)
        # mask[i][j] = 0 if targets[i] != targets[j] (Negative)
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())

        dist_ap, dist_an = [], []
        for i in range(n):
            # Hard Positive: 同一 ID 中距离最远的样本
            # dist[i][mask[i]] 获取第 i 个样本的所有正样本距离
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))

            # Hard Negative: 不同 ID 中距离最近的样本
            # dist[i][mask[i] == 0] 获取第 i 个样本的所有负样本距离
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))

        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # 3. 计算 Ranking Loss
        # y = 1 表示我们希望 dist_an (第二个参数) > dist_ap (第一个参数) + margin
        # MarginRankingLoss 公式: max(0, -y * (x1 - x2) + margin)
        # 这里: max(0, -1 * (dist_an - dist_ap) + margin) = max(0, dist_ap - dist_an + margin)
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)

        return loss