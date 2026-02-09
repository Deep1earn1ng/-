import torch
import torch.nn as nn


class CenterLoss(nn.Module):
    """
    Center Loss：学习每个类别的中心，并最小化特征与中心之间的距离。
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    """

    def __init__(self, num_classes=751, feat_dim=768, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu  # 保留参数以兼容接口，但内部逻辑已改为自适应

        # [优化] 使用 nn.Parameter 注册参数，使其能随模型自动进行 .cuda() / .cpu() 转移
        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: 特征向量 [batch_size, feat_dim]
            labels: 真实标签 [batch_size]
        """
        batch_size = x.size(0)

        # [优化] 动态获取当前 tensor 所在的设备，不再依赖初始化时的 use_gpu
        device = x.device

        # 计算特征 x 与中心 centers 之间的欧氏距离矩阵
        # distmat = x^2 + c^2 - 2xc
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()

        # addmm_ 相当于: distmat = 1 * distmat - 2 * (x @ centers.t())
        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)

        # 仅计算 batch 中出现的类别对应的 loss
        # 创建类别索引矩阵
        classes = torch.arange(self.num_classes, device=device).long()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)

        # 生成 mask: 标记每个样本对应的真实类别位置
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        # 提取对应类别的距离
        dist = distmat * mask.float()

        # 数值稳定性截断 (虽然理论上平方和非负，但防止浮点溢出或极小值)
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss