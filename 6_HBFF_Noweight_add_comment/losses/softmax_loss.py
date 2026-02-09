import torch
import torch.nn as nn


class CrossEntropyLabelSmooth(nn.Module):
    """
    带有标签平滑的交叉熵损失。
    [优化] 移除了不必要的 CPU/GPU 数据搬运，显著提升效率。
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: 模型的预测 logits [batch_size, num_classes]
            targets: 真实的类别标签 [batch_size]
        """
        log_probs = self.logsoftmax(inputs)

        # [优化] 直接在当前设备上创建 one-hot 向量，避免 .cpu() 带来的同步开销
        with torch.no_grad():
            targets_one_hot = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)

        # 标签平滑公式: (1 - epsilon) * one_hot + epsilon / num_classes
        targets_smooth = (1 - self.epsilon) * targets_one_hot + self.epsilon / self.num_classes

        # 计算 Loss
        loss = (- targets_smooth * log_probs).mean(0).sum()
        return loss