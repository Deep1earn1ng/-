import torch
import math
from bisect import bisect_right


class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
            self,
            optimizer,
            milestones,
            gamma=0.1,
            warmup_factor=1.0 / 3,
            warmup_epochs=10,
            warmup_method="linear",
            last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones 必须是升序排列的, 例如 [40, 70]"
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "仅支持 'constant' 或 'linear' 预热方式"
            )

        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_epochs = warmup_epochs
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_epochs:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / self.warmup_epochs
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha

        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]


class WarmupCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    """
    [新增] 带预热的余弦退火调度器 (Cosine Annealing with Warmup)

    逻辑:
    1. Warmup阶段: 线性增长
    2. Cosine阶段: 按照余弦曲线平滑下降到 eta_min
    """

    def __init__(
            self,
            optimizer,
            max_epochs,
            warmup_epochs=10,
            warmup_factor=1.0 / 3,
            eta_min=1e-6,  # 最小学习率
            last_epoch=-1,
    ):
        self.max_epochs = max_epochs
        self.warmup_epochs = warmup_epochs
        self.warmup_factor = warmup_factor
        self.eta_min = eta_min
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        # 1. Warmup 阶段
        if self.last_epoch < self.warmup_epochs:
            alpha = float(self.last_epoch) / self.warmup_epochs
            warmup_factor = self.warmup_factor * (1 - alpha) + alpha
            return [base_lr * warmup_factor for base_lr in self.base_lrs]

        # 2. Cosine Annealing 阶段
        else:
            # 当前进度 (0.0 -> 1.0)
            progress = float(self.last_epoch - self.warmup_epochs) / \
                       (self.max_epochs - self.warmup_epochs)
            progress = min(1.0, max(0.0, progress))  # 确保在范围内

            # 余弦计算: 0.5 * (1 + cos(x)) 从 1 降到 0
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))

            return [
                self.eta_min + (base_lr - self.eta_min) * cosine_decay
                for base_lr in self.base_lrs
            ]


def make_scheduler(cfg, optimizer):
    """
    调度器构建工厂。
    [修改] 默认使用余弦退火 (WarmupCosineAnnealingLR)。
    """
    # 获取最大 Epochs，用于计算余弦周期
    max_epochs = cfg.SOLVER.MAX_EPOCHS

    # 你可以通过配置文件新增一个字段来控制，或者直接在这里硬编码切换
    # 这里直接启用 Cosine，因为它对 ViT 效果更好
    scheduler_name = 'cosine'

    if scheduler_name == 'cosine':
        return WarmupCosineAnnealingLR(
            optimizer,
            max_epochs=max_epochs,
            warmup_epochs=cfg.SOLVER.WARMUP_EPOCHS,
            warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
            eta_min=1e-6  # 默认最小学习率
        )
    else:
        # 保留原有选项作为备选
        return WarmupMultiStepLR(
            optimizer,
            cfg.SOLVER.STEPS,
            cfg.SOLVER.GAMMA,
            cfg.SOLVER.WARMUP_FACTOR,
            cfg.SOLVER.WARMUP_EPOCHS,
            cfg.SOLVER.WARMUP_METHOD
        )