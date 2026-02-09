import torch


class AverageMeter(object):
    """
    计算并存储变量的当前值、平均值、总和及计数。
    常用于记录单条 Loss 或准确率。
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MultiItemAverageMeter(object):
    """
    多项指标计数器。
    专门用于 HBF 这种多分支模型，可以同时追踪 Global Loss, JPM Loss 等。
    """

    def __init__(self):
        self.content = {}

    def update(self, val_dict, n=1):
        """
        val_dict: 字典格式，如 {'total_loss': 0.8, 'id_loss': 0.4}
        """
        for key, val in val_dict.items():
            if key not in self.content:
                self.content[key] = AverageMeter()

            # 如果输入是 Tensor，先转为标量
            if isinstance(val, torch.Tensor):
                val = val.item()

            self.content[key].update(val, n)

    def avg(self, key):
        """获取某项指标的平均值"""
        if key in self.content:
            return self.content[key].avg
        return 0.0

    def get_str(self):
        """
        将所有指标转化为字符串，方便 logger 打印。
        输出格式: "loss1: 0.123  loss2: 0.456"
        """
        return "  ".join([f"{k}: {m.avg:.4f}" for k, m in self.content.items()])

    def reset(self):
        self.content = {}


def compute_accuracy(output, target, topk=(1,)):
    """
    计算前 K 个预测的准确率。
    在训练循环中实时监控准确率 (Acc) 很有用。
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1.0 / batch_size))
        return res