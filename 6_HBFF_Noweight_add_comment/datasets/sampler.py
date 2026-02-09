import copy
import random
import torch
from collections import defaultdict
import numpy as np
from torch.utils.data.sampler import Sampler


class RandomIdentitySampler(Sampler):
    """
    随机采样 N 个 Identity, 每个 Identity 采样 K 个实例 (Instance)。
    Batch Size = N * K

    Args:
        data_source (list): 数据列表 [(img_path, pid, camid), ...]
        batch_size (int): Batch Size
        num_instances (int): 每个 ID 的实例数量 (K)
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances

        # 构建 PID 到 索引列表 的映射
        self.index_dic = defaultdict(list)
        for index, (_, pid, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        # 估算一个 epoch 的长度
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])

            # [关键修复] 边界处理：如果该 ID 的图片数量少于 num_instances (K)
            if len(idxs) < self.num_instances:
                # 使用 replace=True 进行重采样 (复制样本以填补空缺)
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)

            # 随机打乱
            random.shuffle(idxs)

            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                # 每凑够 K 个样本，就作为一个小的 batch 单元
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        # 组装最终的 batch 序列
        while len(avai_pids) >= self.num_pids_per_batch:
            # 随机选择 N 个 PID
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                # 从该 PID 的可用 batch 单元中取出一个
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)

                # 如果该 PID 的数据取完了，从可用列表中移除
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        return iter(final_idxs)

    def __len__(self):
        return self.length