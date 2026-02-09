import torch
import torch.nn as nn
import torch.nn.functional as F

class CircleLoss(nn.Module):
    def __init__(self, m=0.25, gamma=64):
        """
        Circle Loss for ReID.
        Args:
            m (float): margin for relaxation (default: 0.25)
            gamma (float): scale factor (default: 64)
        """
        super(CircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()

    def forward(self, embedding, targets):
        """
        Args:
            embedding: (batch_size, feat_dim)
            targets: (batch_size)
        """
        # 1. 特征归一化 (Circle Loss 必须基于 Cosine Similarity)
        embedding = F.normalize(embedding, p=2, dim=1)

        # 2. 计算余弦相似度矩阵
        sim_mat = torch.matmul(embedding, embedding.t())

        # 3. 获取正负样本掩码
        # targets.unsqueeze(1) -> [N, 1], targets.unsqueeze(0) -> [1, N]
        # is_pos: 同一 ID 为 True
        is_pos = targets.view(-1, 1).eq(targets.view(1, -1))
        # is_neg: 不同 ID 为 True
        is_neg = targets.view(-1, 1).ne(targets.view(1, -1))

        # 移除自身的对角线 (self-similarity)
        # 也就是 mask 掉对角线位置
        batch_size = embedding.size(0)
        diag_mask = torch.eye(batch_size, dtype=torch.bool, device=embedding.device)
        is_pos = is_pos & (~diag_mask)  # 正样本不包含自己

        # 4. 提取分值
        # 这里使用 masked_select 会将矩阵展平，这没问题，因为我们只关心集合
        s_p = sim_mat[is_pos]  # 所有正样本对的相似度
        s_n = sim_mat[is_neg]  # 所有负样本对的相似度

        # 极端情况保护：如果 batch 里没有正样本对或负样本对
        if s_p.nelement() == 0 or s_n.nelement() == 0:
            return s_p.new_zeros(1, requires_grad=True)

        # 5. Circle Loss 核心公式
        # alpha_p = clamp(1 + m - s_p)
        # alpha_n = clamp(s_n + m)
        alpha_p = torch.clamp_min(-s_p.detach() + 1 + self.m, min=0.)
        alpha_n = torch.clamp_min(s_n.detach() + self.m, min=0.)

        delta_p = 1 - self.m
        delta_n = self.m

        # logit_p = - alpha_p * (s_p - delta_p) * gamma
        # logit_n = alpha_n * (s_n - delta_n) * gamma
        logit_p = - alpha_p * (s_p - delta_p) * self.gamma
        logit_n = alpha_n * (s_n - delta_n) * self.gamma

        # loss = softplus( logsumexp(logit_n) + logsumexp(logit_p) )
        loss = self.soft_plus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))

        return loss