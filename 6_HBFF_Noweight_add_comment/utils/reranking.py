import torch
import numpy as np
import gc


def re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3):
    """
    [GPU加速 + 低内存优化版] k-reciprocal re-ranking

    针对硬件: RTX 4070Ti (12GB VRAM) + 32GB/40GB System RAM
    针对数据: MSMT17 (93k samples)

    策略:
    1. 特征常驻 GPU (仅约 300MB)。
    2. GPU 分块计算欧氏距离并 Top-K 排序 (利用 4070Ti 算力，速度极快)。
    3. CPU 接收 Top-K 结果，使用 Float16 构建稀疏图 (解决内存瓶颈)。
    """
    # 1. 设置设备：优先使用 GPU 加速
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"  [Re-Ranking] 🚀 启用 GPU 分块加速模式 (利用 {torch.cuda.get_device_name(0)})...")
    else:
        device = torch.device('cpu')
        print("  [Re-Ranking] 未检测到 GPU，回退到 CPU 模式...")

    # 2. 数据转换
    if isinstance(qf, np.ndarray): qf = torch.from_numpy(qf)
    if isinstance(gf, np.ndarray): gf = torch.from_numpy(gf)

    num_query = qf.shape[0]
    num_gallery = gf.shape[0]
    num_total = num_query + num_gallery

    # 将所有特征一次性上传到 GPU (MSMT17 特征仅约 300MB，非常安全)
    # 使用 float32 保证精度
    feat = torch.cat([qf, gf], dim=0).to(device)

    print(f"  [1/5] GPU 分块计算欧氏距离 & Top-K (N={num_total})...")

    # 3. 预分配 CPU 内存用于存储结果
    # 3.1 仅存储 Query 对 Gallery 的完整距离 (用于最终融合)
    #     Size: 11659 * 82161 * 4 bytes ≈ 3.6 GB (CPU RAM)
    original_dist_q2g = torch.zeros(num_query, num_gallery, dtype=torch.float32)

    # 3.2 存储 Top-K 索引和距离 (替代完整的 NxN 矩阵)
    #     Size: 93820 * 50 * 4 bytes ≈ 18 MB (极小)
    #     只需要 k1+1 个邻居，多存一点余量防止边界效应
    search_k = max(k1 + 10, 50)
    initial_rank = torch.zeros(num_total, search_k, dtype=torch.int32)
    initial_dist = torch.zeros(num_total, search_k, dtype=torch.float32)

    # 4. GPU 分块计算循环
    # 显存占用估算:
    # block_size=4096 -> 距离矩阵 4096 * 93820 * 4 bytes ≈ 1.5 GB
    # 加上中间变量，12GB 显存绰绰有余
    block_size = 4096

    # 预先计算所有特征的平方和 (x^2)，避免在循环中重复计算
    # x_norm: [N, 1]
    all_x_norm = torch.pow(feat, 2).sum(dim=1, keepdim=True)

    for i in range(0, num_total, block_size):
        # 打印进度
        if i % (block_size * 2) == 0:
            print(f"   -> Processing batch {i}/{num_total}...")

        end = min(i + block_size, num_total)

        # 取出当前块的特征: [B, D]
        feat_block = feat[i:end, :]
        x_norm_block = all_x_norm[i:end, :]

        # --- GPU 计算核心区域 ---
        # dist^2 = x^2 + y^2 - 2xy
        # y^2 就是 all_x_norm.t()

        # 1. 矩阵乘法 -2xy: [B, N]
        dist_block = torch.addmm(all_x_norm.t(), feat_block, feat.t(), beta=1, alpha=-2)

        # 2. 加上 x^2
        dist_block.add_(x_norm_block)

        # 3. 开根号 (Clamp 防止 NaN)
        dist_block = dist_block.clamp(min=1e-12).sqrt()

        # --- 数据回传与保存 ---

        # A. 保存 Query-to-Gallery 的距离 (用于最后一步)
        # 如果当前块属于 Query 部分
        if i < num_query:
            valid_q_rows = min(end, num_query) - i
            if valid_q_rows > 0:
                # 必须 .cpu() 回传
                original_dist_q2g[i:i + valid_q_rows, :] = \
                    dist_block[:valid_q_rows, num_query:].cpu()

        # B. GPU 排序 (Top-K)
        # 这一步在 GPU 上极快
        vals, idxs = torch.topk(dist_block, k=search_k, largest=False, dim=1)

        # C. 回传 CPU 并存储
        initial_rank[i:end, :] = idxs.cpu().int()
        initial_dist[i:end, :] = vals.cpu().float()

        # 清理临时变量
        del dist_block, vals, idxs

    # 释放 GPU 显存 (特征不再需要)
    del feat, all_x_norm
    torch.cuda.empty_cache()

    print("  [2/5] 构建稀疏权重矩阵 V (CPU Float16)...")

    # 5. 后续逻辑全部在 CPU 上进行 (逻辑同之前，内存优化版)
    # NxN Float16 矩阵约占 16.5 GB -> 确保你有 32GB+ 内存
    V = torch.zeros(num_total, num_total, dtype=torch.float16)

    for i in range(num_total):
        # 利用 Top-K 索引快速构建
        forward_k1 = initial_rank[i, :k1 + 1]
        backward_k1 = initial_rank[forward_k1.long(), :k1 + 1]

        mask = (backward_k1 == i).any(dim=1)
        k_reciprocal_idx = forward_k1[mask].long()

        # 获取对应距离
        # 利用 mask 从 initial_dist 中筛选
        dist_vals = initial_dist[i, :k1 + 1][mask]

        if dist_vals.numel() > 0:
            v_vals = torch.exp(-dist_vals)
            v_vals = v_vals / torch.sum(v_vals)
            V[i, k_reciprocal_idx] = v_vals.half()

    # 6. Query Expansion
    # 注意：这一步会创建 V_qe，需要额外的 16.5GB 内存。
    # 总峰值内存 = 16.5(V) + 16.5(V_qe) + 3.6(Dist) ≈ 36.6 GB
    # 40GB 总内存应该刚好能跑 (会用到 Swap)
    if k2 > 1:
        print("  [3/5] Query Expansion (High Memory Usage)...")
        try:
            V_qe = torch.zeros_like(V)
            for i in range(num_total):
                nbrs = initial_rank[i, :k2].long()
                V_qe[i, :] = torch.mean(V[nbrs, :].float(), dim=0).half()
            V = V_qe
            del V_qe
        except RuntimeError:
            print("  [Warning] 内存不足，跳过 Query Expansion 步骤。")
            gc.collect()

    del initial_rank, initial_dist
    gc.collect()

    print("  [4/5] 计算 Jaccard 距离...")
    # 7. Jaccard Distance (CPU)
    # 建立倒排索引加速
    invIndex = []
    for i in range(num_total):
        invIndex.append(torch.nonzero(V[:, i]).squeeze(-1))

    jaccard_dist = torch.zeros(num_query, num_total, dtype=torch.float32)

    # 进度条
    for i in range(num_query):
        temp_min = torch.zeros(num_total, dtype=torch.float32)
        indNonZero = torch.nonzero(V[i, :]).squeeze(-1)

        for j in indNonZero:
            temp_ind = invIndex[j]
            # 计算 min(V[i,j], V[nodes, j])
            # 注意类型转换 float16 -> float32
            val_i = V[i, j].float()
            vals_nodes = V[temp_ind, j].float()
            temp_min[temp_ind] += torch.min(val_i, vals_nodes)

        jaccard_dist[i, :] = 1 - temp_min / (2. - torch.sum(V[i, :].float()))

    del V, invIndex
    gc.collect()

    print("  [5/5] 最终融合...")
    final_dist = jaccard_dist[:, num_query:] * (1 - lambda_value) + \
                 original_dist_q2g * lambda_value

    return final_dist.numpy()