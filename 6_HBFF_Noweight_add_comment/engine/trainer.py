import torch
import numpy as np
import logging
import time

# [新增] 引入 thop 用于计算 FLOPs
try:
    from thop import profile
except ImportError:
    profile = None

from utils.meter import MultiItemAverageMeter
from utils.metrics import R1_mAP_eval
from utils.checkpoint import save_checkpoint
from utils.logger import get_rich_progress


def do_train(cfg, model, train_loader, val_loader, optimizer, scheduler, loss_func, num_query):
    """
    重构后的训练引擎：
    1. 分层梯度裁剪：针对 Transformer 与 Classifier 设置不同阈值。
    2. [已修复] 移除 SBSM 手动干预：让 beta 参数自主学习，避免训练崩塌。
    3. 全面评估指标：引入 mINP 作为模型性能的深度参考。
    4. 实时算力显示 (仅显示 TFLOPS，隐藏 img/s)。
    """
    logger = logging.getLogger("reid.train")
    logger.info("开始训练循环 (优化后的 Transformer 稳定性策略已启用)...")

    device = cfg.MODEL.DEVICE
    epochs = cfg.SOLVER.MAX_EPOCHS
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    # 获取每一轮的迭代次数
    iters_per_epoch = len(train_loader)

    # 1. 初始化统计器与 AMP Scaler
    meters = MultiItemAverageMeter()
    model.to(device)
    scaler = torch.amp.GradScaler(device='cuda')

    best_mAP = 0.0
    best_rank1 = 0.0
    best_mINP = 0.0  # 新增：追踪最高 mINP

    # --- [新增] 计算模型理论 GFLOPs (用于计算 TFLOPS) ---
    model_gflops = 0.0
    if profile is not None:
        try:
            # 构造 Dummy Input (Batch Size = 1)
            h, w = cfg.INPUT.SIZE_TRAIN
            dummy_input = torch.randn(1, 3, h, w).to(device)
            # 某些模型 forward 需要 target 和 cam_id
            dummy_label = torch.tensor([0]).to(device)
            dummy_cam = torch.tensor([0]).to(device)

            # 计算 MACs (Multiply-Accumulate Operations)
            # 注意: inputs 需要是 tuple
            logger.info("正在计算模型理论 FLOPs...")
            macs, _ = profile(model, inputs=(dummy_input, dummy_label, dummy_cam), verbose=False)

            # 1 MAC ≈ 2 FLOPs
            model_gflops = 2 * macs / 1e9
            logger.info(f"模型理论计算量: {model_gflops:.2f} GFLOPs")
        except Exception as e:
            logger.warning(f"无法计算 FLOPs (可能是模型结构特殊或缺少thop库): {e}")
    else:
        logger.warning("未检测到 'thop' 库，将跳过 TFLOPS 计算。建议运行: pip install thop")

    # 2. 获取进度条
    progress = get_rich_progress()

    with progress:
        train_task = progress.add_task(
            f"[bold green]Training",
            total=epochs * iters_per_epoch,
            loss_info="Init..."
        )

        for epoch in range(1, epochs + 1):
            meters.reset()
            model.train()

            # [已删除] 导致崩塌的手动 beta 赋值代码已移除

            for iteration, (img, target, cam_id) in enumerate(train_loader):
                # 记录计算开始时间 (包含数据加载后的 GPU 计算时间)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start_time = time.time()

                img = img.to(device)
                target = target.to(device)
                cam_id = cam_id.to(device)

                optimizer.zero_grad()

                with torch.amp.autocast('cuda'):
                    score, feat = model(img, target, cam_id)
                    loss = loss_func(score, feat, target)

                scaler.scale(loss).backward()

                # --- [优化] 分层梯度裁剪 ---
                scaler.unscale_(optimizer)
                backbone_params = []
                other_params = []
                for name, param in model.named_parameters():
                    if 'base' in name:
                        backbone_params.append(param)
                    else:
                        other_params.append(param)

                torch.nn.utils.clip_grad_norm_(backbone_params, max_norm=0.5)
                torch.nn.utils.clip_grad_norm_(other_params, max_norm=1.0)
                # -------------------------

                scaler.step(optimizer)
                scaler.update()

                # 记录计算结束时间
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end_time = time.time()

                # --- [新增] 实时算力计算 ---
                batch_time = end_time - start_time
                # 防止除以零
                batch_time = max(batch_time, 1e-6)

                # 1. 吞吐量 (img/s) - 保留计算用于日志记录，但不显示在进度条
                img_per_sec = img.size(0) / batch_time

                # 2. TFLOPS (Tera Floating-point Operations Per Second)
                # 公式: (单样本GFLOPs * BatchSize) / (时间 * 1000)
                current_tflops = (model_gflops * img.size(0)) / batch_time / 1000

                meters.update({'total_loss': loss.item()}, n=img.size(0))
                current_lr = scheduler.get_lr()[0]

                # 更新进度条显示 (只显示 TFLOPS，不显示 img/s)
                loss_str = f"[red]Loss: {meters.avg('total_loss'):.4f}[/red] [cyan]LR: {current_lr:.2e}[/cyan]"
                flops_str = f"[magenta]{current_tflops:.2f} TFLOPS[/magenta]" if model_gflops > 0 else ""

                progress.update(
                    train_task,
                    advance=1,
                    description=f"[bold green]Epoch {epoch}/{epochs}",
                    loss_info=f"{loss_str} {flops_str}"
                )

                if (iteration + 1) % log_period == 0:
                    logger.debug(f"Epoch[{epoch}] Iteration[{iteration + 1}/{iters_per_epoch}] "
                                 f"Loss: {meters.avg('total_loss'):.4f}, Base Lr: {current_lr:.2e}, "
                                 f"Speed: {img_per_sec:.1f} img/s")

            scheduler.step()

            # 3. 验证阶段
            if epoch % eval_period == 0 or epoch == epochs:
                rank1, mAP, mINP = do_inference(cfg, model, val_loader, num_query)

                if mAP > best_mAP:
                    best_mAP = mAP
                    save_checkpoint(cfg, epoch, model, optimizer, scheduler, is_best_map=True)

                if rank1 > best_rank1:
                    best_rank1 = rank1
                    save_checkpoint(cfg, epoch, model, optimizer, scheduler, is_best_rank1=True)

                if mINP > best_mINP:
                    best_mINP = mINP
                    # save_checkpoint(cfg, epoch, model, optimizer, scheduler, is_best_minp=True)

            if epoch % checkpoint_period == 0 or epoch == epochs:
                save_checkpoint(cfg, epoch, model, optimizer, scheduler)


def do_inference(cfg, model, val_loader, num_query):
    """
    推理提取特征并进行全面评估。
    """
    logger = logging.getLogger("reid.test")
    device = cfg.MODEL.DEVICE

    # 初始化评估器
    evaluator = R1_mAP_eval(
        num_query,
        max_rank=50,
        feat_norm=cfg.TEST.FEAT_NORM,
        reranking=cfg.TEST.RE_RANKING,
        reranking_k1=cfg.TEST.RE_RANKING_K1,
        reranking_k2=cfg.TEST.RE_RANKING_K2
    )
    evaluator.reset()

    model.to(device)
    model.eval()

    with torch.no_grad():
        for img, pid, camid, _, _ in val_loader:
            img = img.to(device)
            with torch.amp.autocast('cuda'):
                feat = model(img)
            evaluator.update((feat, pid, camid))

    cmc, mAP, mINP = evaluator.compute()

    logger.info(f"评估结果: mAP: {mAP:.1%} | Rank-1: {cmc[0]:.1%} | mINP: {mINP:.1%}")

    return cmc[0], mAP, mINP