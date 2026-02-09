import logging
import os
import sys
from rich.logging import RichHandler
from rich.console import Console
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
    SpinnerColumn
)

# 创建全局 Console 实例，确保整个项目的输出风格统一
console = Console()


def setup_logger(name, save_dir, distributed_rank=0):
    """
    配置 Logger：
    1. 控制台输出使用 RichHandler (美观、带颜色)
    2. 文件输出使用 FileHandler (详细、持久化)
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # 避免在多次调用时重复添加 Handler
    if logger.hasHandlers():
        return logger

    # -------------------------------------------------------------------------
    # 1. 控制台处理器 (Console Handler) - 仅在主进程 (Rank 0) 启用
    # -------------------------------------------------------------------------
    if distributed_rank == 0:
        # 使用 RichHandler 替代 StreamHandler
        # show_path=False: 不显示日志来源文件路径，保持界面清爽
        # show_time=True: 显示日志生成时间
        ch = RichHandler(console=console, show_path=False, show_time=True, markup=True)
        ch.setLevel(logging.INFO)

        # Rich 自带格式，这里只需要 msg
        formatter = logging.Formatter("%(message)s")
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    else:
        # 非主进程不输出到控制台，避免 DDP 训练时刷屏
        pass

    # -------------------------------------------------------------------------
    # 2. 文件处理器 (File Handler) - 所有进程或仅主进程记录
    # -------------------------------------------------------------------------
    if save_dir and distributed_rank == 0:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 日志文件名
        log_file = os.path.join(save_dir, "train_log.txt")

        fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        fh.setLevel(logging.DEBUG)

        # 文件日志保持传统详细格式：时间 - 模块名 - 级别 - 消息
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        fh.setFormatter(file_formatter)
        logger.addHandler(fh)

    return logger


def get_rich_progress():
    """
    返回一个配置好的 Rich Progress 对象，用于训练循环中显示进度条。
    包含：任务名、进度条、百分比、10/120 Epochs、已用时间、预计剩余时间、Loss信息。
    """
    return Progress(
        SpinnerColumn(),  # 转圈动画
        TextColumn("[bold blue]{task.description}"),  # 任务描述 (如 "Training")
        BarColumn(bar_width=40),  # 进度条
        "[progress.percentage]{task.percentage:>3.0f}%",  # 百分比
        "•",
        MofNCompleteColumn(),  # 进度 (如 1/120)
        "•",
        TimeElapsedColumn(),  # 已用时间
        "•",
        TimeRemainingColumn(),  # ⏳ 核心需求：预计剩余时间 (ETA)
        "•",
        TextColumn("{task.fields[loss_info]}"),  # 📉 核心需求：动态显示 Loss
        console=console
    )