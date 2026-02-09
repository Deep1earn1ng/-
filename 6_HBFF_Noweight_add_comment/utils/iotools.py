import os
import errno
import os.path as osp
from yacs.config import CfgNode as CN


def mkdir_if_missing(directory):
    """如果文件夹不存在则创建，处理并发创建的异常"""
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def save_config(cfg, output_dir):
    """将当前配置保存到 output_dir，方便后期追溯实验参数"""
    mkdir_if_missing(output_dir)
    save_path = osp.join(output_dir, "config_resolved.yml")
    with open(save_path, 'w') as f:
        f.write(cfg.dump())
    return save_path


def load_config(config_file):
    """
    加载单一 YAML 文件。
    """
    cfg = CN()
    # 允许递归地添加新键，适配独立的 full config 文件
    cfg.set_new_allowed(True)

    if os.path.exists(config_file):
        cfg.merge_from_file(config_file)
    else:
        # 这里其实在 train.py 已经检查过了，但为了健壮性保留
        raise FileNotFoundError(f"未找到配置文件: {config_file}")

    # 冻结配置，防止后续代码意外修改参数
    cfg.freeze()
    return cfg


def check_cfg_keys(cfg, mandatory_keys=None):
    """
    校验 YAML 是否包含必要的训练参数。
    """
    if mandatory_keys is None:
        mandatory_keys = [
            'MODEL.NAME',
            'MODEL.PRETRAIN_PATH',
            'DATASETS.ROOT_DIR',
            'SOLVER.MAX_EPOCHS',
            'SOLVER.BASE_LR'
        ]

    for key in mandatory_keys:
        parts = key.split('.')
        node = cfg
        try:
            for p in parts:
                node = node[p]
        except (KeyError, AttributeError):
            # 打印更友好的错误提示
            print(f"\n❌ [配置检查失败] 配置文件中缺失必要的参数: '{key}'")
            print(f"👉 请检查你的 YAML 文件: {cfg.get('OUTPUT_DIR', 'unknown path')}")
            raise KeyError(f"配置文件缺失参数: {key}")