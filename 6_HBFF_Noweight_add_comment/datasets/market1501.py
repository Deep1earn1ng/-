import glob
import re
import os.path as osp
from .bases import BaseDataset


class Market1501(BaseDataset):
    """
    Market1501 数据集解析类。
    文件名格式示例: 0001_c1s1_000143_01.jpg
    0001 为 pid, c1 为 camid。
    """
    # 修复 1: 将此内置目录设为空字符串
    # 这样可以完全由配置文件的 ROOT_DIR 决定数据集路径，避免路径重复或缺失
    dataset_dir = ''

    def __init__(self, root='/data', verbose=True, **kwargs):
        super(Market1501, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        self._check_before_run()

        # 解析训练、查询和底库数据
        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        # 修复 2: 必须先完成属性赋值，再调用打印函数
        # 否则 show_train() 会因为找不到 self.train 属性而崩溃
        self.train = train
        self.query = query
        self.gallery = gallery

        if verbose:
            self.logger.info("=> Market1501 已加载")
            self.show_train()
            self.show_test()

        # 计算数据集统计信息
        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.parse_dataset(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.parse_dataset(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.parse_dataset(self.gallery)

    def _check_before_run(self):
        """检查数据集目录是否存在"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError(f"'{self.dataset_dir}' 不存在")
        if not osp.exists(self.train_dir):
            raise RuntimeError(f"'{self.train_dir}' 不存在")
        if not osp.exists(self.query_dir):
            raise RuntimeError(f"'{self.query_dir}' 不存在")
        if not osp.exists(self.gallery_dir):
            raise RuntimeError(f"'{self.gallery_dir}' 不存在")

    def _process_dir(self, dir_path, relabel=False):
        """
        读取目录下所有图片并解析 PID 和 CamID。
        relabel: 是否对 PID 进行重映射 (训练集必须为 True)
        """
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            match = pattern.search(osp.basename(img_path))
            if not match: continue
            pid, _ = map(int, match.groups())
            if pid == -1: continue  # 忽略垃圾样本
            pid_container.add(pid)

        # 建立 PID 到 0~N-1 的映射字典
        pid2label = {pid: label for label, pid in enumerate(sorted(pid_container))}

        dataset = []
        for img_path in img_paths:
            match = pattern.search(osp.basename(img_path))
            if not match: continue
            pid, camid = map(int, match.groups())
            if pid == -1: continue  # 忽略垃圾样本

            # 摄像头 ID 从 0 开始计算，Market1501 原始是 1~6
            camid -= 1

            if relabel:
                pid = pid2label[pid]

            dataset.append((img_path, pid, camid))

        return dataset