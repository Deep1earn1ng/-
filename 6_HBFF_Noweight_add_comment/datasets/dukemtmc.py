import glob
import re
import os.path as osp
from .bases import BaseDataset


class DukeMTMC(BaseDataset):
    """
    DukeMTMC-reID 数据集解析类。
    文件名格式示例: 0001_c1_f0053067.jpg
    其中:
      - 0001 为 pid (Person ID)
      - c1 为 camid (Camera ID, 原始范围 1-8)
    """
    # 保持为空，由配置文件中的 ROOT_DIR 决定完整路径
    dataset_dir = ''

    def __init__(self, root='/data', verbose=True, **kwargs):
        super(DukeMTMC, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        self._check_before_run()

        # 解析训练、查询和底库数据
        # 训练集需要重映射 PID (0 ~ N-1)，测试集不需要
        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        self.train = train
        self.query = query
        self.gallery = gallery

        if verbose:
            self.logger.info("=> DukeMTMC-reID 已加载")
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
        relabel: 是否对 PID 进行重映射
        """
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        # Duke 格式: 0001_c1_f0053067.jpg -> 匹配 ([-\d]+) 和 _c(\d)
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            match = pattern.search(osp.basename(img_path))
            if not match:
                continue
            pid, _ = map(int, match.groups())
            if pid == -1:
                continue  # 忽略垃圾样本(如果有)
            pid_container.add(pid)

        # 建立 PID 到 0~N-1 的映射字典
        pid2label = {pid: label for label, pid in enumerate(sorted(pid_container))}

        dataset = []
        for img_path in img_paths:
            match = pattern.search(osp.basename(img_path))
            if not match:
                continue
            pid, camid = map(int, match.groups())
            if pid == -1:
                continue  # 忽略垃圾样本

            # 摄像头 ID 从 0 开始计算，Duke 原始是 1~8 -> 映射为 0~7
            camid -= 1

            if relabel:
                pid = pid2label[pid]

            dataset.append((img_path, pid, camid))

        return dataset