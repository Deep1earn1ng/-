from PIL import Image, ImageFile
from torch.utils.data import Dataset
import os.path as osp
import logging

# 防止读取损坏图片报错
ImageFile.LOAD_TRUNCATED_IMAGES = True


class BaseDataset(object):
    """
    Base class of reid dataset
    """

    def __init__(self):
        # Initialize Logger, using 'reid.dataset' to inherit settings from the main Logger
        self.logger = logging.getLogger("reid.dataset")

    def parse_dataset(self, data):
        """
        Parse dataset list to count PIDs, images, and cameras.
        Args:
            data (list): List of tuples (img_path, pid, camid)
        Returns:
            num_pids (int): Number of identities
            num_imgs (int): Number of images
            num_cams (int): Number of cameras
        """
        pids = set()
        cams = set()
        for _, pid, camid in data:
            pids.add(pid)
            cams.add(camid)
        return len(pids), len(data), len(cams)

    def show_train(self):
        """Print training set statistics"""
        num_train_pids, num_train_imgs, num_train_cams = self.parse_dataset(self.train)
        self.logger.info("=> Loaded Train: {} images, {} pids, {} cameras".format(
            num_train_imgs, num_train_pids, num_train_cams))

    def show_test(self):
        """Print test set (Query/Gallery) statistics"""
        num_query_pids, num_query_imgs, num_query_cams = self.parse_dataset(self.query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams = self.parse_dataset(self.gallery)

        self.logger.info("=> Loaded Query: {} images, {} pids, {} cameras".format(
            num_query_imgs, num_query_pids, num_query_cams))
        self.logger.info("=> Loaded Gallery: {} images, {} pids, {} cameras".format(
            num_gallery_imgs, num_gallery_pids, num_gallery_cams))


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class ImageDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        # Return format: (Image, PID, CamID, Path)
        # Note: collate_fn in loader.py likely expects this tuple structure
        return img, pid, camid, img_path