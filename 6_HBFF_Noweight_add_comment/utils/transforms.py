import random
import math
from PIL import Image, ImageOps


class LSEA(object):
    """
    Local Grayscale (LSEA): 随机将图像的一个矩形区域变为灰度图。
    这能防止模型过度依赖颜色特征，强迫其学习纹理和结构。
    """
    def __init__(self, p=0.5, sl=0.02, sh=0.4, r1=0.3):
        self.p = p          # 执行概率
        self.sl = sl        # 最小面积比例
        self.sh = sh        # 最大面积比例
        self.r1 = r1        # 宽高比范围 (r1 ~ 1/r1)

    def __call__(self, img):
        if random.random() < self.p:
            img_w, img_h = img.size
            area = img_w * img_h

            for _ in range(100): # 尝试 100 次生成合法的框
                target_area = random.uniform(self.sl, self.sh) * area
                aspect_ratio = random.uniform(self.r1, 1 / self.r1)

                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))

                if w < img_w and h < img_h:
                    x1 = random.randint(0, img_w - w)
                    y1 = random.randint(0, img_h - h)

                    # 1. 裁剪出该区域
                    patch = img.crop((x1, y1, x1 + w, y1 + h))
                    # 2. 转为灰度图 (L模式) 再转回 RGB (保持维度一致)
                    patch_gray = ImageOps.grayscale(patch).convert('RGB')
                    # 3. 贴回去
                    img.paste(patch_gray, (x1, y1, x1 + w, y1 + h))
                    return img
        return img


class RandomErasing(object):
    """
    [新增] Random Erasing Data Augmentation (REA)
    论文: 'Random Erasing Data Augmentation' by Zhong et al.
    作用: 随机遮挡图像的矩形区域，模拟遮挡情况，强制模型学习全局特征。
    注意: 该操作应作用于 Tensor (C, H, W)，即在 ToTensor 和 Normalize 之后使用。
    """
    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.485, 0.456, 0.406]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):
        # img: Tensor [C, H, W]
        if random.uniform(0, 1) >= self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img