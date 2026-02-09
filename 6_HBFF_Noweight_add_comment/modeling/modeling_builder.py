import torch
import torch.nn as nn
from .backbones.vit_hbf import vit_base_patch16_224_TransReID


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.weight is not None:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


class build_model(nn.Module):
    def __init__(self, cfg, num_classes, camera_num, view_num):
        """
        修改说明:
        1. 增加了 camera_num 和 view_num 参数。
        2. [优化] 为 JPM 分支增加了独立的 BNNeck (bottleneck_1-4)，确保特征分布解耦。
        """
        super(build_model, self).__init__()
        self.cfg = cfg
        self.num_classes = num_classes

        # 1. 实例化 Backbone
        self.base = vit_base_patch16_224_TransReID(cfg, camera_num=camera_num, view_num=view_num)
        self.in_planes = self.base.embed_dim

        # 2. 构建全局 BNNeck (用于全局特征)
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        # 3. 构建全局分类 Head
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        # 4. [修改] 如果开启 JPM，构建独立的局部 BNNeck 和 分类器
        if cfg.MODEL.JPM:
            # 分支 1
            self.bottleneck_1 = nn.BatchNorm1d(self.in_planes)
            self.bottleneck_1.bias.requires_grad_(False)
            self.bottleneck_1.apply(weights_init_kaiming)
            self.classifier_1 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_1.apply(weights_init_classifier)

            # 分支 2
            self.bottleneck_2 = nn.BatchNorm1d(self.in_planes)
            self.bottleneck_2.bias.requires_grad_(False)
            self.bottleneck_2.apply(weights_init_kaiming)
            self.classifier_2 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_2.apply(weights_init_classifier)

            # 分支 3
            self.bottleneck_3 = nn.BatchNorm1d(self.in_planes)
            self.bottleneck_3.bias.requires_grad_(False)
            self.bottleneck_3.apply(weights_init_kaiming)
            self.classifier_3 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_3.apply(weights_init_classifier)

            # 分支 4
            self.bottleneck_4 = nn.BatchNorm1d(self.in_planes)
            self.bottleneck_4.bias.requires_grad_(False)
            self.bottleneck_4.apply(weights_init_kaiming)
            self.classifier_4 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_4.apply(weights_init_classifier)

        # 5. 加载预训练权重
        if cfg.MODEL.PRETRAIN_CHOICE == 'imagenet':
            self.load_pretrain_with_interpolation(cfg.MODEL.PRETRAIN_PATH)

    def load_pretrain_with_interpolation(self, model_path):
        """处理 ViT 位置编码插值并加载权重"""
        state_dict = torch.load(model_path, map_location='cpu')

        if 'pos_embed' in state_dict:
            pos_embed_checkpoint = state_dict['pos_embed']
            embedding_size = pos_embed_checkpoint.shape[-1]
            num_patches = self.base.patch_embed.num_patches
            num_extra_tokens = self.base.pos_embed.shape[-2] - num_patches

            orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
            new_size_h = self.cfg.INPUT.SIZE_TRAIN[0] // 16
            new_size_w = self.cfg.INPUT.SIZE_TRAIN[1] // 16

            if orig_size != new_size_h or orig_size != new_size_w:
                extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
                pos_tokens = torch.nn.functional.interpolate(
                    pos_tokens, size=(new_size_h, new_size_w), mode='bicubic', align_corners=False)
                pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
                new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                state_dict['pos_embed'] = new_pos_embed

        self.base.load_state_dict(state_dict, strict=False)

    def forward(self, x, label=None, cam_label=None, view_label=None):
        """
        前向传播：全局特征与局部多分支特征解耦处理
        """
        # 1. 提取骨干特征
        features = self.base(x, cam_id=cam_label, view_id=view_label)

        if self.training:
            if isinstance(features, list):
                # JPM 多分支训练模式
                global_feat = features[0]
                # 全局分支使用全局 BNNeck
                score = self.classifier(self.bottleneck(global_feat))

                # [修改] 4 个局部特征分别通过其专属的 BNNeck，避免分布干扰
                s1 = self.classifier_1(self.bottleneck_1(features[1]))
                s2 = self.classifier_2(self.bottleneck_2(features[2]))
                s3 = self.classifier_3(self.bottleneck_3(features[3]))
                s4 = self.classifier_4(self.bottleneck_4(features[4]))

                return [score, s1, s2, s3, s4], global_feat
            else:
                # 普通单分支训练模式
                global_feat = features
                score = self.classifier(self.bottleneck(global_feat))
                return score, global_feat
        else:
            # 推理阶段：通常仅使用归一化后的全局特征
            feat = features[0] if isinstance(features, list) else features
            return self.bottleneck(feat)


def make_model(cfg, num_classes, camera_num, view_num):
    model = build_model(cfg, num_classes, camera_num, view_num)
    return model