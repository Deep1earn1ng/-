import torch
import torch.nn as nn
from timm.models.vision_transformer import Block, PatchEmbed
from torch.nn import functional as F


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


class SaliencyBackgroundSuppression(nn.Module):
    """
    [重构模块] 鲁棒性增强的显著性背景抑制模块 (SBSM v2)
    优化点：
    1. 自适应权重调节：引入可学习的 alpha 和 beta。
    2. 注意力平滑处理：引入温度系数 tau。
    3. 多头注意力一致性：在模块内部模拟多头投影计算。
    """

    def __init__(self, embed_dim, num_heads=8, init_tau=1.0):
        """
        修改: init_tau 默认值从 0.1 改为 1.0，避免 sigmoid 梯度消失。
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # 1. [优化] 温度系数：用于平滑 sigmoid 前的响应分布
        self.tau = nn.Parameter(torch.ones(1) * init_tau)

        # 2. [优化] 自适应权重：让模型自主决定保留比例与抑制强度
        # 初始化为 0.5 左右，对应原始 (0.5 + 0.5 * saliency)
        self.alpha = nn.Parameter(torch.ones(1) * 0.5)
        self.beta = nn.Parameter(torch.ones(1) * 0.5)

        # 3. [优化] 多头一致性投影层：增强定位精度
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        # x: [B, N, C], 其中 N = 1 (CLS) + num_patches
        B, N, C = x.shape
        cls_t = x[:, 0:1, :]  # [B, 1, C]
        pt_t = x[:, 1:, :]  # [B, N-1, C]

        # --- 多头注意力一致性计算 ---
        # 分别投影为多头特征 [B, num_heads, N, head_dim]
        q = self.q_proj(cls_t).reshape(B, 1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(pt_t).reshape(B, N - 1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # 计算多头注意力得分: [B, num_heads, 1, N-1]
        attn = (q @ k.transpose(-1, -2)) * self.scale

        # --- 注意力平滑处理 ---
        # 应用温度系数平滑分布，并在头维度取平均，增强掩码的鲁棒性
        attn = attn / (torch.exp(self.tau) + 1e-6)
        saliency_mask = torch.sigmoid(attn).mean(dim=1)  # [B, 1, N-1]
        saliency_mask = saliency_mask.transpose(-1, -2)  # [B, N-1, 1]

        # --- 自适应软抑制 ---
        # 逻辑：pt_t_suppressed = alpha * pt_t + beta * (pt_t * saliency_mask)
        # alpha 控制原始特征流，beta 控制显著性引导特征流
        pt_t_suppressed = self.alpha * pt_t + self.beta * (pt_t * saliency_mask)

        # 4. 重新拼接
        x = torch.cat([cls_t, pt_t_suppressed], dim=1)
        return x


class VitHBF(nn.Module):
    def __init__(self, img_size=(384, 128), patch_size=16, in_chans=3, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, norm_layer=nn.LayerNorm, camera_num=0, view_num=0, cfg=None):
        super().__init__()
        self.cfg = cfg
        self.num_features = self.embed_dim = embed_dim

        # 1. Patch Embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        # 2. Learnable Tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # 3. [修改] SIE (Side Information Embedding) - 回归标准加法嵌入
        # 移除了复杂的 Conditional LayerNorm 和 Gamma/Beta 调制机制
        self.use_sie_camera = cfg.MODEL.SIE_CAMERA
        self.use_sie_view = cfg.MODEL.SIE_VIEW

        if self.use_sie_camera:
            # 仅保留一个嵌入参数，用于加法
            self.sie_cam_embed = nn.Parameter(torch.zeros(camera_num, 1, embed_dim))
            nn.init.trunc_normal_(self.sie_cam_embed, std=.02)

        if self.use_sie_view:
            self.sie_view_embed = nn.Parameter(torch.zeros(view_num, 1, embed_dim))
            nn.init.trunc_normal_(self.sie_view_embed, std=.02)

        # 4. Transformer Blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                  proj_drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)

        # 5. SBSM 实例化
        self.sbsm = SaliencyBackgroundSuppression(embed_dim, num_heads=num_heads, init_tau=1.0)

        # 6. JPM 参数
        self.jpm = cfg.MODEL.JPM
        if self.jpm:
            self.shift_num = cfg.MODEL.SHIFT_NUM
            self.shuffle_group = cfg.MODEL.SHUFFLE_GROUP

        # 初始化权重
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        self.apply(weights_init_kaiming)

    def forward_features(self, x, cam_id=None, view_id=None):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # 先加入位置编码 (Spatial Information)
        x = x + self.pos_embed

        # [修改] SIE 注入：回归标准加法逻辑
        # x = x + sie_embed * coefficient
        if self.use_sie_camera and cam_id is not None:
            # sie_cam_embed: [camera_num, 1, C] -> [B, 1, C] via indexing
            x = x + self.sie_cam_embed[cam_id] * self.cfg.MODEL.SIE_COE

        elif self.use_sie_view and view_id is not None:
            x = x + self.sie_view_embed[view_id] * self.cfg.MODEL.SIE_COE

        x = self.pos_drop(x)

        # 前 3 层 Transformer
        for i in range(3):
            x = self.blocks[i](x)

        # 调用优化后的 SBSM
        x = self.sbsm(x)

        # 后续层 Transformer
        for i in range(3, len(self.blocks) - 1):
            x = self.blocks[i](x)

        if self.jpm and self.training:
            return self.forward_jpm(x)

        x = self.blocks[-1](x)
        x = self.norm(x)
        return x[:, 0]

    def forward_jpm(self, x):
        B, N, C = x.shape
        cls_token = x[:, 0:1, :]
        patch_tokens = x[:, 1:, :]

        global_x = self.blocks[-1](x)
        global_feat = self.norm(global_x)[:, 0]

        if self.shift_num > 0:
            patch_tokens = torch.roll(patch_tokens, shifts=self.shift_num, dims=1)

        num_patches = N - 1
        group_size = num_patches // self.shuffle_group
        shuffle_idx = torch.randperm(self.shuffle_group, device=x.device)

        patch_groups = [patch_tokens[:, i * group_size: (i + 1) * group_size, :] for i in range(self.shuffle_group)]
        shuffled_patches = []
        for idx in shuffle_idx:
            shuffled_patches.append(patch_groups[idx])
        patch_tokens = torch.cat(shuffled_patches, dim=1)

        part_size = num_patches // 4
        local_feats = []
        for i in range(4):
            part_p = patch_tokens[:, i * part_size: (i + 1) * part_size, :]
            local_x = torch.cat((cls_token, part_p), dim=1)
            local_x = self.blocks[-1](local_x)
            local_feats.append(self.norm(local_x)[:, 0])

        return [global_feat] + local_feats

    def forward(self, x, cam_label=None, view_label=None, cam_id=None, view_id=None):
        if cam_id is None and cam_label is not None:
            cam_id = cam_label
        if view_id is None and view_label is not None:
            view_id = view_label

        x = self.forward_features(x, cam_id, view_id)
        return x


def vit_base_patch16_224_TransReID(cfg, **kwargs):
    model = VitHBF(
        img_size=cfg.INPUT.SIZE_TRAIN, patch_size=16, embed_dim=768, depth=12, num_heads=12,
        mlp_ratio=4, qkv_bias=True, drop_path_rate=cfg.MODEL.DROP_PATH, cfg=cfg, **kwargs)
    return model