# models/deit_aux.py
import torch
import torch.nn as nn
from functools import partial

from timm.models.vision_transformer import VisionTransformer, checkpoint_filter_fn
from timm.models.registry import register_model
from ._builder import build_model_with_cfg
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import torch.nn.functional as F

# ==========================================
# 1. 默认配置
# ==========================================
def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }

# 指向官方 DeiT 权重，方便你偶尔需要用 pretrained=True 做测试
default_cfgs = {
    'deit_aux_tiny_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth'),
    'deit_aux_small_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth'),
    'deit_aux_base_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth'),
}

# ==========================================
# 2. 模型定义
# ==========================================

class DeiTAux(VisionTransformer):
    """
    DeiT with Hybrid Strategy:
    Supports both 'Late Fusion' and 'Dual Branch' via a forward argument.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # 1. Canny 编码器
        self.canny_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, self.embed_dim)) # 映射到与 ViT 相同的维度
        
        # 2. 定义两个不同的 Head 以支持两种模式
        
        # 模式 A (Dual Branch): 只有 Canny 特征通过这里
        self.aux_head = nn.Linear(self.embed_dim, self.num_classes)
        
        # 模式 B (Fusion): RGB(768) + Canny(768) = 1536 维特征通过这里
        self.fusion_head = nn.Linear(self.embed_dim * 2, self.num_classes)

        # 初始化权重
        for m in self.canny_encoder.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        nn.init.trunc_normal_(self.aux_head.weight, std=0.02)
        nn.init.trunc_normal_(self.fusion_head.weight, std=0.02)

    def forward(self, x_rgb, x_edge=None, attn_mask=None, use_fusion=False):
        """
        Args:
            use_fusion (bool): 
                - True: 执行 Late Fusion，返回融合后的 logits (单个 Tensor)
                - False: 执行 Dual Branch，返回 (main_logits, aux_logits)
        """
        
        # ---------------------------------------------------
        # 1. Attention Mask 处理 (用于 Guide RGB分支)
        # ---------------------------------------------------
        if attn_mask is not None:
            grid_size = self.patch_embed.grid_size 
            attn_mask = F.interpolate(attn_mask, size=grid_size, mode='nearest')
            attn_mask = attn_mask.flatten(2).squeeze(1)
            B = attn_mask.shape[0]
            cls_token_mask = torch.ones(B, 1, device=attn_mask.device)
            attn_mask = torch.cat([cls_token_mask, attn_mask], dim=1)
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(1)
            attn_mask = (1.0 - attn_mask) * -100.0 

        # ---------------------------------------------------
        # 2. 主分支特征提取 (RGB)
        # ---------------------------------------------------
        x_rgb_feat = super().forward_features(x_rgb, attn_mask=attn_mask) 
        x_rgb_feat = self.forward_head(x_rgb_feat, pre_logits=True) # [B, 768]

        # 如果没有 Canny 图，直接回退到最原始的 RGB 预测
        if x_edge is None:
            return self.head(x_rgb_feat)

        # ---------------------------------------------------
        # 3. 辅助分支特征提取 (Canny)
        # ---------------------------------------------------
        x_edge_feat = self.canny_encoder(x_edge) # [B, 768]

        # ---------------------------------------------------
        # 4. 模式选择 (Switch)
        # ---------------------------------------------------
        if use_fusion:
            # === 模式 A: Late Fusion (融合) ===
            # 将两者拼接，通过 fusion_head 预测
            x_fused = torch.cat([x_rgb_feat, x_edge_feat], dim=1) # [B, 1536]
            logits_fused = self.fusion_head(x_fused)
            return logits_fused
            
        else:
            # === 模式 B: Dual Branch (独立) ===
            # RGB 走原来的 head, Canny 走 aux_head
            logits_main = self.head(x_rgb_feat)
            logits_aux = self.aux_head(x_edge_feat)
            
            # 训练时返回两个，方便算两个 Loss
            # 推理时你可以选择返回 main，或者返回 (main + aux)/2
            return logits_main, logits_aux

# ==========================================
# 3. 注册部分
# ==========================================

@register_model
def deit_aux_tiny_patch16_224(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    return build_model_with_cfg(
        DeiTAux, 
        'deit_aux_tiny_patch16_224', 
        pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        **model_kwargs
    )

@register_model
def deit_aux_small_patch16_224(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    return build_model_with_cfg(
        DeiTAux, 
        'deit_aux_small_patch16_224', 
        pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        **model_kwargs
    )

@register_model
def deit_aux_base_patch16_224(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    return build_model_with_cfg(
        DeiTAux, 
        'deit_aux_base_patch16_224', 
        pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        **model_kwargs
    )