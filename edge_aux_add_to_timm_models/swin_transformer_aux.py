# models/swin_transformer_aux.py
import torch
import torch.nn as nn
from timm.models.swin_transformer import SwinTransformer, checkpoint_filter_fn
from timm.models.registry import register_model
from timm.models._builder import build_model_with_cfg
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

# ==========================================
# 1. 默认配置
# ==========================================
def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head.fc', # 注意这里是 head.fc
        **kwargs
    }

default_cfgs = {
    'swin_aux_tiny_patch4_window7_224': _cfg(
        url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'),
    'swin_aux_small_patch4_window7_224': _cfg(
        url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth'),
    'swin_aux_base_patch4_window7_224': _cfg(
        url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224.pth'),
}

# ==========================================
# 2. 模型定义
# ==========================================
class SwinTransformerAux(SwinTransformer):
    """
    Swin Transformer with Hybrid Strategy
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Swin 的特征维度在 self.num_features
        feature_dim = self.num_features
        
        # 1. Canny 编码器
        self.canny_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, feature_dim) 
        )
        
        # 2. Heads
        self.aux_head = nn.Linear(feature_dim, self.num_classes)
        self.fusion_head = nn.Linear(feature_dim * 2, self.num_classes)

        # Init
        for m in self.canny_encoder.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        nn.init.trunc_normal_(self.aux_head.weight, std=0.02)
        nn.init.trunc_normal_(self.fusion_head.weight, std=0.02)

    def forward(self, x_rgb, x_edge=None, use_fusion=False):
        # ---------------------------------------------------
        # 1. 主分支 (RGB)
        # ---------------------------------------------------
        # forward_features 返回 [B, H, W, C]
        x_rgb_feat = super().forward_features(x_rgb) 
        
        # forward_head(pre_logits=True) 执行 Pooling -> Flatten
        # 返回 [B, C] (特征向量)
        x_rgb_feat = self.forward_head(x_rgb_feat, pre_logits=True)

        # [关键修改 1] 如果没有 edge，使用 self.head.fc 而不是 self.head
        # 因为 x_rgb_feat 已经是池化过的了，不能再进 self.head 重复池化
        if x_edge is None:
            return self.head.fc(x_rgb_feat)

        # ---------------------------------------------------
        # 2. 辅助分支 (Canny)
        # ---------------------------------------------------
        x_edge_feat = self.canny_encoder(x_edge) # [B, C]

        # ---------------------------------------------------
        # 3. 模式选择
        # ---------------------------------------------------
        if use_fusion:
            # Late Fusion
            x_fused = torch.cat([x_rgb_feat, x_edge_feat], dim=1) 
            logits_fused = self.fusion_head(x_fused)
            return logits_fused
            
        else:
            # Dual Branch
            # [关键修改 2] 使用 self.head.fc (全连接层)
            logits_main = self.head.fc(x_rgb_feat)
            logits_aux = self.aux_head(x_edge_feat)
            
            return logits_main, logits_aux

# ==========================================
# 3. 注册部分
# ==========================================
def _create_swin_transformer_aux(variant, pretrained=False, **kwargs):
    # Swin 需要 out_indices
    default_out_indices = tuple(i for i, _ in enumerate(kwargs.get('depths', (1, 1, 3, 1))))
    out_indices = kwargs.pop('out_indices', default_out_indices)

    return build_model_with_cfg(
        SwinTransformerAux, 
        variant, 
        pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        feature_cfg=dict(flatten_sequential=True, out_indices=out_indices),
        **kwargs
    )

@register_model
def swin_aux_tiny_patch4_window7_224(pretrained=False, **kwargs):
    model_args = dict(patch_size=4, window_size=7, embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24))
    return _create_swin_transformer_aux(
        'swin_aux_tiny_patch4_window7_224', pretrained=pretrained, **dict(model_args, **kwargs))

@register_model
def swin_aux_small_patch4_window7_224(pretrained=False, **kwargs):
    model_args = dict(patch_size=4, window_size=7, embed_dim=96, depths=(2, 2, 18, 2), num_heads=(3, 6, 12, 24))
    return _create_swin_transformer_aux(
        'swin_aux_small_patch4_window7_224', pretrained=pretrained, **dict(model_args, **kwargs))

@register_model
def swin_aux_base_patch4_window7_224(pretrained=False, **kwargs):
    model_args = dict(patch_size=4, window_size=7, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32))
    return _create_swin_transformer_aux(
        'swin_aux_base_patch4_window7_224', pretrained=pretrained, **dict(model_args, **kwargs))