import torch
import torch.nn as nn
import torch.nn.functional as F
from src.core import register

from .hybrid_encoder import ConvNormLayer, CSPRepLayer


# @register
# class FusionWrapper(nn.Module):
#     """跨层融合：与 hybrid_encoder 相同的 top-down FPN + bottom-up PAN。
#     输入: feats [P3, P4, P5]，可选 potentials。
#     输出: [P3', P4', P5']。
#     """
#     def __init__(self, channels=256, num_levels=3, expansion=1.0, depth_mult=1.0, act='silu',
#                  enabled=True, use_potentials=True):
#         """Args:
#             channels: hidden_dim (与 encoder 一致)
#             num_levels: 特征层数，默认 3 (P3,P4,P5)
#             expansion: CSPRepLayer 的 expansion
#             depth_mult: CSPRepLayer 的 block 数量系数
#             act: 激活
#             enabled: 若 False 则恒等返回
#             use_potentials: 若 False 则忽略 forward 的 potentials
#         """
#         super().__init__()
#         self.channels = channels
#         self.num_levels = num_levels
#         self.enabled = bool(enabled)
#         self.use_potentials = bool(use_potentials)

#         # top-down FPN: lateral_convs + fpn_blocks (与 HybridEncoder 一致)
#         self.lateral_convs = nn.ModuleList()
#         self.fpn_blocks = nn.ModuleList()
#         for _ in range(num_levels - 1, 0, -1):
#             self.lateral_convs.append(ConvNormLayer(channels, channels, 1, 1, act=act))
#             self.fpn_blocks.append(
#                 CSPRepLayer(channels * 2, channels, round(3 * depth_mult), act=act, expansion=expansion)
#             )

#         # bottom-up PAN
#         self.downsample_convs = nn.ModuleList()
#         self.pan_blocks = nn.ModuleList()
#         for _ in range(num_levels - 1):
#             self.downsample_convs.append(
#                 ConvNormLayer(channels, channels, 3, 2, act=act)
#             )
#             self.pan_blocks.append(
#                 CSPRepLayer(channels * 2, channels, round(3 * depth_mult), act=act, expansion=expansion)
#             )

#         # potential map 投影: [B,H,W] -> [B,C,H,W]
#         self.pot_proj = nn.ModuleList([nn.Conv2d(1, channels, 1, bias=False) for _ in range(num_levels)])

#     def forward(self, feats, potentials=None):
#         # feats: list [P3, P4, P5]; potentials: optional list of [B,H,W] per level
#         if not self.enabled:
#             return feats
#         proj_feats = list(feats)
#         if self.use_potentials and potentials is not None:
#             try:
#                 for i in range(self.num_levels):
#                     p = potentials[i]
#                     proj_feats[i] = proj_feats[i] + self.pot_proj[i](p.unsqueeze(1))
#             except Exception:
#                 pass

#         # top-down FPN (与 HybridEncoder 一致)
#         inner_outs = [proj_feats[-1]]
#         for idx in range(self.num_levels - 1, 0, -1):
#             feat_high = inner_outs[0]
#             feat_low = proj_feats[idx - 1]
#             feat_high = self.lateral_convs[self.num_levels - 1 - idx](feat_high)
#             inner_outs[0] = feat_high
#             upsample_feat = F.interpolate(feat_high, scale_factor=2.0, mode='nearest')
#             inner_out = self.fpn_blocks[self.num_levels - 1 - idx](torch.cat([upsample_feat, feat_low], dim=1))
#             inner_outs.insert(0, inner_out)

#         # bottom-up PAN
#         outs = [inner_outs[0]]
#         for idx in range(self.num_levels - 1):
#             feat_low = outs[-1]
#             feat_high = inner_outs[idx + 1]
#             downsample_feat = self.downsample_convs[idx](feat_low)
#             out = self.pan_blocks[idx](torch.cat([downsample_feat, feat_high], dim=1))
#             outs.append(out)
#         return outs


import torch
import torch.nn as nn
import torch.nn.functional as F
from src.core import register
from .utils import get_activation


class SEBlock(nn.Module):
    def __init__(self, channels, r=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // r, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(channels // r, channels, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.fc(x)
        return x * w


class LowResContextEnhancer(nn.Module):
    """LCE: lightweight module to enhance P5 features (low-res) with depthwise branches and SE.
    Input: P5 [B,C,H,W]
    Output: P5_enh [B,C,H,W]
    """
    def __init__(self, channels=256, hidden_ratio=0.5, alpha=0.12, act='gelu'):
        super().__init__()
        hidden = max(32, int(channels * hidden_ratio))
        self.project = nn.Conv2d(channels, hidden, 1, bias=False)
        self.dw1 = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=True)
        self.dw2 = nn.Conv2d(hidden, hidden, 3, padding=2, dilation=2, groups=hidden, bias=True)
        self.restore = nn.Conv2d(hidden * 2, channels, 1, bias=False)
        self.se = SEBlock(channels, r=4)
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.act = get_activation(act)

    def forward(self, x):
        # x: [B,C,H,W]
        r = self.project(x)
        b1 = self.act(self.dw1(r))
        b2 = self.act(self.dw2(r))
        cat = torch.cat([b1, b2], dim=1)
        out = self.restore(cat)
        out = self.se(out)
        return x + self.alpha * out


class LightweightScaleAwareFusion(nn.Module):
    """LSFM: Fuse P5 into P4/P3 (top-down injection) with lightweight per-location gates.
    Inputs: feats list [P3, P4, P5]
    Outputs: new_feats [P3', P4', P5]
    """
    def __init__(self, in_channels=256, reduce_ratio=0.5, act='gelu'):
        super().__init__()
        self.in_channels = in_channels
        self.reduced = max(32, int(in_channels * reduce_ratio))
        self.p5_proj = nn.Conv2d(in_channels, self.reduced, 1, bias=False)
        self.p4_proj = nn.Conv2d(in_channels, self.reduced, 1, bias=False)
        self.p3_proj = nn.Conv2d(in_channels, self.reduced, 1, bias=False)

        self.dw = nn.Conv2d(self.reduced, self.reduced, 3, padding=1, groups=self.reduced, bias=True)
        self.fuse_conv = nn.Conv2d(self.reduced * 2, in_channels, 1, bias=False)

        self.gate4 = nn.Sequential(nn.Conv2d(self.reduced * 2, 1, 1, bias=True), nn.Sigmoid())
        self.gate3 = nn.Sequential(nn.Conv2d(self.reduced * 2, 1, 1, bias=True), nn.Sigmoid())
        self.act = get_activation(act)

    def forward(self, feats):
        # feats: [P3,P4,P5]
        p3, p4, p5 = feats
        # project
        p5r = self.p5_proj(p5)
        # up to p4
        p5_up_4 = F.interpolate(p5r, size=p4.shape[2:], mode='nearest')
        p5_up_4 = self.act(self.dw(p5_up_4))

        p4r = self.p4_proj(p4)
        cat4 = torch.cat([p4r, p5_up_4], dim=1)
        g4 = self.gate4(cat4)
        fused4 = self.fuse_conv(cat4)
        p4_out = p4 + g4 * fused4

        # up to p3 (from fused p4)
        p4r_up = F.interpolate(self.p4_proj(p4_out), size=p3.shape[2:], mode='nearest')
        p4r_up = self.act(self.dw(p4r_up))
        p3r = self.p3_proj(p3)
        cat3 = torch.cat([p3r, p4r_up], dim=1)
        g3 = self.gate3(cat3)
        fused3 = self.fuse_conv(cat3)
        p3_out = p3 + g3 * fused3

        # bottom-up fusion (single pass): propagate enhanced high-resolution context back up
        # p3 -> p4
        p3_down_4 = F.interpolate(self.p3_proj(p3_out), size=p4_out.shape[2:], mode='nearest')
        p3_down_4 = self.act(self.dw(p3_down_4))
        cat4_bu = torch.cat([self.p4_proj(p4_out), p3_down_4], dim=1)
        g4_bu = self.gate4(cat4_bu)
        fused4_bu = self.fuse_conv(cat4_bu)
        p4_out = p4_out + g4_bu * fused4_bu

        # p4 -> p5
        p4_down_5 = F.interpolate(self.p4_proj(p4_out), size=p5.shape[2:], mode='nearest')
        p4_down_5 = self.act(self.dw(p4_down_5))
        cat5_bu = torch.cat([self.p5_proj(p5), p4_down_5], dim=1)
        # reuse gate4 for simplicity (spatial shapes differ but gate conv is 1x1)
        g5_bu = self.gate4(cat5_bu)
        fused5_bu = self.fuse_conv(cat5_bu)
        p5_out = p5 + g5_bu * fused5_bu

        return [p3_out, p4_out, p5_out]


class ChannelFusion(nn.Module):
    """Lightweight channel-wise fusion that computes per-level channel gates from global pooled concat.
    Very cheap: global avg pool + small FC (1x1 conv) -> sigmoid gating per level.
    """
    def __init__(self, channels=256, reduction=8, num_levels=3):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.global_fc = nn.Sequential(
            nn.Conv2d(channels * num_levels, hidden, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, channels * num_levels, 1, bias=True),
            nn.Sigmoid()
        )
        self.num_levels = num_levels
        self.channels = channels

    def forward(self, feats):
        # feats: list of [B,C,H,W]
        B = feats[0].shape[0]
        # adapt spatial via adaptive avg pool to 1x1
        pooled = [F.adaptive_avg_pool2d(f, 1) for f in feats]
        cat = torch.cat(pooled, dim=1)  # [B, C*num_levels, 1, 1]
        gates = self.global_fc(cat)  # [B, C*num_levels, 1, 1]
        gates = gates.view(B, self.num_levels, self.channels, 1, 1)
        out = []
        for i, f in enumerate(feats):
            g = gates[:, i]
            out.append(f * g)
        return out


class SpatialEnhancer(nn.Module):
    """Lightweight spatial enhancer for the finest feature (P3).
    Uses upsampled P4/P5 context, depthwise conv and a sigmoid spatial gate to modulate P3.
    """
    def __init__(self, channels=256, reduce_ratio=0.5, alpha=0.15, act='gelu'):
        super().__init__()
        reduced = max(32, int(channels * reduce_ratio))
        self.p4_proj = nn.Conv2d(channels, reduced, 1, bias=False)
        self.p5_proj = nn.Conv2d(channels, reduced, 1, bias=False)
        self.dw = nn.Conv2d(reduced, reduced, 3, padding=1, groups=reduced, bias=True)
        self.fuse = nn.Conv2d(reduced, channels, 1, bias=False)
        self.gate = nn.Sequential(nn.Conv2d(channels, 1, 1, bias=True), nn.Sigmoid())
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.act = get_activation(act)

    def forward(self, p3, p4, p5):
        # upsample higher-level contexts to p3 spatial
        p4r = self.p4_proj(p4)
        p5r = self.p5_proj(p5)
        p4_up = F.interpolate(p4r, size=p3.shape[2:], mode='nearest')
        p5_up = F.interpolate(p5r, size=p3.shape[2:], mode='nearest')
        ctx = p4_up + p5_up
        ctx = self.act(self.dw(ctx))
        ctx = self.fuse(ctx)
        gate = self.gate(ctx)  # [B,1,H,W]
        return p3 + self.alpha * (gate * ctx)


class PostFusionRefiner(nn.Module):
    """Lightweight per-level refiner: 1x1 conv -> depthwise 3x3 -> 1x1 conv with residual scaling.
    Keeps computational cost low while improving local representations.
    """
    def __init__(self, channels=256, hidden_ratio=0.5, alpha=0.15, act='gelu'):
        super().__init__()
        hidden = max(32, int(channels * hidden_ratio))
        self.reduce = nn.Conv2d(channels, hidden, 1, bias=False)
        self.dw = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=True)
        self.restore = nn.Conv2d(hidden, channels, 1, bias=False)
        self.norm = nn.BatchNorm2d(channels)
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.act = get_activation(act)

    def forward(self, x):
        # x: [B,C,H,W]
        r = self.act(self.dw(self.reduce(x)))
        out = self.restore(r)
        out = self.norm(out + x)
        return x + self.alpha * out


@register
class FusionWrapper(nn.Module):
    """Wrapper to run LCE on P5 and LSFM for multi-scale fusion.
    Configurable via kwargs.
    """
    def __init__(self, channels=256, reduce_ratio=0.5, lce_alpha=0.12, spatial_alpha=0.15, ref_alpha=0.12,
                 enabled=True, use_potentials=True):
        """Args:
            enabled: if False, FusionWrapper acts as identity pass-through (no fusion applied)
            use_potentials: if False, ignore any `potentials` argument passed to forward even if present
        """
        super().__init__()
        self.lce = LowResContextEnhancer(channels=channels, hidden_ratio=reduce_ratio, alpha=lce_alpha)
        self.lsfm = LightweightScaleAwareFusion(in_channels=channels, reduce_ratio=reduce_ratio)
        self.chan_fuse = ChannelFusion(channels=channels, reduction=8, num_levels=3)
        self.spatial = SpatialEnhancer(channels=channels, reduce_ratio=reduce_ratio, alpha=spatial_alpha)
        self.refiners = nn.ModuleList([PostFusionRefiner(channels=channels, hidden_ratio=reduce_ratio, alpha=ref_alpha) for _ in range(3)])
        # potential map projectors: convert [B,H,W] -> [B,C,H,W]
        self.pot_proj = nn.ModuleList([nn.Conv2d(1, channels, 1, bias=False) for _ in range(3)])
        # configuration flags
        self.enabled = bool(enabled)
        self.use_potentials = bool(use_potentials)

    def forward(self, feats, potentials=None):
        # feats: list [P3,P4,P5]; potentials: optional list [P3_pot, P4_pot, P5_pot] each [B,H,W]
        # if fusion disabled, return features unchanged
        if not self.enabled:
            return feats
        p3, p4, p5 = feats
        # if potentials provided, project and add to features
        if self.use_potentials and potentials is not None:
            try:
                # potentials are list of [B,H,W]
                p3_p, p4_p, p5_p = potentials
                p3 = p3 + self.pot_proj[0](p3_p.unsqueeze(1))
                p4 = p4 + self.pot_proj[1](p4_p.unsqueeze(1))
                p5 = p5 + self.pot_proj[2](p5_p.unsqueeze(1))
            except Exception:
                # fallback: ignore potentials if unexpected shape
                pass
        # apply LCE to P5 first
        p5_enh = self.lce(p5)
        fused = self.lsfm([p3, p4, p5_enh])
        # apply light channel fusion gating
        fused = self.chan_fuse(fused)
        # spatial enhance P3 using higher-level context
        p3_f, p4_f, p5_f = fused
        p3_f = self.spatial(p3_f, p4_f, p5_f)
        # per-level refinement
        out = [self.refiners[i](f) for i, f in enumerate([p3_f, p4_f, p5_f])]
        return out
