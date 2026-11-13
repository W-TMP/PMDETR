"""
潜能图（Potential Map）模块：从红外特征预测空间先验，用于编码器增强。
- PotentialMapHead: 轻量卷积头，输出单通道 logits -> sigmoid 得到 G_ir ∈ [0,1]
- 监督：由标注框生成二值 mask，Focal Loss
- 三种编码器增强：Pre-Modulation / Post-Attention Gating / Attention Score Bias
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core import register


def box_xyxy_to_cxcywh_norm(boxes, h, w):
    """将 [x1,y1,x2,y2] 像素坐标转为 [cx,cy,w,h] 归一化 (0~1)，或保持原格式。"""
    if boxes.shape[-1] == 4:
        x1, y1, x2, y2 = boxes.unbind(-1)
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        wb = x2 - x1
        hb = y2 - y1
        if boxes.max() > 1.1:  # 像素坐标
            cx, cy = cx / w, cy / h
            wb, hb = wb / w, hb / h
        return torch.stack([cx, cy, wb, hb], dim=-1)
    return boxes


def build_potential_target_mask(targets, feat_h, feat_w, stride, device, orig_size=None):
    """
    根据 targets 中的 boxes 在特征图上生成二值 mask Y ∈ {0,1}^{B,1,H,W}。
    targets: list of dict, 每个 dict 含 'boxes' [N,4]，格式为 [cx,cy,w,h] 归一化 或 [x1,y1,x2,y2] 像素。
    orig_size: (H_orig, W_orig) 当 boxes 为像素坐标时用于归一化；若为 None 且 boxes 已归一化则用 feat_h*stride 等。
    """
    B = len(targets)
    Y = torch.zeros(B, 1, feat_h, feat_w, dtype=torch.float32, device=device)
    if orig_size is None:
        orig_h, orig_w = feat_h * stride, feat_w * stride
    else:
        orig_h, orig_w = orig_size

    for b in range(B):
        boxes = targets[b].get('boxes')
        if boxes is None or len(boxes) == 0:
            continue
        boxes = boxes.to(device)
        # 归一化 -> 特征图坐标 (0~feat_w-1, 0~feat_h-1)
        if boxes.shape[-1] == 4:
            cx, cy, wb, hb = boxes.unbind(-1)
            if boxes.max() <= 1.01:  # 已归一化 [cx,cy,w,h] in [0,1]
                cx = cx * feat_w
                cy = cy * feat_h
                wb = wb * feat_w
                hb = hb * feat_h
            else:
                # 像素坐标，按 stride 映射到特征图
                cx = (cx / stride).clamp(0, feat_w - 1)
                cy = (cy / stride).clamp(0, feat_h - 1)
                wb = (wb / stride).clamp(min=1)
                hb = (hb / stride).clamp(min=1)
            x1 = (cx - wb / 2).long().clamp(0, feat_w - 1)
            x2 = (cx + wb / 2).long().clamp(0, feat_w - 1)
            y1 = (cy - hb / 2).long().clamp(0, feat_h - 1)
            y2 = (cy + hb / 2).long().clamp(0, feat_h - 1)
            for i in range(boxes.shape[0]):
                Y[b, 0, y1[i]:y2[i] + 1, x1[i]:x2[i] + 1] = 1.0
    return Y


@register
class PotentialMapHead(nn.Module):
    """
    从红外特征 F_ir [B, C, H, W] 预测潜能图 G_ir ∈ [0,1]^{B,1,H,W}。
    forward 返回 (G_ir, logits)，logits 用于 Focal Loss 监督。
    """

    def __init__(self, in_channels, mid_channels=64, act='relu'):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv_out = nn.Conv2d(mid_channels, 1, 1)
        self.act = nn.ReLU(inplace=True) if act == 'relu' else nn.SiLU(inplace=True)

    def forward(self, f_ir):
        """
        f_ir: [B, C, H, W]
        Returns:
            G_ir: [B, 1, H, W] sigmoid 后的潜能图
            logits: [B, 1, H, W] 用于监督的 logits
        """
        x = self.act(self.bn1(self.conv1(f_ir)))
        x = self.act(self.bn2(self.conv2(x)))
        logits = self.conv_out(x)
        G_ir = torch.sigmoid(logits)
        return G_ir, logits


# --------------- 三种编码器增强方式 ---------------

def pre_encoder_spatial_modulation(feat, G_ir, alpha):
    """
    编码前特征调制: F̃ = F ⊙ (1 + α·G_ir)
    feat: [B, C, H, W], G_ir: [B, 1, H, W], alpha: 可学习标量或 float
    """
    if isinstance(alpha, torch.Tensor):
        alpha = alpha.view(1, -1, 1, 1)
    return feat * (1.0 + alpha * G_ir)


def post_attention_gating(attn_weights, G_ir_flat, gamma):
    """
    Attention 后门控: Ã_ij = A_ij ⊙ (1 + γ·G_ir(j))
    attn_weights: [B, nH, L, L] 或 [B*nH, L, L]; G_ir_flat: [B, L] 或 [L]; gamma: 标量
    对 key/value 位置 j 施加 G_ir(j)，即对 attn 的最后一维（key 维）乘 (1+γ*G_ir)。
    """
    # attn_weights: (B, nH, L, L)，最后一维是 key 索引 j
    # G_ir_flat: (B, L) 与 j 对应
    if G_ir_flat.dim() == 1:
        G_ir_flat = G_ir_flat.unsqueeze(0)
    B, L = G_ir_flat.shape
    gate = (1.0 + gamma * G_ir_flat).unsqueeze(1)  # [B, 1, L]
    while gate.dim() < attn_weights.dim():
        gate = gate.unsqueeze(2)
    return attn_weights * gate


def attention_score_bias(attn_scores, G_ir_flat, beta):
    """
    Softmax 前在 logits 上加 bias: S_ij = S_ij + β·G_ir(j)
    attn_scores: [B, nH, L, L], G_ir_flat: [B, L], beta: 标量
    """
    if G_ir_flat.dim() == 1:
        G_ir_flat = G_ir_flat.unsqueeze(0)
    # 对 key 维 j 加 bias
    bias = (beta * G_ir_flat).unsqueeze(1)  # [B, 1, L]
    while bias.dim() < attn_scores.dim():
        bias = bias.unsqueeze(2)
    return attn_scores + bias


try:
    import torchvision
except ImportError:
    torchvision = None


def potential_map_focal_loss(logits, target_mask, alpha=0.25, gamma=2.0, reduction='mean'):
    """
    潜能图二值监督的 Focal Loss。
    logits: [B, 1, H, W], target_mask: [B, 1, H, W] ∈ {0, 1}
    """
    if torchvision is not None:
        return torchvision.ops.sigmoid_focal_loss(
            logits, target_mask, alpha=alpha, gamma=gamma, reduction=reduction
        )
    return F.binary_cross_entropy_with_logits(logits, target_mask, reduction=reduction)
