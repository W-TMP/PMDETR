# explanation: Add Hierarchically Gated Sparse Transformer Encoder module with
# potential-guided dynamic enhancement and Potential Map Guidance Loss.
# ...new file...
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from .hybrid_encoder import TransformerEncoderLayer, TransformerEncoder
from src.core import register
from .utils import get_activation


class LightweightGateNet(nn.Module):
    """Lightweight gate network to produce importance score per spatial token.
    Implemented with a 1x1 conv -> depthwise 3x3 -> pointwise -> sigmoid output.
    Returns scores in [0,1] per spatial location.
    """
    def __init__(self, in_channels, hidden=64, act='gelu'):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1, bias=True),
            get_activation(act),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=True),  # depthwise
            get_activation(act),
            nn.Conv2d(hidden, 1, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B, C, H, W]
        return self.net(x).squeeze(1)  # [B, H, W]


class PotentialGuidedEnhancer(nn.Module):
    """Potential-Guided Dynamic Enhancement module.
    Computes A_agg = DWConv3x3(GELU(Conv1x1(F))) then applies gate map G to selectively add detail.
    """
    def __init__(self, in_channels, hidden_dim=None, act='gelu'):
        super().__init__()
        hidden = hidden_dim or max(32, in_channels // 4)
        self.project = nn.Conv2d(in_channels, hidden, 1, bias=False)
        self.act = get_activation(act)
        self.dw = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=True)
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.restore = nn.Conv2d(hidden, in_channels, 1, bias=False)

    def forward(self, F_l, G_l):
        # F_l: [B, C, H, W], G_l: [B, H, W]
        A = self.project(F_l)
        A = self.act(A)
        A = self.dw(A)
        A = self.restore(A)
        # broadcast gate
        G = G_l.unsqueeze(1)
        return F_l + self.alpha * (G * A)


class HierarchicallyGatedSparseEncoder(nn.Module):
    """A hierarchical gated sparse encoder suitable for multi-scale features.

    Strategy summary:
    - For each resolution level, compute gate map g_i^l with a lightweight gate net.
    - Keep top-k tokens per layer using per-layer keep ratio rho_l; perform attention only on retained tokens.
    - During downsampling (from high-res to low-res) propagate gated tokens' features into lower res via gated injection.
    - Provide potential map outputs for supervision and for dynamic enhancement.

    本实现中，gate map 可以由 P2 层级的红外特征图统一生成：
    - 先用 ir_channels -> hidden_dim 的 1x1 Conv 将 P2-IR 投影到编码器通道；
    - 再插值到各个尺度 (P3/P4/P5) 后送入每一层的 LightweightGateNet，得到对应尺度的 G_i^l；
    - 然后用这些 G_i^l 去增强各层投影后的特征。
    """

    def __init__(self,
                 in_channels_list,
                 hidden_dim=256,
                 use_levels=(0,1,2),
                 encoder_layers_per_level=(1,1,1),
                 nhead=8,
                 dim_feedforward=1024,
                 rho=(0.6,0.8,1.0),
                 pe_temperature=10000.,
                 enc_act='gelu',
                 eval_spatial_size=None,
                 gate_sharpness=1.1,
                 local_dw_alpha=0.1,
                 learnable_local_dw_alpha=True,
                 max_keep_tokens=4096,
                 ir_channels=256):
        super().__init__()
        assert len(in_channels_list) == len(use_levels)
        self.levels = len(in_channels_list)
        self.in_channels_list = in_channels_list
        self.hidden_dim = hidden_dim
        self.use_levels = use_levels
        self.rho = rho
        self.eval_spatial_size = eval_spatial_size
        self.pe_temperature = pe_temperature
        self.gate_sharpness = gate_sharpness
        # learnable or fixed scaling for local depthwise residual
        # if learnable_local_dw_alpha is True we expose it as nn.Parameter; otherwise register as buffer (fixed)
        if learnable_local_dw_alpha:
            self.local_dw_alpha = nn.Parameter(torch.tensor(float(local_dw_alpha)))
        else:
            # register as buffer so it moves with module but is not trainable
            self.register_buffer('local_dw_alpha', torch.tensor(float(local_dw_alpha)))

        # maximum number of tokens to keep per level after top-k selection
        # Set to None to disable capping. Default tuned to avoid OOM on high-res S2.
        self.max_keep_tokens = max_keep_tokens

        # projection for each level: backbone channels -> hidden_dim (统一到 256 通道，
        # 与原始 EfficientFusionEncoder/RTDETRTransformer 的接口保持一致)
        self.input_proj = nn.ModuleList([])
        for ch in in_channels_list:
            self.input_proj.append(
                nn.Sequential(
                    nn.Conv2d(ch, hidden_dim, 1, bias=False),
                    nn.BatchNorm2d(hidden_dim)
                )
            )

        # gate nets（可接受每个尺度下由 P2-IR 插值得到的特征）
        self.gate_nets = nn.ModuleList([LightweightGateNet(hidden_dim, hidden=max(32, hidden_dim//4), act=enc_act)
                                        for _ in in_channels_list])

        # potential-guided enhancers per level
        self.enhancers = nn.ModuleList([PotentialGuidedEnhancer(hidden_dim, hidden_dim//4, act=enc_act)
                                        for _ in in_channels_list])

        # small local depthwise conv per level to add low-cost spatial smoothing/residual
        self.local_dw = nn.ModuleList([
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim, bias=True)
            for _ in range(self.levels)
        ])

        # P2 层级 IR 特征投影到 hidden_dim，用于生成各尺度 gate map
        self.ir_proj = nn.Sequential(
            nn.Conv2d(ir_channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim)
        )

        # Transformer encoders per level (operate on retained tokens only)
        base_layer = TransformerEncoderLayer(hidden_dim, nhead=nhead, dim_feedforward=dim_feedforward,
                                             dropout=0.0, activation=enc_act)
        self.encoders = nn.ModuleList([
            TransformerEncoder(copy.deepcopy(base_layer), encoder_layers_per_level[i])
            for i in range(self.levels)
        ])

        # cross-resolution gating injection MLP
        self.inject_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim*2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU()
            ) for _ in range(self.levels-1)
        ])

        # position embedding helper (compatible with existing builder)
        # stored pos_embed per level if eval spatial size provided
        if self.eval_spatial_size:
            for idx in range(self.levels):
                h = self.eval_spatial_size[1] // (2**(idx+3)) if idx+3>0 else self.eval_spatial_size[1]
                w = self.eval_spatial_size[0] // (2**(idx+3)) if idx+3>0 else self.eval_spatial_size[0]
                pos = self.build_2d_sincos_position_embedding(w, h, hidden_dim, self.pe_temperature)
                setattr(self, f'pos_embed{idx}', pos)

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.):
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        assert embed_dim % 4 == 0
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)
        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]
        # use torch.cat for compatibility
        return torch.cat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]

    def forward(self, feats, feats_ir=None):
        """feats: list of tensors [B, C_l, H_l, W_l] ordered from high->low resolution.
        feats_ir: optional P2-level IR feature map [B, C_ir, H2, W2].
                  当提供时，将用其统一生成 P3/P4/P5 的 gate map，对各层投影特征进行增强。
        returns: enhanced feats list in the same ordering and potential maps per level.
        """
        assert len(feats) == self.levels
        proj = [self.input_proj[i](feats[i]) for i in range(self.levels)]
        B = proj[0].shape[0]
        device = proj[0].device

        # compute gates (potential maps)
        if feats_ir is not None:
            # 使用 P2 层级 IR 特征生成各尺度 gate map：先投影到 hidden_dim，再按尺度插值
            ir_feat = self.ir_proj(feats_ir)  # [B, hidden_dim, H2, W2]
            potentials = []
            for i in range(self.levels):
                _, _, h_l, w_l = proj[i].shape
                ir_i = ir_feat
                if ir_i.shape[2] != h_l or ir_i.shape[3] != w_l:
                    ir_i = F.interpolate(ir_i, size=(h_l, w_l), mode='bilinear', align_corners=False)
                potentials.append(self.gate_nets[i](ir_i))  # [B, H_l, W_l]
        else:
            # 退化为用各层自身特征生成 gate map
            potentials = [self.gate_nets[i](proj[i]) for i in range(self.levels)]  # [B, H, W]

        # potential-guided enhancement（此时 potentials 已由 P2-IR 或各层自身生成）
        enhanced = [self.enhancers[i](proj[i], potentials[i]) for i in range(self.levels)]

        # hierarchical gated sparse attention per level (working in hidden_dim),
        # then project back to original channels for downstream decoder.
        processed_hidden = [None] * self.levels
        for i in range(self.levels):
            feat = enhanced[i]
            b, c, h, w = feat.shape
            num_tokens = h*w
            # flatten
            tokens = feat.flatten(2).permute(0,2,1)  # [B, N, C]
            pos_embed = None
            if self.training or not hasattr(self, f'pos_embed{i}'):
                pos_embed = self.build_2d_sincos_position_embedding(w, h, self.hidden_dim, self.pe_temperature).to(device)
            else:
                pos_embed = getattr(self, f'pos_embed{i}').to(device)
            pos = pos_embed.repeat(B,1,1)

            # gate scores per token
            g = potentials[i].flatten(1)  # [B, N]

            # apply mild sharpening to emphasize confident tokens (low-cost)
            if self.gate_sharpness != 1.0:
                # ensure numerically safe
                g = g.clamp(1e-6, 1.0 - 1e-6)
                g = g.pow(self.gate_sharpness)

            # compute top-k mask per batch element
            rho_i = self.rho[i] if i < len(self.rho) else self.rho[-1]
            k = max(1, int(num_tokens * rho_i))
            # apply a hard cap on kept tokens to avoid quadratic blow-up in attention
            if self.max_keep_tokens is not None:
                k = min(k, int(self.max_keep_tokens))

            # True sparse path: gather top-k tokens per batch, run encoder on compact [B, k, C], scatter back
            # get topk indices per batch
            topk = torch.topk(g, k, dim=1)
            topk_idx = topk.indices  # [B, k]

            # gather tokens at topk indices
            # tokens: [B, N, C], we want kept: [B, k, C]
            idx_exp = topk_idx.unsqueeze(-1).expand(-1, -1, c)  # [B, k, C]
            kept_tokens = torch.gather(tokens, dim=1, index=idx_exp)  # [B, k, C]

            # gather pos for kept
            pos_exp = topk_idx.unsqueeze(-1).expand(-1, -1, pos.shape[-1])
            pos_kept = torch.gather(pos, dim=1, index=pos_exp)

            # run encoder on kept tokens
            memory_kept = self.encoders[i](kept_tokens, pos_embed=pos_kept)

            # scatter encoded kept tokens back to full token map
            memory_full = torch.zeros_like(tokens)
            idx_exp_scatter = topk_idx.unsqueeze(-1).expand(-1, -1, c)
            memory_full.scatter_(1, idx_exp_scatter, memory_kept)

            # construct processed tokens: update only kept positions, leave others as original tokens
            tokens_updated = tokens.clone()
            tokens_updated.scatter_(1, idx_exp_scatter, memory_kept)

            # reconstruct spatial in hidden_dim space
            processed_hidden[i] = tokens_updated.permute(0,2,1).reshape(b, c, h, w).contiguous()

            # low-cost local depthwise residual to add local context (still in hidden_dim)
            processed_hidden[i] = processed_hidden[i] + self.local_dw_alpha * self.local_dw[i](processed_hidden[i])

        # cross-resolution gated injection from high-res -> low-res
        # assume levels ordered high->low, inject i -> i+1
        for i in range(self.levels-1):
            high = processed_hidden[i]
            low = processed_hidden[i+1]
            # downsample high to low spatial
            high_ds = F.interpolate(high, size=low.shape[2:], mode='nearest')
            # combine only at positions where high potential is high
            G_high = potentials[i]
            G_high_ds = F.interpolate(G_high.unsqueeze(1), size=low.shape[2:], mode='nearest').squeeze(1)  # [B,H2,W2]
            # create concat and inject via MLP
            b, c, h2, w2 = low.shape
            concat = torch.cat([low, high_ds], dim=1)  # [B, 2C, H2, W2]
            # flatten last dims
            concat_flat = concat.flatten(2).permute(0,2,1)  # [B, N2, 2C]
            # apply mlp
            injected = self.inject_mlps[i](concat_flat).permute(0,2,1).reshape(b, c, h2, w2)
            # gating beta: use G_high_ds passed through small MLP (here simple scaling)
            beta = G_high_ds.unsqueeze(1)  # [B,1,H2,W2]
            processed_hidden[i+1] = low + beta * injected

        # 输出直接保持为 hidden_dim 通道（通常为 256），
        # 这样与 EfficientFusionEncoder 一致，便于直接接入 FusionWrapper 和 RTDETRTransformer。
        processed = processed_hidden

        # store last potentials for external access
        self._last_potentials = potentials

        return processed, potentials


@register
class PotentialMapGuidanceLoss(nn.Module):
    """Focal-like loss for potential maps. Outputs scalar loss.
    Expects predictions G: [B,H,W] and masks M: [B,H,W] binary (1 foreground)
    """
    def __init__(self, gamma_pos=2.0, gamma_neg=4.0, alpha_pos=1.0, alpha_neg=1.0):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.alpha_pos = alpha_pos
        self.alpha_neg = alpha_neg

    def forward(self, G, M):
        # clamp
        eps = 1e-6
        G = G.clamp(eps, 1.0 - eps)
        pos_mask = (M > 0.5).float()
        neg_mask = (M <= 0.5).float()
        pos_loss = - self.alpha_pos * ( (1 - G) ** self.gamma_pos) * torch.log(G + eps) * pos_mask
        neg_loss = - self.alpha_neg * ( (G) ** self.gamma_neg) * torch.log(1 - G + eps) * neg_mask
        # normalize pos and neg components independently to avoid domination by abundant background
        pos_den = max(1.0, float(pos_mask.sum().item()))
        neg_den = max(1.0, float(neg_mask.sum().item()))
        loss = pos_loss.sum() / pos_den + neg_loss.sum() / neg_den
        loss = loss * 0.5
        return loss


def generate_potential_map_targets(boxes_list, image_size, feature_map_shapes, soft=True):
    """Generate soft potential masks per feature map shape.

    For each GT box we place a Gaussian-shaped heatmap centered at the box center in the feature map
    coordinate space. This produces softer supervision that encourages the model to focus near object
    centers rather than a hard filled rectangle. Returns list of float tensors [B, H_l, W_l].

    Args:
        boxes_list: list (length B) of tensors Nx4 (x1,y1,x2,y2) in absolute pixel coords.
        image_size: (W, H) of input image in pixels.
        feature_map_shapes: list of (H_l, W_l) tuples for each level (same ordering as encoder inputs).
        soft: bool, whether to create soft gaussian masks (default True). If False, falls back to binary box masks.

    Returns:
        masks: list of tensors [B, H_l, W_l] (float)
    """
    B = len(boxes_list)
    L = len(feature_map_shapes)
    W, H = image_size
    masks = []
    for l, (h_l, w_l) in enumerate(feature_map_shapes):
        mask_lvl = torch.zeros((B, h_l, w_l), dtype=torch.float32)
        # precompute coordinate grids
        ys = torch.arange(0, h_l, dtype=torch.float32)
        xs = torch.arange(0, w_l, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        for b in range(B):
            boxes = boxes_list[b]
            if boxes is None or boxes.numel() == 0:
                continue
            # convert boxes to feature map coords (absolute)
            xs1 = (boxes[:, 0] / W * w_l).clamp(0, w_l - 1)
            ys1 = (boxes[:, 1] / H * h_l).clamp(0, h_l - 1)
            xs2 = (boxes[:, 2] / W * w_l).clamp(0, w_l - 1)
            ys2 = (boxes[:, 3] / H * h_l).clamp(0, h_l - 1)
            for i in range(boxes.shape[0]):
                x1 = float(xs1[i].item())
                y1 = float(ys1[i].item())
                x2 = float(xs2[i].item())
                y2 = float(ys2[i].item())
                if x2 < x1 or y2 < y1:
                    continue
                cx = (x1 + x2) * 0.5
                cy = (y1 + y2) * 0.5
                bw = max(1.0, x2 - x1)
                bh = max(1.0, y2 - y1)
                # sigma proportional to box size (so larger boxes have wider gaussian)
                sigma = max(1.0, min(bw, bh) * 0.5)
                # squared distance map
                dist2 = (grid_x - cx) ** 2 + (grid_y - cy) ** 2
                gauss = torch.exp(- dist2 / (2.0 * (sigma ** 2)))
                mask_lvl[b] = torch.clamp(mask_lvl[b] + gauss, 0.0, 1.0)
        masks.append(mask_lvl)
    return masks



# register helper so it can be used from configs if desired
@register
class HGS_EncoderWrapper(nn.Module):
    """A wrapper to expose HierarchicallyGatedSparseEncoder with same API as HybridEncoder forward.
    """
    def __init__(self, in_channels=(512,1024,2048), hidden_dim=256, **kwargs):
        """
        Args (from config):
            in_channels: list of backbone feature channels per level (high -> low resolution)
            hidden_dim: internal transformer dimension
            enabled: if False, acts as identity
        Other kwargs are forwarded to HierarchicallyGatedSparseEncoder.
        """
        super().__init__()
        chs = list(in_channels)
        enabled = kwargs.pop('enabled', True)
        self.enabled = bool(enabled)
        if self.enabled:
            self.encoder = HierarchicallyGatedSparseEncoder(
                in_channels_list=chs,
                hidden_dim=hidden_dim,
                **kwargs,
            )
        else:
            self.encoder = None

    def forward(self, feats, feats_ir=None):
        """
        Forward API made compatible with HybridEncoder:
            feats: list of backbone features [B, C_l, H_l, W_l]，例如 P3/P4/P5；
            feats_ir: 可选 P2 层级 IR 特征图 [B, C_ir, H2, W2]，
                      若提供则用于生成各尺度 gate map，对 P3/P4/P5 投影后的特征进行增强。

        Returns:
            outs: list of encoded feature maps (same channels as input per level)
            aux:  dict with key 'potentials' -> list of [B, H_l, W_l] potential maps
        """
        if not self.enabled or self.encoder is None:
            return feats, {}

        outs, potentials = self.encoder(feats, feats_ir=feats_ir)
        aux = {'potentials': potentials} if potentials is not None else {}
        return outs, aux
