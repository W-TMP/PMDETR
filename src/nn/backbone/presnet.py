'''by lyuwenyu
'''
import torch
import torch.nn as nn 
import torch.nn.functional as F 

from collections import OrderedDict
from pathlib import Path

from .common import get_activation, ConvNormLayer, FrozenBatchNorm2d

from src.core import register


__all__ = ['PResNetX']


ResNet_cfg = {
    18: [2, 2, 2, 2],
    34: [3, 4, 6, 3],
    50: [3, 4, 6, 3],
    101: [3, 4, 23, 3],
    # 152: [3, 8, 36, 3],
}


donwload_url = {
    18: 'https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet18_vd_pretrained_from_paddle.pth',
    34: 'https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet34_vd_pretrained_from_paddle.pth',
    50: 'https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet50_vd_ssld_v2_pretrained_from_paddle.pth',
    101: 'https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet101_vd_ssld_pretrained_from_paddle.pth',
}

# 预训练权重本地文件名（与 URL 中一致），默认从项目下 weights 目录加载
PRETRAINED_LOCAL_DIR = Path('/home1/wqy/code/PG-RTDETR/weights')
pretrained_filename = {
    18: 'ResNet18_vd_pretrained_from_paddle.pth',
    34: 'ResNet34_vd_pretrained_from_paddle.pth',
    50: 'ResNet50_vd_ssld_v2_pretrained_from_paddle.pth',
    101: 'ResNet101_vd_ssld_pretrained_from_paddle.pth',
}


def _fuse_convnorm(conv_norm: ConvNormLayer):
    """Fuse ConvNormLayer BN into conv weight and bias."""
    kernel = conv_norm.conv.weight
    running_mean = conv_norm.norm.running_mean
    running_var = conv_norm.norm.running_var
    gamma = conv_norm.norm.weight
    beta = conv_norm.norm.bias
    eps = conv_norm.norm.eps
    std = (running_var + eps).sqrt()
    t = (gamma / std).reshape(-1, 1, 1, 1)
    return kernel * t, beta - running_mean * gamma / std


def _is_avgpool_short(module) -> bool:
    """True if shortcut is AvgPool2d + ConvNormLayer (ResNet-vd downsample)."""
    if not isinstance(module, nn.Sequential) or len(module) != 2:
        return False
    return isinstance(module[0], nn.AvgPool2d) and isinstance(module[1], ConvNormLayer)


def _fuse_avgpool_convnorm(short: nn.Sequential) -> ConvNormLayer:
    """Fuse AvgPool2d(2,2) + ConvNormLayer(1x1) into ConvNormLayer(2x2, stride=2)."""
    conv_norm = short[1]
    ch_in = conv_norm.conv.in_channels
    ch_out = conv_norm.conv.out_channels
    kernel, bias = _fuse_convnorm(conv_norm)
    w2 = (kernel / 4.0).expand(-1, -1, 2, 2).contiguous()

    fused = ConvNormLayer(ch_in, ch_out, 2, 2, padding=0, act=None)
    fused.conv = nn.Conv2d(ch_in, ch_out, 2, stride=2, padding=0, bias=True)
    fused.conv.weight.data.copy_(w2)
    fused.conv.bias.data.copy_(bias)
    with torch.no_grad():
        fused.norm.weight.fill_(1)
        fused.norm.bias.zero_()
        fused.norm.running_mean.zero_()
        fused.norm.running_var.fill_(1)
    return fused


# 使用修正后的block类
class BiPathFusionBlock(nn.Module):
    def __init__(self, dim=256, r=16, L=32):
        super().__init__()
        d = max(dim // r, L)
        self.conv0 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 5, stride=1, padding=4, groups=dim, dilation=2)
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        self.conv = nn.Conv2d(2, 2, 1)
        self.conv_proj = nn.Conv2d(dim // 2, dim, 1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Sequential(
            nn.Conv2d(dim, d, 1, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Conv2d(d, dim, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # 兼容不同的输入格式
        if isinstance(x, (list, tuple)):
            x_vis, x_ir = x[0], x[1]
        else:
            raise ValueError("DFF requires two inputs: [x_vis, x_ir]")
        batch_size = x_vis.size(0)
        dim = x_vis.size(1)

        # 空间特征提取
        attn1 = self.conv0(x_vis)
        attn2 = self.conv_spatial(x_ir)

        # 投影到较低维度
        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)

        # 拼接空间特征
        attn = torch.cat([attn1, attn2], dim=1)

        # 空间注意力
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        spatial_attn = self.conv(agg).sigmoid()

        # 通道注意力
        ch_attn = self.global_pool(attn)
        # 当batch_size=1时，临时将BatchNorm设置为eval模式
        if batch_size == 1 and self.training:
            self.fc1.eval()
            z = self.fc1(ch_attn)
            self.fc1.train()
        else:
            z = self.fc1(ch_attn)
        a_b = self.fc2(z)

        # 修正的reshape操作
        a_b = a_b.view(batch_size, 2, dim // 2, 1, 1)
        a_b = self.softmax(a_b)
        a1, a2 = a_b.chunk(2, dim=1)
        a1 = a1.view(batch_size, dim // 2, 1, 1)
        a2 = a2.view(batch_size, dim // 2, 1, 1)

        # 应用空间注意力到通道权重
        w1 = a1 * spatial_attn[:, 0:1, :, :]
        w2 = a2 * spatial_attn[:, 1:2, :, :]

        # 融合注意力
        attn = attn1 * w1 + attn2 * w2
        attn = self.conv_proj(attn).sigmoid()

        return x_ir * attn + (1 - attn) * x_vis


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='b'):
        super().__init__()

        self.shortcut = shortcut

        if not shortcut:
            if variant == 'd' and stride == 2:
                self.short = nn.Sequential(OrderedDict([
                    ('pool', nn.AvgPool2d(2, 2, 0, ceil_mode=True)),
                    ('conv', ConvNormLayer(ch_in, ch_out, 1, 1))
                ]))
            else:
                self.short = ConvNormLayer(ch_in, ch_out, 1, stride)

        self.branch2a = ConvNormLayer(ch_in, ch_out, 3, stride, act=act)
        self.branch2b = ConvNormLayer(ch_out, ch_out, 3, 1, act=None)
        self.act = nn.Identity() if act is None else get_activation(act) 


    def forward(self, x):
        out = self.branch2a(x)
        out = self.branch2b(out)
        if self.shortcut:
            short = x
        else:
            short = self.short(x)
        
        out = out + short
        out = self.act(out)

        return out

    def convert_to_deploy(self):
        if self.shortcut or not _is_avgpool_short(self.short):
            return self
        self.short = _fuse_avgpool_convnorm(self.short)
        return self


class ConvAdapter(nn.Module):
    """
    Design 2 v4 - Adapter模块用于多模态特征适配
    """
    def __init__(self, inplanes, outplanes, width, 
                kernel_size=3, padding=1, stride=1, groups=1, dilation=1, norm_layer=None, act_layer=None, **kwargs):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.Identity
        if act_layer is None:
            act_layer = nn.Identity

        # depth-wise conv
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=kernel_size, stride=stride, groups=groups, padding=padding, dilation=int(dilation))
        self.act = act_layer()
        # point-wise conv
        self.conv2 = nn.Conv2d(width, outplanes, kernel_size=1, stride=1)
        # se - 可学习的缩放参数
        self.se = nn.Parameter(1.0 * torch.ones((1, outplanes, 1, 1)), requires_grad=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act(out)
        out = self.conv2(out)
        out = out * self.se
        return out


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='b', use_adapter=False):
        super().__init__()

        if variant == 'a':
            stride1, stride2 = stride, 1
        else:
            stride1, stride2 = 1, stride

        width = ch_out 

        self.branch2a = ConvNormLayer(ch_in, width, 1, stride1, act=act)
        self.branch2b = ConvNormLayer(width, width, 3, stride2, act=act)
        self.branch2c = ConvNormLayer(width, ch_out * self.expansion, 1, 1)

        # 添加adapter模块（用于多模态特征适配）
        self.use_adapter = use_adapter
        if use_adapter:
            self.adapter = ConvAdapter(
                width, width,
                kernel_size=3,
                padding=1,
                width=int(width // 8),
                stride=stride2,
                groups=int(width // 8),
                dilation=1,
                act_layer=nn.ReLU
            )

        self.shortcut = shortcut
        if not shortcut:
            if variant == 'd' and stride == 2:
                self.short = nn.Sequential(OrderedDict([
                    ('pool', nn.AvgPool2d(2, 2, 0, ceil_mode=True)),
                    ('conv', ConvNormLayer(ch_in, ch_out * self.expansion, 1, 1))
                ]))
            else:
                self.short = ConvNormLayer(ch_in, ch_out * self.expansion, 1, stride)

        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x, flag=1):
        """
        Args:
            x: 输入特征
            flag: 1=单模态（不使用adapter），2=双模态（使用adapter）
        """
        out = self.branch2a(x)
        
        # 在branch2a和branch2b之间使用adapter（仅当flag==2时）
        if self.use_adapter and flag == 2:
            out_adapter = self.adapter(out)
        
        out = self.branch2b(out)
        
        # 将adapter输出加到branch2b输出上
        if self.use_adapter and flag == 2:
            out = out + out_adapter
        
        out = self.branch2c(out)

        if self.shortcut:
            short = x
        else:
            short = self.short(x)

        out = out + short
        out = self.act(out)

        return out

    def convert_to_deploy(self):
        if self.shortcut or not _is_avgpool_short(self.short):
            return self
        self.short = _fuse_avgpool_convnorm(self.short)
        return self


class Blocks(nn.Module):
    def __init__(self, block, ch_in, ch_out, count, stage_num, act='relu', variant='b', use_adapter=False):
        super().__init__()

        self.blocks = nn.ModuleList()
        for i in range(count):
            block_kwargs = {
                'ch_in': ch_in,
                'ch_out': ch_out,
                'stride': 2 if i == 0 and stage_num != 2 else 1,
                'shortcut': False if i == 0 else True,
                'variant': variant,
                'act': act
            }
            # 如果是BottleNeck且需要adapter，添加use_adapter参数
            if block == BottleNeck and use_adapter:
                block_kwargs['use_adapter'] = True
            
            self.blocks.append(block(**block_kwargs))

            if i == 0:
                ch_in = ch_out * block.expansion

    def forward(self, x, flag=1):
        """
        Args:
            x: 输入特征
            flag: 1=单模态（不使用adapter），2=双模态（使用adapter）
        """
        out = x
        for block in self.blocks:
            # 检查block是否有use_adapter属性且为True，以及forward方法是否接受flag参数
            if hasattr(block, 'use_adapter') and block.use_adapter:
                out = block(out, flag=flag)
            else:
                out = block(out)
        return out


@register
class PResNetX(nn.Module):
    def __init__(
        self, 
        depth, 
        variant='d', 
        num_stages=4, 
        return_idx=[0, 1, 2, 3], 
        act='relu',
        freeze_at=-1,
        freeze_norm=True,
        pretrained=True,
        use_adapter=True,
        return_ir_feat=False):
        super().__init__()

        block_nums = ResNet_cfg[depth]
        ch_in = 64
        if variant in ['c', 'd']:
            conv_def = [
                [3, ch_in // 2, 3, 2, "conv1_1"],
                [ch_in // 2, ch_in // 2, 3, 1, "conv1_2"],
                [ch_in // 2, ch_in, 3, 1, "conv1_3"],
            ]
        else:
            conv_def = [[3, ch_in, 7, 2, "conv1_1"]]

        self.conv1 = nn.Sequential(OrderedDict([
            (_name, ConvNormLayer(c_in, c_out, k, s, act=act)) for c_in, c_out, k, s, _name in conv_def
        ]))

        ch_out_list = [64, 128, 256, 512]
        block = BottleNeck if depth >= 50 else BasicBlock

        _out_channels = [block.expansion * v for v in ch_out_list]
        _out_strides = [4, 8, 16, 32]

        self.res_layers = nn.ModuleList()
        for i in range(num_stages):
            stage_num = i + 2
            self.res_layers.append(
                Blocks(block, ch_in, ch_out_list[i], block_nums[i], stage_num, act=act, variant=variant, use_adapter=use_adapter)
            )
            ch_in = _out_channels[i]

        # 在 P2（stage0 输出）处双路径融合，维度为 stage0 输出通道（ResNet50 为 256）
        self.bi_fusion = BiPathFusionBlock()

        self.return_idx = return_idx
        self.out_channels = [_out_channels[_i] for _i in return_idx]
        self.out_strides = [_out_strides[_i] for _i in return_idx]
        self.use_adapter = use_adapter
        self.return_ir_feat = return_ir_feat

        if freeze_at >= 0:
            self._freeze_parameters(self.conv1)
            for i in range(min(freeze_at, num_stages)):
                self._freeze_parameters(self.res_layers[i])

        if freeze_norm:
            self._freeze_norm(self)

        if pretrained:
            state = torch.hub.load_state_dict_from_url(donwload_url[depth])
            result = self.load_state_dict(state, strict=False)
            print(f'Load PResNet{depth} state_dict: {result}')

    def _freeze_parameters(self, m: nn.Module):
        for p in m.parameters():
            p.requires_grad = False

    def _freeze_norm(self, m: nn.Module):
        if isinstance(m, nn.BatchNorm2d):
            m = FrozenBatchNorm2d(m.num_features)
        else:
            for name, child in m.named_children():
                _child = self._freeze_norm(child)
                if _child is not child:
                    setattr(m, name, _child)
        return m

    def forward(self, x, x_ir=None):
        """
        双模态前向（P2 融合）：
        - 输入：可见光 x_vis、红外 x_ir（或 [x_ir, x_vis]）
        - P2：双路经 stage0 后在 BiPathFusionBlock 融合，若 0 in return_idx 则输出
        - P3/P4/P5：仅对融合特征继续经 stage1/2/3，按 return_idx 输出
        - return_idx 可为 [0,1,2]（P2、P3、P4）或 [0,1,2,3]（P2～P5，需 num_stages=4）
        - return_ir_feat=True 时返回 (outs, x_ir)，x_ir 为 P2 尺度（stage0 后）的 IR 特征
        """
        # 兼容两种调用方式：
        # 1) backbone([x_ir, x_vis])
        # 2) backbone(x_vis, x_ir)
        if isinstance(x, (list, tuple)) and len(x) == 2:
            x_ir, x_vis = x[0], x[1]
        else:
            x_vis = x
            assert x_ir is not None, "PResNetX 现在固定为双模态，请同时传入 x_vis 和 x_ir"

        # stem：分别下采样两路
        conv1_vis = self.conv1(x_vis)
        x_vis = F.max_pool2d(conv1_vis, kernel_size=3, stride=2, padding=1)

        conv1_ir = self.conv1(x_ir)
        x_ir = F.max_pool2d(conv1_ir, kernel_size=3, stride=2, padding=1)

        outs = []

        # -------- Stage0 (P2)：双路分别提取，在 P2 处融合 --------
        x_vis = self.res_layers[0](x_vis, flag=1)
        x_ir = self.res_layers[0](x_ir, flag=2)
        x = self.bi_fusion([x_vis, x_ir])
        if 0 in self.return_idx:
            outs.append(x_ir)

        # -------- Stage1/2/3 (P3/P4/P5)：仅对融合特征继续前向 --------
        for idx, stage in enumerate(self.res_layers[1:], start=1):
            x = stage(x, flag=1)
            if idx in self.return_idx:
                outs.append(x)

        assert len(outs) == len(self.return_idx), "PResNetX 返回的特征数量与 return_idx 不一致"
        if self.return_ir_feat:
            return outs, x_ir
        return outs
