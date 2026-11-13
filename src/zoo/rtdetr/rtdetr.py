"""by lyuwenyu
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import random 
import numpy as np 

from src.core import register


__all__ = ['RTDETR', 'RTDETRX']


@register
class RTDETR(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', 'fusion']

    def __init__(self, backbone: nn.Module, encoder, decoder, fusion=None, multi_scale=None):
        super().__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder
        self.fusion = fusion
        self.multi_scale = multi_scale

    def forward(self, x, targets=None):
        if self.multi_scale and self.training:
            sz = np.random.choice(self.multi_scale)
            x = F.interpolate(x, size=[sz, sz])

        feats = self.backbone(x)

        # encoder may return feats or (feats, aux)
        enc_out = self.encoder(feats)
        if isinstance(enc_out, tuple):
            feats_enc, enc_aux = enc_out
        else:
            feats_enc, enc_aux = enc_out, {}

        # optional fusion neck, may use potentials from enc_aux
        if self.fusion is not None:
            potentials = None
            if isinstance(enc_aux, dict):
                potentials = enc_aux.get('potentials', None)
            try:
                feats_enc = self.fusion(feats_enc, potentials=potentials)
            except TypeError:
                feats_enc = self.fusion(feats_enc)

        out = self.decoder(feats_enc, targets)
        if isinstance(out, dict) and isinstance(enc_aux, dict) and enc_aux:
            out.update(enc_aux)
        return out
    
    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self 


@register
class RTDETRX(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', 'fusion']

    def __init__(self, backbone: nn.Module, encoder, decoder, fusion=None, multi_scale=None):
        super().__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder
        self.fusion = fusion
        self.multi_scale = multi_scale

    def forward(self, x, x1, targets=None):
        if self.multi_scale and self.training:
            sz = np.random.choice(self.multi_scale)
            x = F.interpolate(x, size=[sz, sz])
            x1 = F.interpolate(x1, size=[sz, sz])

        # backbone can return feats or (feats, feats_ir)
        backbone_out = self.backbone(x, x1)
        if isinstance(backbone_out, (list, tuple)) and len(backbone_out) == 2:
            feats, feats_ir = backbone_out[0], backbone_out[1]
        else:
            feats, feats_ir = backbone_out, None

        # First try encoder(feats, feats_ir=...), fall back to encoder(feats)
        try:
            enc_out = self.encoder(feats, feats_ir=feats_ir)
        except TypeError:
            enc_out = self.encoder(feats)

        if isinstance(enc_out, tuple):
            feats_enc, enc_aux = enc_out
        else:
            feats_enc, enc_aux = enc_out, {}

        # optional fusion neck (e.g., FusionWrapper), may consume potentials
        if self.fusion is not None:
            potentials = None
            if isinstance(enc_aux, dict):
                potentials = enc_aux.get('potentials', None)
            try:
                feats_enc = self.fusion(feats_enc, potentials=potentials)
            except TypeError:
                feats_enc = self.fusion(feats_enc)

        out = self.decoder(feats_enc, targets)
        if isinstance(out, dict) and isinstance(enc_aux, dict) and enc_aux:
            out.update(enc_aux)
        return out

    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self