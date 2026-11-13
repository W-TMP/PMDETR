"""Export RTDETRX / GMDETR (visible + IR) to ONNX.

Dual-modal models take ``images_vis`` and ``images_ir`` (NCHW, RGB) plus
``orig_target_sizes`` (N x 2, same convention as single-modal export).
Outputs: ``labels``, ``boxes``, ``scores``.

Before export, ``cfg.model.deploy()`` is called. This fuses ResNet-vd
``AvgPool2d + 1x1`` shortcuts in PResNetX into a single ``Conv2d stride=2``
(see ``BasicBlock`` / ``BottleNeck.convert_to_deploy`` in presnet.py) so ONNX
export does not hit ``get_pool_ceil_padding`` errors.

Example::

    python tools/export_onnx_X.py \\
        -c configs/gmdetr/gmdetr_r50vd_6x_llvip_1024_X_potential_train2.yml \\
        -r output/gmdetr_r50vd_6x_llvip_1024_X_potential_train2/checkpoint.pth \\
        -f gmdetr_llvip_1024.onnx --check

    python tools/export_onnx_X.py -c ... -r ... --verify-deploy
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import argparse
import copy

from src.core import YAMLConfig

import torch
import torch.nn as nn


def resolve_input_hw(cfg, height=None, width=None):
    """Return (height, width) for dummy tensors and orig_target_sizes."""
    if height is not None or width is not None:
        if height is None or width is None:
            raise ValueError('Both --height and --width must be set when overriding input size.')
        return int(height), int(width)

    yaml_cfg = getattr(cfg, 'yaml_cfg', {})
    for key in ('HGS_EncoderWrapper', 'HybridEncoder', 'RTDETRTransformer'):
        block = yaml_cfg.get(key)
        if isinstance(block, dict):
            es = block.get('eval_spatial_size')
            if es is not None:
                return int(es[0]), int(es[1])

    return 640, 640


def _output_max_diff(out_a, out_b):
    """Max abs diff over pred_logits and pred_boxes in model output dict."""
    max_diff = 0.0
    for key in ('pred_logits', 'pred_boxes'):
        if key in out_a and key in out_b:
            d = (out_a[key] - out_b[key]).abs().max().item()
            max_diff = max(max_diff, d)
    return max_diff


def verify_deploy(cfg, height, width, atol=1e-5):
    """Compare RTDETRX outputs before and after deploy() on dummy inputs."""
    vis = torch.rand(1, 3, height, width)
    ir = torch.rand(1, 3, height, width)

    model_before = copy.deepcopy(cfg.model)
    model_before.eval()
    with torch.no_grad():
        out_before = model_before(vis, ir)

    model_after = copy.deepcopy(cfg.model)
    model_after.deploy()
    model_after.eval()
    with torch.no_grad():
        out_after = model_after(vis, ir)

    max_diff = _output_max_diff(out_before, out_after)
    ok = max_diff <= atol
    print(f'verify-deploy: max_diff={max_diff:.6e} (atol={atol}) -> {"OK" if ok else "FAIL"}')
    if not ok:
        raise RuntimeError(
            f'deploy() changed model outputs beyond tolerance (max_diff={max_diff})')
    return max_diff


def main(args):
    cfg = YAMLConfig(args.config, resume=args.resume)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
    else:
        raise AttributeError('only support resume to load model.state_dict by now.')

    cfg.model.load_state_dict(state)

    height, width = resolve_input_hw(cfg, args.height, args.width)

    if args.verify_deploy:
        verify_deploy(cfg, height, width)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
            print(self.postprocessor.deploy_mode)

        def forward(self, images_vis, images_ir, orig_target_sizes):
            outputs = self.model(images_vis, images_ir)
            return self.postprocessor(outputs, orig_target_sizes)

    model = Model()
    model.eval()
    print(f'Export input size: {height}x{width} (NCHW)')

    dynamic_axes = {
        'images_vis': {0: 'N'},
        'images_ir': {0: 'N'},
        'orig_target_sizes': {0: 'N'},
    }

    data_vis = torch.rand(1, 3, height, width)
    data_ir = torch.rand(1, 3, height, width)
    size = torch.tensor([[height, width]])

    torch.onnx.export(
        model,
        (data_vis, data_ir, size),
        args.file_name,
        input_names=['images_vis', 'images_ir', 'orig_target_sizes'],
        output_names=['labels', 'boxes', 'scores'],
        dynamic_axes=dynamic_axes,
        opset_version=16,
        verbose=False,
    )

    if args.check:
        import onnx
        onnx_model = onnx.load(args.file_name)
        onnx.checker.check_model(onnx_model)
        print('Check export onnx model done...')

    if args.simplify:
        import onnx
        import onnxsim
        dynamic = True
        input_shapes = {
            'images_vis': data_vis.shape,
            'images_ir': data_ir.shape,
            'orig_target_sizes': size.shape,
        } if dynamic else None
        onnx_model_simplify, check = onnxsim.simplify(
            args.file_name, input_shapes=input_shapes, dynamic_input_shape=dynamic)
        onnx.save(onnx_model_simplify, args.file_name)
        print(f'Simplify onnx model {check}...')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export RTDETRX/GMDETR to ONNX')
    parser.add_argument('--config', '-c', type=str, required=True)
    parser.add_argument('--resume', '-r', type=str, required=True)
    parser.add_argument('--file-name', '-f', type=str, default='model_X.onnx')
    parser.add_argument('--check', action='store_true', default=False)
    parser.add_argument('--simplify', action='store_true', default=False)
    parser.add_argument('--verify-deploy', action='store_true', default=False,
                        help='Compare model outputs before/after deploy() then export')
    parser.add_argument('--height', type=int, default=None,
                        help='Input height (must set --width together)')
    parser.add_argument('--width', type=int, default=None,
                        help='Input width (must set --height together)')

    main(parser.parse_args())
