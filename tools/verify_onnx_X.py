"""Compare ONNX Runtime inference with PyTorch (deploy path used at export).

Uses the same val_dataloader and ``orig_size`` as ``infer_X.py``. The primary
comparison is **deploy PyTorch** (model.deploy + postprocessor.deploy) vs ONNX,
which matches ``export_onnx_X.py``.

Optionally compares with **infer_X-style** PyTorch (no deploy, postprocessor
returns per-image dicts) to see export vs training-time inference differences.

Example::

    python tools/verify_onnx_X.py \\
        -c configs/gmdetr/gmdetr_r50vd_6x_llvip_1024_X_potential_train2.yml \\
        -r output/gmdetr_r50vd_6x_llvip_1024_X_potential_train2/checkpoint_best.pth \\
        --onnx gmdetr_llvip_1024.onnx

    python tools/verify_onnx_X.py -c ... -r ... --onnx model.onnx --max-batches 5
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import argparse
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

import src.misc.dist as dist
from src.core import YAMLConfig
from src.solver import TASKS


def build_deploy_model(cfg) -> nn.Module:
    """Same wrapper as export_onnx_X.py (deploy model + deploy postprocessor)."""

    class DeployModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images_vis, images_ir, orig_target_sizes):
            outputs = self.model(images_vis, images_ir)
            return self.postprocessor(outputs, orig_target_sizes)

    return DeployModel()


def load_checkpoint(cfg, resume: str) -> None:
    checkpoint = torch.load(resume, map_location='cpu')
    if 'ema' in checkpoint:
        state = checkpoint['ema']['module']
    else:
        state = checkpoint['model']
    cfg.model.load_state_dict(state)


def load_val_dataloader(config_path: str, checkpoint: str, use_amp: bool = False):
    cfg = YAMLConfig(config_path, resume=checkpoint, use_amp=use_amp)
    solver = TASKS[cfg.yaml_cfg['task']](cfg)
    solver.eval()
    device = solver.device
    model = solver.ema.module if solver.ema is not None else solver.model
    postprocessor = solver.postprocessor
    return cfg, solver.val_dataloader, model, postprocessor, device


def tensor_stats(name: str, a: torch.Tensor, b: torch.Tensor) -> Tuple[float, float]:
    diff = (a.float() - b.float()).abs()
    return diff.max().item(), diff.mean().item()


def compare_deploy_vs_onnx(
    pt_model: nn.Module,
    ort_session,
    data_loader,
    device: torch.device,
    max_batches: Optional[int],
    atol: float,
) -> bool:
    all_ok = True
    n_batch = 0

    for samples_vis, samples_ir, targets in data_loader:
        samples_vis = samples_vis.to(device)
        samples_ir = samples_ir.to(device)
        orig_sizes = torch.stack([t['orig_size'] for t in targets], dim=0).to(device)

        with torch.no_grad():
            pt_labels, pt_boxes, pt_scores = pt_model(samples_vis, samples_ir, orig_sizes)

        feed = {
            'images_vis': samples_vis.detach().cpu().numpy().astype(np.float32),
            'images_ir': samples_ir.detach().cpu().numpy().astype(np.float32),
            'orig_target_sizes': orig_sizes.detach().cpu().numpy().astype(np.float32),
        }
        ort_labels, ort_boxes, ort_scores = ort_session.run(None, feed)

        ort_labels = torch.as_tensor(ort_labels, device=device)
        ort_boxes = torch.as_tensor(ort_boxes, device=device)
        ort_scores = torch.as_tensor(ort_scores, device=device)

        batch_ok = True
        for key, pt_t, ort_t in (
            ('labels', pt_labels, ort_labels),
            ('boxes', pt_boxes, ort_boxes),
            ('scores', pt_scores, ort_scores),
        ):
            mx, mn = tensor_stats(key, pt_t, ort_t)
            ok = mx <= atol
            batch_ok = batch_ok and ok
            print(
                f'  batch {n_batch} {key}: max_diff={mx:.6e} mean_diff={mn:.6e} '
                f'-> {"OK" if ok else "FAIL"}'
            )

        image_ids = [int(t['image_id']) for t in targets]
        print(f'  image_ids={image_ids} orig_size[0]={orig_sizes[0].tolist()}')

        all_ok = all_ok and batch_ok
        n_batch += 1
        if max_batches is not None and n_batch >= max_batches:
            break

    return all_ok


def compare_infer_x_vs_onnx(
    model: nn.Module,
    postprocessor: nn.Module,
    ort_session,
    data_loader,
    device: torch.device,
    max_batches: Optional[int],
    score_thr: float,
) -> None:
    """Heuristic compare infer_X postprocessor dicts vs ONNX (shapes/formats differ)."""
    print('\n--- infer_X-style PyTorch vs ONNX (top detections, score_thr=%.2f) ---' % score_thr)
    n_batch = 0

    for samples_vis, samples_ir, targets in data_loader:
        samples_vis = samples_vis.to(device)
        samples_ir = samples_ir.to(device)
        orig_sizes = torch.stack([t['orig_size'] for t in targets], dim=0)

        with torch.no_grad():
            outputs = model(samples_vis, samples_ir)
            results = postprocessor(outputs, orig_sizes.to(device))

        feed = {
            'images_vis': samples_vis.detach().cpu().numpy().astype(np.float32),
            'images_ir': samples_ir.detach().cpu().numpy().astype(np.float32),
            'orig_target_sizes': orig_sizes.numpy().astype(np.float32),
        }
        ort_labels, ort_boxes, ort_scores = ort_session.run(None, feed)

        for i, (tgt, res) in enumerate(zip(targets, results)):
            image_id = int(tgt['image_id'])
            pt_n = int((res['scores'] >= score_thr).sum())
            ort_n = int((ort_scores[i] >= score_thr).sum())
            print(f'  image_id={image_id}: pt_dets@{score_thr}={pt_n} onnx_dets@{score_thr}={ort_n}')

        n_batch += 1
        if max_batches is not None and n_batch >= max_batches:
            break


def main(args) -> None:
    dist.init_distributed()

    if not os.path.isfile(args.onnx):
        raise FileNotFoundError(f'ONNX file not found: {args.onnx}')

    import onnxruntime as ort

    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    if args.cpu_only:
        providers = ['CPUExecutionProvider']
    ort_session = ort.InferenceSession(args.onnx, providers=providers)
    print(f'ONNX inputs: {[i.name for i in ort_session.get_inputs()]}')
    print(f'ONNX outputs: {[o.name for o in ort_session.get_outputs()]}')
    print(f'ORT providers: {ort_session.get_providers()}')

    _, data_loader, infer_model, postprocessor, device = load_val_dataloader(
        args.config, args.resume, use_amp=args.amp)

    cfg = YAMLConfig(args.config, resume=args.resume)
    load_checkpoint(cfg, args.resume)
    deploy_model = build_deploy_model(cfg).to(device).eval()

    print('\n=== Deploy PyTorch vs ONNX (should match export_onnx_X.py) ===')
    ok = compare_deploy_vs_onnx(
        deploy_model,
        ort_session,
        data_loader,
        device,
        args.max_batches,
        args.atol,
    )
    print(f'\nDeploy vs ONNX: {"PASS" if ok else "FAIL"} (atol={args.atol})')

    if args.compare_infer_x:
        compare_infer_x_vs_onnx(
            infer_model.eval(),
            postprocessor,
            ort_session,
            data_loader,
            device,
            args.max_batches,
            args.score_thr,
        )
        print(
            '\nNote: infer_X uses non-deploy postprocessor (may remap labels). '
            'Large diffs here are expected; rely on Deploy vs ONNX for export validation.'
        )

    if not ok:
        sys.exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Verify ONNX vs PyTorch for RTDETRX')
    parser.add_argument('--config', '-c', type=str, required=True)
    parser.add_argument('--resume', '-r', type=str, required=True)
    parser.add_argument('--onnx', type=str, required=True, help='Path to exported .onnx')
    parser.add_argument('--max-batches', type=int, default=3,
                        help='Number of val batches to compare (default: 3)')
    parser.add_argument('--atol', type=float, default=1e-4,
                        help='Max abs diff tolerance for deploy vs ONNX')
    parser.add_argument('--score-thr', type=float, default=0.3,
                        help='Score threshold for infer_X vs ONNX count comparison')
    parser.add_argument('--compare-infer-x', action='store_true',
                        help='Also print infer_X-style detection counts vs ONNX')
    parser.add_argument('--cpu-only', action='store_true',
                        help='Use CPU ORT provider only')
    parser.add_argument('--amp', action='store_true', default=False)
    main(parser.parse_args())
