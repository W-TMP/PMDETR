"""多光谱 RT-DETRX 推理可视化脚本

功能：
- 从 config 的 output_dir/checkpoint.pth（或指定 checkpoint）加载训练好的 RTDETRX 模型；
- 在验证集（val_dataloader）上做前向推理；
- 将预测框画到可见光图像上并保存到指定目录，便于直观查看检测效果。

适配多光谱 CocoXDetection：
- backbone=PResNetX，encoder=HybridEncoder/HGS_EncoderWrapper，model=RTDETRX；
- DataLoader 返回 (samples_vis, samples_ir, targets)，与 evaluateX 保持一致。
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import argparse
from typing import Optional

import torch
from PIL import Image, ImageDraw

import src.misc.dist as dist
from src.core import YAMLConfig
from src.core.yaml_utils import load_config
from src.solver import TASKS


# 简单的类别调色板，不同类别用不同颜色
CATEGORY_PALETTE = [
    (255, 0, 0),
    (0, 0, 255),
    (0, 255, 0),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 127, 127),
    (127, 127, 255),
    (127, 255, 127),
    (255, 127, 255),
    (255, 255, 127),
]


def load_model_and_data(
    config_path: str,
    checkpoint: Optional[str],
    use_amp: bool = False,
):
    """根据配置和 checkpoint 构建 solver，并返回模型、后处理、数据加载器等对象。"""
    # 先读取配置，拿到 output_dir（若未显式指定 checkpoint）
    raw_cfg = load_config(config_path)
    output_dir = raw_cfg.get('output_dir', './output')

    if checkpoint is None:
        checkpoint = os.path.join(output_dir, 'checkpoint.pth')
    print(f'加载推理 checkpoint: {checkpoint}')

    cfg = YAMLConfig(config_path, resume=checkpoint, use_amp=use_amp)
    solver = TASKS[cfg.yaml_cfg['task']](cfg)

    # eval() 内部会调用 setup() + resume()，构建 model / postprocessor / 数据加载器
    solver.eval()

    model = solver.ema.module if solver.ema is not None else solver.model
    postprocessor = solver.postprocessor
    data_loader = solver.val_dataloader
    device = solver.device

    return model, postprocessor, data_loader, device


def visualize_one_image(
    dataset,
    target: dict,
    result: dict,
    save_dir: str,
    score_thr: float = 0.3,
):
    """根据 COCO 标注 id 重新读取原始可见光/红外图像，在其上画预测框并分别保存。

    - 不同类别使用不同颜色；
    - 类别标签显示为具体的物体名称（来自 COCO/VEDAI 的 category.name），而不是数字 id。
    """
    image_id = int(target["image_id"])
    coco = dataset.coco
    file_name = coco.loadImgs(image_id)[0]["file_name"]
    vis_path = os.path.join(dataset.img_folder, file_name)

    # 推导红外图路径：.../vis_imgs/xxx.jpg -> .../ir_imgs/xxx.jpg
    ir_path = os.path.join(
        os.path.dirname(os.path.dirname(vis_path)),
        'ir_imgs',
        os.path.basename(vis_path),
    )

    os.makedirs(save_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(file_name))[0]
    save_vis_path = os.path.join(save_dir, f'{stem}_vis_pred.jpg')
    save_ir_path = os.path.join(save_dir, f'{stem}_ir_pred.jpg')

    img_vis = Image.open(vis_path).convert("RGB")
    # 若缺失 IR 图，则只可视化可见光
    if os.path.exists(ir_path):
        img_ir = Image.open(ir_path).convert("RGB")
    else:
        img_ir = img_vis.copy()

    draw_vis = ImageDraw.Draw(img_vis)
    draw_ir = ImageDraw.Draw(img_ir)

    boxes = result["boxes"]
    scores = result["scores"]
    labels = result["labels"]

    # 为类别 id 分配稳定的颜色索引
    cats = getattr(dataset, "coco", None).cats if hasattr(dataset, "coco") else {}
    if isinstance(cats, dict) and len(cats) > 0:
        sorted_ids = sorted(cats.keys())
        catid2idx = {cid: idx for idx, cid in enumerate(sorted_ids)}
    else:
        catid2idx = {}

    for box, score, label in zip(boxes, scores, labels):
        if float(score) < score_thr:
            continue

        cat_id = int(label)
        # 获取类别名称：优先从 COCO/VEDAI 的 categories 中读取
        if isinstance(cats, dict) and cat_id in cats:
            class_name = str(cats[cat_id].get("name", cat_id))
        else:
            class_name = str(cat_id)

        # 为该类别选择颜色
        color_idx = catid2idx.get(cat_id, 0)
        color = CATEGORY_PALETTE[color_idx % len(CATEGORY_PALETTE)]

        x1, y1, x2, y2 = box.tolist()
        draw_vis.rectangle([x1, y1, x2, y2], outline=color, width=2)
        draw_ir.rectangle([x1, y1, x2, y2], outline=color, width=2)

        txt = f"{class_name}:{float(score):.2f}"
        # 文本简单画在框左上角
        draw_vis.text((x1, y1), txt, fill=color)
        draw_ir.text((x1, y1), txt, fill=color)

    # 分别保存可见光和红外结果
    img_vis.save(save_vis_path)
    img_ir.save(save_ir_path)
    print(f"保存可视化结果到: {save_vis_path}")
    print(f"保存可视化结果到: {save_ir_path}")


def main(args) -> None:
    dist.init_distributed()

    model, postprocessor, data_loader, device = load_model_and_data(
        args.config,
        args.checkpoint,
        use_amp=args.amp,
    )

    model.eval()
    dataset = data_loader.dataset

    os.makedirs(args.save_dir, exist_ok=True)

    num_vis = 0
    max_vis = args.max_images if args.max_images is not None and args.max_images > 0 else None

    with torch.no_grad():
        for samples_vis, samples_ir, targets in data_loader:
            samples_vis = samples_vis.to(device)
            samples_ir = samples_ir.to(device)
            targets = [{k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                        for k, v in t.items()} for t in targets]

            outputs = model(samples_vis, samples_ir)
            orig_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
            results = postprocessor(outputs, orig_sizes)

            for tgt, res in zip(targets, results):
                visualize_one_image(
                    dataset,
                    tgt,
                    res,
                    save_dir=args.save_dir,
                    score_thr=args.score_thr,
                )
                num_vis += 1
                if max_vis is not None and num_vis >= max_vis:
                    print(f"已达到最大可视化数量 {max_vis}，结束。")
                    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='多光谱 RT-DETRX 推理可视化脚本')
    parser.add_argument(
        '--config', '-c', type=str,
        default=r'./configs/gmdetr/gmdetr_r50vd_6x_llvip_1024_X_potential_train2.yml',
        help='配置文件路径'
    )
    parser.add_argument(
        '--checkpoint', '-ckpt', type=str, default=None,
        help='模型权重路径；若不指定，则使用 config 中 output_dir/checkpoint.pth'
    )
    parser.add_argument(
        '--save-dir', '-o', type=str,
        default='./inference',
        help='可视化结果保存目录'
    )
    parser.add_argument(
        '--score-thr', type=float, default=0.4,
        help='可视化时的分数阈值'
    )
    parser.add_argument(
        '--max-images', type=int, default=0,
        help='最多可视化的图像数量；<=0 表示全部'
    )
    parser.add_argument(
        '--amp', action='store_true', default=False,
        help='是否使用混合精度'
    )

    main(parser.parse_args())

