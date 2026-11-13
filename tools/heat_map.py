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
import numpy as np

import src.misc.dist as dist
from src.core import YAMLConfig
from src.core.yaml_utils import load_config
from src.solver import TASKS


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
    potentials_per_level=None,
    save_potential: bool = False,
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

    # 为类别 id 分配稳定的颜色和名称
    cats = getattr(dataset, "coco", None).cats if hasattr(dataset, "coco") else {}
    if isinstance(cats, dict) and len(cats) > 0:
        sorted_ids = sorted(cats.keys())
        catid2name = {cid: str(cats[cid].get("name", cid)) for cid in sorted_ids}
    else:
        catid2name = {}

    # 固定的颜色方案（参考给定 draw_bbox 实现）
    base_colors = {
        0: np.array([255, 0, 0]),
        1: np.array([0, 0, 255]),
        2: np.array([0, 255, 0]),
        3: np.array([255, 165, 0]),
        4: np.array([160, 32, 240]),
        5: np.array([0, 255, 255]),
        6: np.array([148, 87, 235]),
        7: np.array([255, 127, 80]),
        8: np.array([135, 206, 250]),
    }
    catid2color = {}

    # 根据图像尺寸设定基础线宽和字体大小（随分辨率自适应）
    img_w, img_h = img_vis.size
    base_thickness = max(1, int(min(img_w, img_h) * 0.003))  # 例如 1024 边长 -> 线宽约 3
    try:
        from PIL import ImageFont
        base_font_size = max(8, int(min(img_w, img_h) * 0.02))  # 例如 1024 边长 -> 字号约 20
        font = ImageFont.truetype(font=None, size=base_font_size)
    except Exception:
        font = None

    for box, score, label in zip(boxes, scores, labels):
        if float(score) < score_thr:
            continue

        cat_id = int(label)
        # 获取类别名称：优先从 COCO/VEDAI 的 categories 中读取
        class_name = catid2name.get(cat_id, str(cat_id))

        # 为该类别选择颜色（优先使用固定方案，否则随机从 colormap 中采样）
        if cat_id not in catid2color:
            if cat_id in base_colors:
                catid2color[cat_id] = base_colors[cat_id]
            else:
                # 退化方案：随机颜色但对同类稳定（用 cat_id 当种子）
                rng = np.random.RandomState(cat_id)
                catid2color[cat_id] = rng.randint(0, 255, size=3)
        color = tuple(int(c) for c in catid2color[cat_id])

        x1, y1, x2, y2 = box.tolist()

        # 根据目标框尺寸微调线宽
        box_w, box_h = abs(x2 - x1), abs(y2 - y1)
        thickness = max(base_thickness, int(min(box_w, box_h) * 0.01))

        draw_vis.rectangle([x1, y1, x2, y2], outline=color, width=thickness)
        draw_ir.rectangle([x1, y1, x2, y2], outline=color, width=thickness)

        # 文本与底色框（参考 draw_bbox 实现）
        txt = f"{class_name} {float(score):.2f}"
        if font is not None:
            # 获取文本尺寸
            xy0 = (0, 0)
            _, _, tw, th = draw_vis.textbbox(xy0, txt, font=font)
        else:
            # 没有字体对象时，大致估算文本框高度
            tw, th = (len(txt) * 6, base_font_size if 'base_font_size' in locals() else 10)

        # 计算文本背景框位置：画在框左上角上方，并保证 y1 >= y0
        xmin, ymin = x1, y1
        text_top = max(0.0, ymin - th)
        y0 = min(text_top, ymin)
        y1 = max(text_top, ymin)
        x0 = xmin + 1.0
        x1_bg = xmin + tw + 1.0
        text_bg_vis = [(x0, y0), (x1_bg, y1)]
        text_bg_ir = text_bg_vis

        # 先画实心矩形作为文字背景，再用白字绘制类别+分数
        draw_vis.rectangle(text_bg_vis, fill=color)
        draw_ir.rectangle(text_bg_ir, fill=color)

        if font is not None:
            draw_vis.text((xmin + 1, text_top), txt, fill=(255, 255, 255), font=font)
            draw_ir.text((xmin + 1, text_top), txt, fill=(255, 255, 255), font=font)
        else:
            draw_vis.text((xmin + 1, text_top), txt, fill=(255, 255, 255))
            draw_ir.text((xmin + 1, text_top), txt, fill=(255, 255, 255))

    # 分别保存可见光和红外结果
    img_vis.save(save_vis_path)
    img_ir.save(save_ir_path)
    print(f"保存可视化结果到: {save_vis_path}")
    print(f"保存可视化结果到: {save_ir_path}")

    # 可选：保存潜能图热力图（对每个尺度各存一张，可叠加在原始可见光图像上，风格类似 CAM）
    if save_potential and potentials_per_level is not None:
        try:
            img_raw = Image.open(vis_path).convert("RGB")
        except Exception:
            img_raw = img_vis
        img_w, img_h = img_raw.size
        for lvl, pot in enumerate(potentials_per_level):
            if pot is None:
                continue
            # pot: [H_l, W_l] 或 [1, H_l, W_l]
            if pot.dim() == 3 and pot.shape[0] == 1:
                pot = pot[0]
            pot = pot.float()
            if pot.numel() == 0:
                continue
            # 归一化到 [0,1]
            pot_min = float(pot.min())
            pot_max = float(pot.max())
            if pot_max - pot_min < 1e-6:
                continue
            pot_norm = (pot - pot_min) / (pot_max - pot_min + 1e-6)
            pot_np = pot_norm.cpu().numpy()

            # 使用彩色 colormap（类似 Grad-CAM 的 JET），若 matplotlib 不可用则退化为红色热图
            heat_rgb = None
            try:
                import matplotlib.cm as cm

                cmap = cm.get_cmap('jet')
                heat_color = cmap(pot_np)  # [H, W, 4] RGBA, 值域 0~1
                heat_rgb = np.uint8(heat_color[..., :3] * 255.0)
            except Exception:
                pot_u8 = np.uint8(pot_np * 255.0)
                heat_rgb = np.stack(
                    [pot_u8, np.zeros_like(pot_u8), np.zeros_like(pot_u8)],
                    axis=2,
                )

            heat_img = Image.fromarray(heat_rgb).resize((img_w, img_h), resample=Image.BILINEAR)
            overlay = Image.blend(img_raw, heat_img, alpha=0.45)
            save_pot_path = os.path.join(save_dir, f'{stem}_pot_l{lvl}.jpg')
            overlay.save(save_pot_path)
            print(f"保存潜能图热力图到: {save_pot_path}")


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
            # 从模型输出中提取潜能图（兼容 HGS_EncoderWrapper 和 HybridEncoder+PotentialMapHead）
            batch_potentials = None
            if isinstance(outputs, dict):
                # HGS_EncoderWrapper: 'potentials' 是每个尺度的列表，每个元素 [B, H_l, W_l]
                if 'potentials' in outputs and outputs['potentials'] is not None:
                    pot_list = outputs['potentials']
                    if isinstance(pot_list, (list, tuple)) and len(pot_list) > 0:
                        B = pot_list[0].shape[0]
                        batch_potentials = []
                        for b in range(B):
                            per_img = [p[b].detach().cpu() for p in pot_list]
                            batch_potentials.append(per_img)
                # HybridEncoder: 'potential_map' 为 [B,1,H,W]，这里也支持一下
                elif 'potential_map' in outputs and outputs['potential_map'] is not None:
                    pm = outputs['potential_map']
                    if pm.dim() == 4:
                        B = pm.shape[0]
                        batch_potentials = []
                        for b in range(B):
                            batch_potentials.append([pm[b, 0].detach().cpu()])

            orig_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
            results = postprocessor(outputs, orig_sizes)

            for idx, (tgt, res) in enumerate(zip(targets, results)):
                visualize_one_image(
                    dataset,
                    tgt,
                    res,
                    save_dir=args.save_dir,
                    score_thr=args.score_thr,
                    potentials_per_level=batch_potentials[idx] if batch_potentials is not None else None,
                    save_potential=args.save_potential,
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
        default='./inference_vis',
        help='可视化结果保存目录'
    )
    parser.add_argument(
        '--score-thr', type=float, default=0.3,
        help='可视化时的分数阈值'
    )
    parser.add_argument(
        '--max-images', type=int, default=5,
        help='最多可视化的图像数量；<=0 表示全部'
    )
    parser.add_argument(
        '--amp', action='store_true', default=False,
        help='是否使用混合精度'
    )
    parser.add_argument(
        '--save-potential', action='store_true', default=True,
        help='是否同时保存潜能图热力图（从编码器或潜能图头输出中提取）'
    )

    main(parser.parse_args())

