"""评估脚本 - 对应 train_X.py 的 multispectral 模型评估
在配置中的验证/测试集上评估，并从 output_dir/checkpoint.pth 加载模型权重。
by lyuwenyu
modified for multispectral by yimingshaw
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import argparse

import src.misc.dist as dist
from src.core import YAMLConfig
from src.core.yaml_utils import load_config
from src.solver import TASKS


def main(args) -> None:
    """在测试集（config 中 val_dataloader）上评估，并加载 checkpoint 权重。"""
    dist.init_distributed()

    # 先读取配置，拿到 output_dir
    raw_cfg = load_config(args.config)
    output_dir = raw_cfg.get('output_dir', './output')
    checkpoint = os.path.join(output_dir, 'checkpoint_best.pth')
    print(f'加载主干网络及模型 checkpoint: {checkpoint}')

    # 通过 resume 把 checkpoint 传进去，BaseSolver.eval() 会调用 self.resume() 加载模型
    cfg = YAMLConfig(args.config, resume=checkpoint, use_amp=args.amp)
    solver = TASKS[cfg.yaml_cfg['task']](cfg)
    solver.eval()
    solver.valX()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='在测试集上评估 (valX)，从 checkpoint 加载主干网络权重')
    parser.add_argument(
        '--config', '-c', type=str,
        default=r'./configs/gmdetr/gmdetr_r50vd_6x_llvip_1024_X_potential_train2.yml',
        help='配置文件路径（其中 output_dir 决定 checkpoint 路径）'
    )
    parser.add_argument(
        '--amp', action='store_true', default=False,
        help='是否使用混合精度'
    )

    args = parser.parse_args()
    main(args)