"""by lyuwenyu
modified for multispectral by yimingshaw
"""

import os 
import sys 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import argparse

import src.misc.dist as dist 
from src.core import YAMLConfig 
from src.solver import TASKS


def main(args, ) -> None:
    '''main
    '''
    dist.init_distributed()
    cfg = YAMLConfig(args.config, resume=args.resume, use_amp=args.amp)
    solver = TASKS[cfg.yaml_cfg['task']](cfg)
    
    if args.test_only:
        solver.valX()
    else:
        solver.fitX()



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default=r'./configs/gmdetr/gmdetr_r50vd_6x_vedai_1024_X_potential_train2.yml')
    parser.add_argument('--resume', '-r', type=str, default=None)
    parser.add_argument('--test-only', action='store_true', default=False,)
    parser.add_argument('--amp', action='store_true', default=False,)

    args = parser.parse_args()

    main(args)
