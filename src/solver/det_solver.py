'''
by lyuwenyu
modified for multispectral by yimingshaw
'''
import time 
import json
import datetime

import torch 

from src.misc import dist
from src.data import get_coco_api_from_dataset

from .solver import BaseSolver
from .det_engine import train_one_epoch, evaluate, train_one_epoch_X, evaluateX


class DetSolver(BaseSolver):
    
    def fit(self, ):
        print("Start training")
        self.train()

        args = self.cfg 
        
        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)
        # best_stat = {'coco_eval_bbox': 0, 'coco_eval_masks': 0, 'epoch': -1, }
        best_stat = {'epoch': -1, }

        start_time = time.time()
        for epoch in range(self.last_epoch + 1, args.epoches):
            if dist.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)
            
            train_stats = train_one_epoch(
                self.model, self.criterion, self.train_dataloader, self.optimizer, self.device, epoch,
                args.clip_max_norm, print_freq=args.log_step, ema=self.ema, scaler=self.scaler)

            self.lr_scheduler.step()
            
            if self.output_dir:
                checkpoint_paths = [self.output_dir / 'checkpoint.pth']
                # extra checkpoint before LR drop and every 100 epochs
                if (epoch + 1) % args.checkpoint_step == 0:
                    checkpoint_paths.append(self.output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    dist.save_on_master(self.state_dict(epoch), checkpoint_path)

            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module, self.criterion, self.postprocessor, self.val_dataloader, base_ds, self.device, self.output_dir
            )

            # TODO 
            for k in test_stats.keys():
                if k in best_stat:
                    best_stat['epoch'] = epoch if test_stats[k][0] > best_stat[k] else best_stat['epoch']
                    best_stat[k] = max(best_stat[k], test_stats[k][0])
                else:
                    best_stat['epoch'] = epoch
                    best_stat[k] = test_stats[k][0]
            print('best_stat: ', best_stat)

            # 若本 epoch 为当前最优，保存为 best 权重
            if self.output_dir and best_stat['epoch'] == epoch:
                dist.save_on_master(self.state_dict(epoch), self.output_dir / 'checkpoint_best.pth')
                print(f'Save best checkpoint at epoch {epoch}')

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch}

            if self.output_dir and dist.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if coco_evaluator is not None:
                    (self.output_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                    self.output_dir / "eval" / name)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))


    def val(self, ):
        self.eval()

        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)
        
        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(module, self.criterion, self.postprocessor,
                self.val_dataloader, base_ds, self.device, self.output_dir)
                
        if self.output_dir:
            dist.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")
        
        return

    def fitX(self, ):
        print("Start training")
        self.train()

        args = self.cfg

        # 打印并记录 FLOPs/Params（必须用 model 而非 ema.module，避免 thop 注入 total_ops 污染 EMA）
        flops_g, params_m = None, None
        try:
            from thop import profile
            module = dist.de_parallel(self.model)
            module.eval()
            h, w = 640, 640
            if hasattr(self.cfg, 'yaml_cfg'):
                es = self.cfg.yaml_cfg.get('HGS_EncoderWrapper', {}).get('eval_spatial_size')
                print(es)
                if es:
                    h, w = es[0], es[1]
            x_ir = torch.randn(1, 3, h, w, device=self.device)
            x_vis = torch.randn(1, 3, h, w, device=self.device)
            with torch.no_grad():
                flops, params = profile(module, inputs=(x_ir, x_vis), verbose=False)
            flops_g = flops / 1024 ** 3
            params_m = params / 1e6
            print(f'FLOPs: {flops_g:.2f} G, Params: {params_m:.2f} M')
            module.train()
        except Exception as e:
            print(f'FLOPs: skip ({e})')
        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)
        # best_stat = {'coco_eval_bbox': 0, 'coco_eval_masks': 0, 'epoch': -1, }
        best_stat = {'epoch': -1, }

        start_time = time.time()
        for epoch in range(self.last_epoch + 1, args.epoches):
            if dist.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)

            train_stats = train_one_epoch_X(
                self.model, self.criterion, self.train_dataloader, self.optimizer, self.device, epoch,
                args.clip_max_norm, print_freq=args.log_step, ema=self.ema, scaler=self.scaler)

            self.lr_scheduler.step()

            if self.output_dir:
                checkpoint_paths = [self.output_dir / 'checkpoint.pth']
                # extra checkpoint before LR drop and every 100 epochs
                if (epoch + 1) % args.checkpoint_step == 0:
                    checkpoint_paths.append(self.output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    dist.save_on_master(self.state_dict(epoch), checkpoint_path)

            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluateX(
                module, self.criterion, self.postprocessor, self.val_dataloader, base_ds, self.device, self.output_dir
            )

            # TODO
            for k in test_stats.keys():
                if k in best_stat:
                    best_stat['epoch'] = epoch if test_stats[k][0] > best_stat[k] else best_stat['epoch']
                    best_stat[k] = max(best_stat[k], test_stats[k][0])
                else:
                    best_stat['epoch'] = epoch
                    best_stat[k] = test_stats[k][0]
            print('best_stat: ', best_stat)

            # 若本 epoch 为当前最优，保存为 best 权重
            if self.output_dir and best_stat['epoch'] == epoch:
                dist.save_on_master(self.state_dict(epoch), self.output_dir / 'checkpoint_best.pth')
                print(f'Save best checkpoint at epoch {epoch}')

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch}
            if flops_g is not None:
                log_stats['flops_g'] = round(flops_g, 4)
            if params_m is not None:
                log_stats['params_m'] = round(params_m, 4)

            if self.output_dir and dist.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if coco_evaluator is not None:
                    (self.output_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                       self.output_dir / "eval" / name)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

    def valX(self, ):

        self.eval()

        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)

        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluateX(module, self.criterion, self.postprocessor,
                                              self.val_dataloader, base_ds, self.device, self.output_dir)

        if self.output_dir:
            dist.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")

        flops_g, params_m = None, None
        try:
            from thop import profile
            module = dist.de_parallel(self.model)
            module.eval()
            h, w = 640, 640
            if hasattr(self.cfg, 'yaml_cfg'):
                es = self.cfg.yaml_cfg.get('HGS_EncoderWrapper', {}).get('eval_spatial_size')
                print(es)
                if es:
                    h, w = es[0], es[1]
            x_ir = torch.randn(1, 3, h, w, device=self.device)
            x_vis = torch.randn(1, 3, h, w, device=self.device)
            with torch.no_grad():
                flops, params = profile(module, inputs=(x_ir, x_vis), verbose=False)
            flops_g = flops / 1024 ** 3
            params_m = params / 1e6
            print(f'FLOPs: {flops_g:.2f} G, Params: {params_m:.2f} M')
            module.train()
        except Exception as e:
            print(f'FLOPs: skip ({e})')
        return
