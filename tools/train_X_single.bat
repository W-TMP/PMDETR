@echo off
REM 单卡训练 - Windows
set CUDA_VISIBLE_DEVICES=0
python tools/train_X.py -c configs/gmdetr/gmdetr_r50vd_6x_align_flir_X_train2.yml
