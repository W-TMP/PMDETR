# PMDETR

Official implementation of **PMDETR: Potential Map-Guided Detection Transformer for Infrared-Visible Small-Object Detection**.

PMDETR combines shallow infrared-visible feature fusion with potential map-guided feature enhancement, sparse self-attention, and cross-level propagation.

## Installation

The project uses PyTorch and torchvision. Install versions compatible with your CUDA environment, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```



## Dataset Preparation

PMDETR uses paired infrared and visible images with matching filenames and COCO-format annotations. The loader reads visible images from `vis_imgs/` and the corresponding infrared images from the sibling `ir_imgs/` folder.

```text
dataset/
  train_imgs/
    vis_imgs/
    ir_imgs/
  val_imgs/
    vis_imgs/
    ir_imgs/
  annotations/
    train.json
    val.json
```

Set the dataset image and annotation paths (`img_folder` and `ann_file`) in the dataset YAML referenced by the model configuration. For example, the FLIR configurations use `configs/dataset/align_flir_X_ir_detection_train1.yml` or `configs/dataset/align_flir_X_ir_detection_train2.yml`.

## Training

Run commands from the repository root. To train on one GPU:

```bash
export CUDA_VISIBLE_DEVICES=0
python tools/train_X.py \
  -c configs/gmdetr/gmdetr_r101vd_6x_align_flir_X_train2.yml
```

To train on two GPUs:

```bash
export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 tools/train_X.py \
  -c configs/gmdetr/gmdetr_r50vd_6x_align_flir_X_train1.yml
```

To continue training from a checkpoint, provide it with `-r`:

```bash
torchrun --nproc_per_node=2 tools/train_X.py \
  -c configs/gmdetr/gmdetr_r50vd_6x_align_flir_X_train2.yml \
  -r output/gmdetr_r50vd_6x_align_flir_X_train1/checkpoint_best.pth
```

Training settings, dataset paths, and output directory are specified in the YAML configuration.

## Evaluation

Evaluate a trained checkpoint on the validation split specified by the config:

```bash
export CUDA_VISIBLE_DEVICES=0
python tools/train.py \
  -c configs/gmdetr/gmdetr_r101vd_6x_align_flir_X_train2.yml \
  -r path/to/checkpoint.pth \
  --test-only
```

## Repository Structure

- `configs/`: model and dataset configurations.
- `src/`: model, data pipeline, losses, and training utilities.
- `tools/train_X.py`: paired infrared-visible training entry point.
- `tools/train.py`: training and checkpoint evaluation entry point.
- `tools/eval_X.py`: validation using the best checkpoint in the configured output directory.
- `tools/infer_X.py`: prediction visualization for paired images.
