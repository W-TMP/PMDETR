export CUDA_VISIBLE_DEVICES=2

# python tools/train_X.py -c configs/gmdetr/gmdetr_r50vd_6x_vedai_1024_X_potential_train2.yml

python tools/train_X.py -c configs/gmdetr/gmdetr_r50vd_6x_llvip_1024_X_potential_train2.yml

# python tools/train_X.py -c configs/gmdetr/gmdetr_r50vd_6x_llvip_1024_X_potential_train2.yml -r output/gmdetr_r50vd_6x_llvip_1024_X_potential_train2/checkpoint0070.pth --test-only