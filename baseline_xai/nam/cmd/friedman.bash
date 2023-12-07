CUDA_VISIBLE_DEVICES=4 python xai_nam.py \
    --data_name='fried_man' \
    --root_path='/workspace/tripx/MCS/xai_causality/nam_run/fried_man/' \
    --data_path='/workspace/tripx/MCS/xai_causality/dataset/scaled_friedman1.csv' \
    --wandb_mode='disabled' \
    --no_seeds=20 \
    --regression=True \
    --epochs=50