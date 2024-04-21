CUDA_VISIBLE_DEVICES=4 python xai_nam.py \
    --data_name='fried_man' \
    --root_path='/workspace/tripx/MCS/xai_causality/run/nam_run_v3/friedman1/' \
    --data_path='/workspace/tripx/MCS/xai_causality/dataset/new_freidman1.csv' \
    --wandb_mode='disabled' \
    --no_seeds=20 \
    --regression=True \
    --epochs=50