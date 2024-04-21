CUDA_VISIBLE_DEVICES=1 python xai_nam.py \
    --data_name='boston_housing' \
    --root_path='/workspace/tripx/MCS/xai_causality/run/nam_run_v3/boston_housing/' \
    --data_path='/workspace/tripx/MCS/xai_causality/dataset/new_scale_boston_v2.csv' \
    --wandb_mode='disabled' \
    --no_seeds=20 \
    --regression=True \
    --epochs=400