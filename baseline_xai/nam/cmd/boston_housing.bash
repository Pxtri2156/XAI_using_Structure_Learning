CUDA_VISIBLE_DEVICES=6 python xai_nam.py \
    --data_name='boston_housing' \
    --root_path='/workspace/tripx/MCS/xai_causality/run/nam_run_v2/boston_housing/' \
    --data_path='/workspace/tripx/MCS/xai_causality/dataset/scaled_boston_housing.csv' \
    --wandb_mode='disabled' \
    --no_seeds=20 \
    --regression=True \
    --epochs=50