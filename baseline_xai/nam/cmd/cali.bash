CUDA_VISIBLE_DEVICES=4 python xai_nam.py \
    --data_name='cali_housing' \
    --root_path='/workspace/tripx/MCS/xai_causality/nam_run/cali_housing/' \
    --data_path='/workspace/tripx/MCS/xai_causality/dataset/scaled_cali_housing.csv' \
    --wandb_mode='disabled' \
    --no_seeds=20 \
    --regression=True \
    --epochs=300