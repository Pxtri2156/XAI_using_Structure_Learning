CUDA_VISIBLE_DEVICES=5 python xai_nam.py \
    --data_name='diabetes' \
    --root_path='/workspace/tripx/MCS/xai_causality/run/nam_run_v2/diabetes/' \
    --data_path='/workspace/tripx/MCS/xai_causality/dataset/scaled_diabetes.csv' \
    --wandb_mode='disabled' \
    --no_seeds=20 \
    --regression=True \
    --epochs=50


