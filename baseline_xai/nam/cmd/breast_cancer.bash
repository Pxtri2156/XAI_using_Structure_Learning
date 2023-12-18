CUDA_VISIBLE_DEVICES=5 python xai_nam.py \
    --data_name='breast_cancer' \
    --root_path='/workspace/tripx/MCS/xai_causality/nam_run/breast_cancer/' \
    --data_path='/workspace/tripx/MCS/xai_causality/dataset/norm_breast_cancer_uci.csv' \
    --wandb_mode='disabled' \
    --no_seeds=20 \
    --epochs=50