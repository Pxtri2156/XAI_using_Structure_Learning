CUDA_VISIBLE_DEVICES=5 python xai_nam.py \
    --data_name='ov_cancer' \
    --root_path='/workspace/tripx/MCS/xai_causality/nam_run/ov_cancer/' \
    --data_path='/dataset/PANCAN/OV_gene_filter.csv' \
    --wandb_mode='disabled' \
    --no_seeds=20 \
    --epochs=50