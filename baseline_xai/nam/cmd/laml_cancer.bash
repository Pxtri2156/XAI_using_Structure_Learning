CUDA_VISIBLE_DEVICES=5 python xai_nam.py \
    --data_name='laml_cancer' \
    --root_path='/workspace/tripx/MCS/xai_causality/nam_run/laml_cancer/' \
    --data_path='/dataset/PANCAN/LAML_gene_filter.csv' \
    --wandb_mode='disabled' \
    --no_seeds=20 \
    --epochs=50