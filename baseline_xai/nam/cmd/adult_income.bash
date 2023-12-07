CUDA_VISIBLE_DEVICES=4 python xai_nam.py \
    --data_name='adult_income' \
    --root_path='/workspace/tripx/MCS/xai_causality/nam_run/adult_income/' \
    --data_path='/workspace/tripx/MCS/xai_causality/dataset/adult_income/norm_adult_income.csv' \
    --wandb_mode='disabled' \
    --no_seeds=20 \
    --epochs=50