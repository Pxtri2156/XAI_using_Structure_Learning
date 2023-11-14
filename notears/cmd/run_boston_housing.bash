python notears/xai_reg.py \
    --data_name='boston_housing' \
    --root_path='/workspace/tripx/MCS/xai_causality/run_tuning/boston_housing' \
    --data_path='/workspace/tripx/MCS/xai_causality/dataset/scaled_boston_housing.csv' \
    --wandb_mode='disabled' \
    --lambda_reg=5 \
    --lambda1=0.005 \
    --lambda2=0.0025 \
    --no_seeds=20