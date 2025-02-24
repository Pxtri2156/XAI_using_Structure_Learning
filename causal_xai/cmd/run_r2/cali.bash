python notears/xai_reg.py \
    --data_name='cali_housing' \
    --root_path='/workspace/tripx/MCS/xai_causality/run/run_v4/cali_housing/' \
    --data_path='/workspace/tripx/MCS/xai_causality/dataset/scaled_cali_housing.csv' \
    --wandb_mode='online' \
    --lambda_reg=2 \
    --lambda1=0.0005 \
    --lambda2=0.0005 \
    --no_seeds=20