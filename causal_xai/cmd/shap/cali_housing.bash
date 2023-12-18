python notears/xai_shap_notear_reg.py \
    --data_name='cali_housing' \
    --root_path='/workspace/tripx/MCS/xai_causality/run_shap/cali_housing/' \
    --data_path='/workspace/tripx/MCS/xai_causality/dataset/scaled_cali_housing.csv' \
    --wandb_mode='disabled' \
    --lambda_reg=2 \
    --lambda1=0.0001 \
    --lambda2=0.0005 \
    --no_seeds=1