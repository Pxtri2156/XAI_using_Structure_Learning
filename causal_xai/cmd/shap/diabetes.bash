python notears/xai_shap_notear_reg.py \
    --data_name='diabetes' \
    --root_path='/workspace/tripx/MCS/xai_causality/run_shap/diabetes/' \
    --data_path='/workspace/tripx/MCS/xai_causality/dataset/scaled_diabetes.csv' \
    --wandb_mode='disabled' \
    --lambda_reg=2 \
    --lambda1=0.002 \
    --lambda2=0.01 \
    --no_seeds=1