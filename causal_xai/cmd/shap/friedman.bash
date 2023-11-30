python notears/xai_shap_notear_reg.py \
    --data_name='friedman1' \
    --root_path='/workspace/tripx/MCS/xai_causality/run_shap/fried_man1/' \
    --data_path='/workspace/tripx/MCS/xai_causality/dataset/scaled_friedman1.csv' \
    --wandb_mode='disabled' \
    --lambda_reg=2 \
    --lambda1=0.0002 \
    --lambda2=0.01 \
    --no_seeds=1