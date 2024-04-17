python causal_xai/notears/xai_reg.py \
    --data_name='boston_housing_v2' \
    --root_path='/workspace/binhtlh/projects/causality/XAI_using_Structure_Learning/demo_site/run/reg/' \
    --data_path='/workspace/tripx/MCS/xai_causality/dataset/new_scale_boston_v2.csv' \
    --wandb_mode='online' \
    --lambda_reg=3 \
    --lambda1=0.0004 \
    --lambda2=0.0005 \
    --no_seeds=20