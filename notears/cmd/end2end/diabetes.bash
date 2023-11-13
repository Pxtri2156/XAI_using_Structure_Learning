python notears/xai_reg.py \
    --data_name='diabetes' \
    --root_path='/workspace/tripx/MCS/xai_causality/run_v2/diabetes/' \
    --data_path='/workspace/tripx/MCS/xai_causality/dataset/scaled_diabetes.csv' \
    --wandb_mode='disabled' \
    --lambda_reg=2 \
    --lambda1=0.0002 \
    --lambda2=0.01 \
    --no_seeds=20

python notears/average_graph.py \
    --data_name='diabetes' \
    --root_path='/workspace/tripx/MCS/xai_causality/run_v2/diabetes/' 

