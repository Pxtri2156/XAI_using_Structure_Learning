python notears/xai_cls.py \
    --data_name='parkinson'\
    --root_path='/workspace/tripx/MCS/xai_causality/run/run_v2/parkinson/' \
    --data_path='/dataset/Parkinson_Disease/Parkinson_Disease_GC_MS.csv' \
    --wandb_mode='online' \
    --lambda_cls=3 \
    --lambda1=0.002 \
    --lambda2=0.005 \
    --ratio_test=0.1 \
    --no_seeds=1