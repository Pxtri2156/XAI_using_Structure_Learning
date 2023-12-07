python regression/filter_byXAI.py \
    --dataset='cali_housing' \
    --graph_path='/workspace/tripx/MCS/xai_causality/run_tuning/cali_housing/avg_seed/avg_dag_W_est.csv' \
    --data_path='/workspace/tripx/MCS/xai_causality/dataset/scaled_cali_housing.csv' \
    --save_pdp_path='/workspace/binhtlh/projects/causality/XAI_using_Structure_Learning/regression/run/pdp/cali_housing/' \
    > regression/run/filter_byXAI/cali_housing.txt 