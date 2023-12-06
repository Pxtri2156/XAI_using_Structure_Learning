python regression/filter_byXAI.py \
    --dataset='boston_housing' \
    --graph_path='/workspace/tripx/MCS/xai_causality/run_v2/boston_housing/avg_seed/avg_dag_W_est.csv' \
    --data_path='/workspace/tripx/MCS/xai_causality/dataset/scaled_boston_housing.csv' \
    --save_pdp_path='/workspace/binhtlh/projects/causality/XAI_using_Structure_Learning/regression/run/pdp/boston_housing/' \
    > regression/run/filter_byXAI/boston_housing.txt 