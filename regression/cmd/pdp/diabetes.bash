python regression/filter_byXAI.py \
    --dataset='diabetes' \
    --graph_path='/workspace/tripx/MCS/xai_causality/run_v2/diabetes/avg_seed/avg_dag_W_est.csv' \
    --data_path='/workspace/tripx/MCS/xai_causality/dataset/scaled_diabetes.csv' \
    --save_pdp_path='/workspace/binhtlh/projects/causality/XAI_using_Structure_Learning/regression/run/pdp/diabetes/' \
    > regression/run/filter_byXAI/diabetes.txt 