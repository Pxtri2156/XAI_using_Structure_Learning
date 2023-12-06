python classification/filter_byXAI.py \
    --dataset='breast_cancer' \
    --graph_path='/workspace/tripx/MCS/xai_causality/run_v2/breast_cancer/avg_seed/avg_dag_W_est.csv' \
    --data_path='/workspace/tripx/MCS/xai_causality/dataset/breast_cancer_uci.csv' \
    --save_pdp_path='/workspace/binhtlh/projects/causality/XAI_using_Structure_Learning/classification/run/pdp/breast_cancer/' \
    > classification/run/filter_byXAI/breast_cancer.txt 