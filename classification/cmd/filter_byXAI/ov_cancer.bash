python classification/filter_byXAI.py \
    --dataset='ov_cancer' \
    --graph_path='/workspace/tripx/MCS/xai_causality/run_v2/ov_cancer/avg_seed/avg_dag_W_est.csv' \
    --data_path='/dataset/PANCAN/OV_gene_filter.csv' \
    --save_pdp_path='/workspace/binhtlh/projects/causality/XAI_using_Structure_Learning/classification/run/pdp/ov_cancer/' \
    > classification/run/filter_byXAI/ov_cancer.txt 