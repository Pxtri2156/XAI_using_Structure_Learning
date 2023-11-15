python classification/filter_byXAI.py \
    --dataset='adult_income' \
    --graph_path='/workspace/tripx/MCS/xai_causality/run_v2/adult_income/avg_seed/avg_dag_W_est.csv' \
    --data_path='/workspace/tripx/MCS/xai_causality/dataset/selected_adult_income.csv' \
    --save_pdp_path='/workspace/binhtlh/projects/causality/XAI_using_Structure_Learning/classification/run/pdp/adult_income/' \
    > classification/run/filter_byXAI/adult_income.txt 