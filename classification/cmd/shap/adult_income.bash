CUDA_VISIBLE_DEVICES=4 python classification/shap_plot.py \
    --dataset='adult_income' \
    --data_path "/workspace/tripx/MCS/xai_causality/dataset/adult_income/norm_adult_income.csv" \
    --save_shap_path='/workspace/binhtlh/projects/causality/XAI_using_Structure_Learning/classification/run/shap/adult_income/'