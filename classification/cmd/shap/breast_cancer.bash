CUDA_VISIBLE_DEVICES=4 python classification/shap_plot.py \
    --dataset='breast_cancer' \
    --data_path "/workspace/tripx/MCS/xai_causality/dataset/breast_cancer_uci.csv" \
    --save_shap_path='/workspace/binhtlh/projects/causality/XAI_using_Structure_Learning/classification/run/shap/breast_cancer/'