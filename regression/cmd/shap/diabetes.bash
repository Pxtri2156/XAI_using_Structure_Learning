CUDA_VISIBLE_DEVICES=4 python regression/shap_plot.py \
    --dataset='diabetes' \
    --data_path='/workspace/tripx/MCS/xai_causality/dataset/scaled_diabetes.csv' \
    --save_shap_path='/workspace/binhtlh/projects/causality/XAI_using_Structure_Learning/regression/run/shap/diabetes/'