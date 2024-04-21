
CUDA_VISIBLE_DEVICES=4 python regression/shap_plot.py \
    --dataset='diabetes' \
    --data_path='/workspace/tripx/MCS/xai_causality/dataset/new_scaled_diabetes.csv' \
    --save_shap_path='/workspace/tripx/MCS/xai_causality/regression/run/shap/diabetes'