CUDA_VISIBLE_DEVICES=4 python regression/shap_plot.py \
    --dataset='cali_housing' \
    --data_path='/workspace/tripx/MCS/xai_causality/dataset/scaled_cali_housing.csv' \
    --save_shap_path='/workspace/binhtlh/projects/causality/XAI_using_Structure_Learning/regression/run/shap/cali_housing/'