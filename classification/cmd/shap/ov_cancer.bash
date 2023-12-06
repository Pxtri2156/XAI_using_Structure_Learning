CUDA_VISIBLE_DEVICES=4 python classification/shap_plot.py \
    --dataset='ov_cancer' \
    --data_path "/dataset/PANCAN/OV_gene_filter.csv" \
    --save_shap_path='/workspace/binhtlh/projects/causality/XAI_using_Structure_Learning/classification/run/shap/ov_cancer/'