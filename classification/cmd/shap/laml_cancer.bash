CUDA_VISIBLE_DEVICES=4 python classification/shap_plot.py \
    --dataset='laml_cancer' \
    --data_path "/dataset/PANCAN/LAML_gene_filter.csv" \
    --save_shap_path='/workspace/binhtlh/projects/causality/XAI_using_Structure_Learning/classification/run/shap/laml_cancer/'