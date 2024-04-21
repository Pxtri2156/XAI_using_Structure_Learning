python causal_xai/notears/xai_cls.py \
    --data_name='laml' \
    --root_path='/workspace/binhtlh/projects/causality/XAI_using_Structure_Learning/demo_site/run/cls/' \
    --data_path='/dataset/PANCAN/LAML_gene_filter.csv' \
    --wandb_mode='online' \
    --lambda_cls=2 \
    --lambda1=0.0005 \
    --lambda2=0.0005 \
    --no_seeds=20