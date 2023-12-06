import argparse

def main(args):
    print(args.data_path)
    return 


def arg_parser():
    parser = argparse.ArgumentParser()
    # path
    parser.add_argument("--dataset", 
                        default="boston_housing", 
                        type=str),
    parser.add_argument("--graph_path", 
                        default="/workspace/tripx/MCS/xai_causality/run_v2/boston_housing/avg_seed/avg_dag_W_est.csv", 
                        type=str)
    parser.add_argument("--data_path", 
                        default="/workspace/tripx/MCS/xai_causality/dataset/scaled_boston_housing.csv", 
                        type=str)
    parser.add_argument("--save_pdp_path", 
                        default="/workspace/binhtlh/projects/causality/XAI_using_Structure_Learning/regression/run/pdp/boston_housing/", 
                        type=str)
    return parser.parse_args()

if __name__ == "__main__":
    args = arg_parser()
    main(args)