import  sys
sys.path.append("./")
import notears.utils as ut
import numpy as np
import argparse
import os
import json

def main(args):
    # Example usage:
    avg_path = args.root_path + f"avg_seed/avg_dag_W_est_{args.data_name}.csv"
    adjacency_matrix = np.loadtxt(avg_path, delimiter=",", dtype=float)
    
    adjacency_matrix[np.abs(adjacency_matrix) > 0] = 1
    # print("adjacency_matrix: ", adjacency_matrix)
    contribution_dic = ut.get_contribution(adjacency_matrix)
    # json_object = json.dumps(contribution_dic, indent = 4) 

    contribution_path = args.root_path + f"avg_seed/contribution_{args.data_name}.json"
    with open(contribution_path, "w") as outfile: 
        json.dump(contribution_dic, outfile)


def arg_parser():
    parser = argparse.ArgumentParser()
    # path
    parser.add_argument("--root_path", 
                        default="/workspace/tripx/MCS/xai_causality/run/run_v2/ov_cancer/", 
                        type=str)
    parser.add_argument("--data_name", 
                        default="ov_cancer", 
                        type=str)
    return parser.parse_args()

if __name__ == '__main__':
    args = arg_parser()
    main(args)