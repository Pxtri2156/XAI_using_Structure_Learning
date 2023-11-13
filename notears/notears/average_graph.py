import  sys
sys.path.append("./")
import os
import argparse
import numpy as np
import pandas as pd
import notears.utils as ut
from sklearn.metrics import classification_report, f1_score, accuracy_score
import networkx as nx 
import matplotlib.pyplot as plt

def main(args):
    np.set_printoptions(precision=3)
    
    # thresholds_lst = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
    # Read labels 
    labels = ut.get_labels(args.data_name)
    # Get avarage W_est from random seeds 
    W_est_list = []
    for seed in range(args.no_seeds):
        seed_name = 'seed_'+str(seed)
        out_folder = args.root_path + f"{seed_name}"
        if not os.path.isdir(out_folder):
            os.mkdir(out_folder)
        out_folder += "/"
        seed_result_path = out_folder + 'W_est.csv'
        ### Continue
        seed_W_est = np.loadtxt(seed_result_path, delimiter=",", dtype=float)
        W_est_list.append(seed_W_est)
    W_est_list = np.array(W_est_list)
    
    mean_W_est =  W_est_list.mean(axis=0)
    std_W_est = W_est_list.std(axis=0)
    
    # Calculate avg DAG with thresholds
    avg_seed_path =  args.root_path + f"avg_seed"
    if not os.path.isdir(avg_seed_path):
        os.mkdir(avg_seed_path)
    avg_seed_path += '/'
    ut.draw_head_map(std_W_est, avg_seed_path + f'std_array.png')
    ut.draw_head_map(mean_W_est, avg_seed_path + f'mean_array.png')
    
    
    while(not ut.is_dag(mean_W_est)):
        min_val = np.unique(mean_W_est)[1]
        mean_W_est[mean_W_est == min_val] = 0
        print(f'min_val: {min_val}')
        
    ## Save results with threshold 
    np.savetxt(avg_seed_path + f'avg_dag_W_est.csv', mean_W_est, delimiter=',')

    ## Visualize graph 
    vis_path = avg_seed_path + f"vis_avg_dag_graph.png"
    ut.draw_directed_graph(mean_W_est, vis_path, labels) 
    ut.draw_head_map(mean_W_est, avg_seed_path + f'mean_dag_array.png')

    # for w_threshold in thresholds_lst:
    #     tmp_W_est = mean_W_est.copy()
    #     tmp_W_est[np.abs(tmp_W_est) < w_threshold] = 0
    #     try: 
    #         assert ut.is_dag(tmp_W_est)
    #     except:
    #         print(f"Skip threshold {w_threshold}")
    #         continue

    #     ## Save results with threshold 
    #     np.savetxt(avg_seed_path + f'avg_W_est_{str(w_threshold)}.csv', tmp_W_est, delimiter=',')

    #     ## Visualize graph 
    #     vis_path = avg_seed_path + f"vis_avg_graph_{str(w_threshold)}.png"
    #     ut.draw_directed_graph(tmp_W_est, vis_path, labels) 
        
        
 
def arg_parser():
    parser = argparse.ArgumentParser()
    # path
    parser.add_argument("--root_path", 
                        default="/workspace/tripx/MCS/xai_causality/run_v2/ov_cancer/", 
                        type=str)
    parser.add_argument("--data_name", 
                        default="ov_cancer", 
                        type=str)
    # training
    parser.add_argument("--no_seeds", 
                        default=20, 
                        type=int,
                        help='Number of random seed')

    return parser.parse_args()
   
if __name__ == '__main__':
    args = arg_parser()
    main(args)
