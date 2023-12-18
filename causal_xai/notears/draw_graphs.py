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
    
    thresholds = 0.3 
    # Read labels 
    labels = ut.get_labels(args.data_name)
    # Get avarage W_est from random seeds 
    ut.draw_seed_graphs(args.root_path, args.w_threshold, args.no_seeds, labels)
        
 
def arg_parser():
    parser = argparse.ArgumentParser()
    # Path
    parser.add_argument("--root_path", 
                        default="/workspace/tripx/MCS/xai_causality/run_v2/ov_cancer/", 
                        type=str)
    parser.add_argument("--data_name", 
                        default="ov_cancer", 
                        type=str)
    
    # Training
    parser.add_argument("--no_seeds", 
                        default=20, 
                        type=int,
                        help='Number of random seed')
    parser.add_argument("--w_threshold", 
                        default=0.3, 
                        type=float,
                        help='Threshold to be DAG')
    return parser.parse_args()
   
if __name__ == '__main__':
    args = arg_parser()
    main(args)
