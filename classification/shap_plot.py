import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import shap
import random
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import argparse
import os
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def loadData(data_path): 
    data = pd.read_csv(data_path)
    column_name = list(data.columns)
    if 'Unnamed: 0' in column_name: 
        column_name.remove('Unnamed: 0')
    X = data[column_name[:-1]]
    y = data[column_name[-1]]
    return X, y

def model_list_by_dataset(dataset_name): 
    if dataset_name == 'adult_income': 
        model_list = [svm.SVC(), RandomForestClassifier()]
    elif dataset_name == 'breast_cancer': 
        model_list = [svm.SVC(), RandomForestClassifier()]
    elif dataset_name == 'laml_cancer': 
        model_list = [svm.SVC(), RandomForestClassifier()]
    elif dataset_name == 'ov_cancer': 
        model_list = [svm.SVC(C=10, decision_function_shape="ovr", kernel="rbf", random_state=100),
                    RandomForestClassifier(criterion='entropy', n_estimators=10, n_jobs=-1)]
    else: 
        print('Model list problem!')
    return model_list

def shap_plots(X, y, model_list, save_path): 
    for model in model_list:
        model.fit(X, y)
        X100 = shap.utils.sample(X)
        explainer = shap.Explainer(model.predict, X100)
        shap_values = explainer(X)
        print(shap_values.shape)
        shap.plots.bar(shap_values, show=False)
        file_name = str(model).split('(')[0] + '.pdf'
        plt.savefig(os.path.join(save_path,file_name), bbox_inches='tight')
        plt.clf()
def main(args):
    # filter dataset by XAI
    X, y = loadData(args.data_path)
    if args.dataset == 'adult_income':
        X, y = X[:int(0.1*len(X))], y[:int(0.1*len(y))]
    model_list = model_list_by_dataset(args.dataset)
    shap_plots(X, y, model_list, args.save_shap_path)
    print(f'SUCCESSFULLY!')


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
    parser.add_argument("--save_shap_path", 
                        default="/workspace/binhtlh/projects/causality/XAI_using_Structure_Learning/regression/run/shap/boston_housing/", 
                        type=str)
    return parser.parse_args()

if __name__ == "__main__":
    args = arg_parser()
    main(args)