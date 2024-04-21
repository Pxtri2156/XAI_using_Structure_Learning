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
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
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
    model_list = []
    if dataset_name == 'boston_housing_v2': 
        # model_list = [LinearRegression(fit_intercept=False),
        #         Ridge(alpha=1, positive=True), 
        #         Lasso(alpha=1, positive=True),
        #         ElasticNet(alpha=1, l1_ratio=0.0),
        #         svm.SVR(C=3, epsilon=0.0, gamma='auto'), 
        #         DecisionTreeRegressor(max_leaf_nodes=50, min_samples_leaf=2, min_samples_split=5, random_state=42, splitter='random'), 
        #         RandomForestRegressor(n_estimators=10)]
        model_list = [svm.SVR(C=10, gamma=1)]
    elif dataset_name == 'cali_housing': 
        # model_list = [LinearRegression(fit_intercept=False),
        #         Ridge(alpha=1, positive=True, fit_intercept=False), 
        #         Lasso(alpha=1, positive=True),
        #         ElasticNet(alpha=1, l1_ratio=0.0),
        #         svm.SVR(C=2), 
        #         DecisionTreeRegressor(max_leaf_nodes=50, min_samples_leaf=4,min_samples_split=5), 
        #         RandomForestRegressor()]
        model_list = [svm.SVR(C=10, gamma=1)]

    elif dataset_name == 'diabetes': 
        # model_list = [LinearRegression(),
        #         Ridge(alpha=1, positive=True), 
        #         Lasso(alpha=1, positive=True),
        #         ElasticNet(alpha=1, l1_ratio=0.9),
        #         svm.SVR(C=9, epsilon=0.6000000000000001, kernel='sigmoid'), 
        #         DecisionTreeRegressor(ccp_alpha=0.2, max_depth=20, max_leaf_nodes=50,
        #               min_impurity_decrease=0.1, min_samples_leaf=2,
        #               min_samples_split=10, min_weight_fraction_leaf=0.1,
        #               splitter='random'), 
        #         RandomForestRegressor()]
        model_list = [svm.SVR(C=10, gamma=1)]

    elif dataset_name == 'fried_man1': 
        # model_list = [LinearRegression(positive=True),
        #         Ridge(alpha=1, positive=True), 
        #         Lasso(alpha=1, positive=True, fit_intercept=False),
        #         ElasticNet(alpha=1, fit_intercept=False, l1_ratio=0.9, positive=True),
        #         svm.SVR(C=10, epsilon=0.0), 
        #         DecisionTreeRegressor(ccp_alpha=0.1, max_depth=30, max_leaf_nodes=50,
        #               min_samples_leaf=4, min_samples_split=5,
        #               splitter='random'), 
        #         RandomForestRegressor()]
        model_list = [svm.SVR(C=10, gamma=1)]
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
    if args.dataset == 'cali_housing':
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