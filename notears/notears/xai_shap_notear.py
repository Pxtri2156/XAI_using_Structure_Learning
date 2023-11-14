import  sys
sys.path.append("./")
import os
import argparse
import json
import math
import wandb
import torch
import numpy as np
import pandas as pd
import argparse
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, accuracy_score
import shap
import matplotlib.pyplot as plt

from notears.locally_connected import LocallyConnected
from notears.lbfgsb_scipy import LBFGSBScipy
from notears.trace_expm import trace_expm

import notears.utils as ut
import networkx as nx 
import matplotlib.pyplot as plt

model = None

class NotearsMLP(nn.Module):
    def __init__(self, dims, bias=True):
        super(NotearsMLP, self).__init__()
        assert len(dims) >= 2
        assert dims[-1] == 1
        d = dims[0]
        self.dims = dims
        # fc1: variable splitting for l1
        self.fc1_pos = nn.Linear(d, d * dims[1], bias=bias)
        self.fc1_neg = nn.Linear(d, d * dims[1], bias=bias)
        self.fc1_pos.weight.bounds = self._bounds()
        self.fc1_neg.weight.bounds = self._bounds()
        layers = []
        for l in range(len(dims) - 2):
            layers.append(LocallyConnected(d, dims[l + 1], dims[l + 2], bias=bias))
        self.fc2 = nn.ModuleList(layers)
        
    def _bounds(self):
        d = self.dims[0]
        bounds = []
        for j in range(d):
            for m in range(self.dims[1]):
                for i in range(d):
                    if i != d-1:
                    # if True:
                        if i == j:
                            bound = (0, 0)
                        else:
                            bound = (0, None)
                    else:
                        bound = (0, 0) 
                    bounds.append(bound)

        # constraint y=0 (chú ý số chiều)
        return bounds

    def forward(self, x):  # [n, d] -> [n, d]
        x = self.fc1_pos(x) - self.fc1_neg(x)  # [n, d * m1]
        x = x.view(-1, self.dims[0], self.dims[1])  # [n, d, m1]
        for fc in self.fc2:
            x = torch.sigmoid(x)  # [n, d, m1]
            x = fc(x)  # [n, d, m2]
        x = x.squeeze(dim=2)  # [n, d]
        return x

    def h_func(self):
        """Constrain 2-norm-squared of fc1 weights along m1 dim to be a DAG"""
        d = self.dims[0]
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        fc1_weight = fc1_weight.view(d, -1, d)  # [j, m1, i]
        A = torch.sum(fc1_weight * fc1_weight, dim=1).t()  # [i, j]
        h = trace_expm(A) - d  # (Zheng et al. 2018)
        # A different formulation, slightly faster at the cost of numerical stability
        # M = torch.eye(d) + A / d  # (Yu et al. 2019)
        # E = torch.matrix_power(M, d - 1)
        # h = (E.t() * M).sum() - d
        return h

    def l2_reg(self):
        """Take 2-norm-squared of all parameters"""
        reg = 0.
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        reg += torch.sum(fc1_weight ** 2)
        for fc in self.fc2:
            reg += torch.sum(fc.weight ** 2)
        return reg

    def fc1_l1_reg(self):
        """Take l1 norm of fc1 weight"""
        reg = torch.sum(self.fc1_pos.weight + self.fc1_neg.weight)
        return reg

    @torch.no_grad()
    def fc1_to_adj(self) -> np.ndarray:  # [j * m1, i] -> [i, j]
        """Get W from fc1 weights, take 2-norm over m1 dim"""
        d = self.dims[0]
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        # print(fc1_weight)
        fc1_weight = fc1_weight.view(d, -1, d)  # [j, m1, i]
        # print(fc1_weight)
        A = torch.sum(fc1_weight * fc1_weight, dim=1).t()  # [i, j]
        W = torch.sqrt(A)  # [i, j]
        W = W.cpu().detach().numpy()  # [i, j]
        return W

def squared_loss(output, target):
    # Use weight
    n = target.shape[0]
    loss = 0.5 / n * torch.sum((output[:,:-1] - target[:,:-1]) ** 2)
    return loss

def cal_reg_loss(output, target):
    n = target.shape[0]
    # Mean Squared Error 
    loss = 0.5 / n * torch.sum((output[:,-1] - target[:,-1]) ** 2)
    return loss

def pred_reg(X_test, model):
    X_torch = torch.from_numpy(X_test)
    X_hat = model(X_torch)
    cls_predict = X_hat[:,-1]
    target = X_test[:,-1]
    return cls_predict.detach().numpy(), target

def dual_ascent_step(model, X, wandb, lambda_reg, lambda1, lambda2, rho, alpha, h, rho_max):
    """Perform one step of dual ascent in augmented Lagrangian."""
    h_new = None
    optimizer = LBFGSBScipy(model.parameters())
    X_torch = torch.from_numpy(X)
    while rho < rho_max:
        def closure():
            optimizer.zero_grad()
            X_hat = model(X_torch)
            causal_loss = squared_loss(X_hat, X_torch)
            reg_loss = cal_reg_loss(X_hat, X_torch)
            reg_loss = lambda_reg*reg_loss
            
            # print('cls_loss: ', cls_loss)
            loss = causal_loss + reg_loss
            h_val = model.h_func()
            penalty = 0.5 * rho * h_val * h_val + alpha * h_val
            l2_reg = 0.5 * lambda2 * model.l2_reg()
            l1_reg = lambda1 * model.fc1_l1_reg()
            primal_obj = loss + penalty + l2_reg + l1_reg
            result_dict = {'obj_func': primal_obj, 
                            'sq_loss': loss, 
                            'causal_loss': causal_loss,
                            'reg_loss': reg_loss,
                            'penalty': penalty,
                            'h_func': h_val.item(), 
                            'l2_reg': l2_reg,
                            'l1_reg': l1_reg}
            wandb.log(result_dict)
            primal_obj.backward()
            return primal_obj
        optimizer.step(closure)  # NOTE: updates model in-place
        with torch.no_grad():
            h_new = model.h_func().item()
        if h_new > 0.25 * h:
            rho *= 10
        else:
            break
    alpha += rho * h_new
    return rho, alpha, h_new

def notears_nonlinear(model: nn.Module,
                      X: np.ndarray,
                      wandb, 
                      lambda_reg: float = 1.2,
                      lambda1: float = 0.,
                      lambda2: float = 0.,
                      max_iter: int = 100,
                      h_tol: float = 1e-8,
                      rho_max: float = 1e+16,
                      w_threshold: float = 0.3):
    rho, alpha, h = 1.0, 0.0, np.inf
    for _ in range(max_iter):
        rho, alpha, h = dual_ascent_step(model, X, wandb, lambda_reg, lambda1, lambda2,
                                         rho, alpha, h, rho_max)
        if h <= h_tol or rho >= rho_max:
            break
    W_est = model.fc1_to_adj()
    # W_est[np.abs(W_est) < w_threshold] = 0
    return W_est, model

def notear_predict(X_train):
    global model
    X_torch = torch.from_numpy(X_train)
    X_hat = model(X_torch)
    cls_predict = X_hat[:,-1]
    return cls_predict

def main(args):
    global model
    # Read and process data
    X = pd.read_csv(args.data_path)
    copy_X = X
    explained_data = copy_X.drop(['Unnamed: 0'], axis=1)
    
    X = X.to_numpy()[:,1:].astype(float)
    np.random.shuffle(X)
    split_n = int((1-args.ratio_test)*X.shape[0])
    X_train = X[:split_n,:]
    X_test = X[split_n:, :]
    d = X.shape[1]
    
    print("X_train: ", X_train.shape)
    # Modeling
    model = NotearsMLP(dims=[d, 10, 1], bias=True)
    W_est, model = notears_nonlinear(model, X_train, wandb, args.lambda_reg, args.lambda1, args.lambda2)
    
    ## to be DAG 
    while(not ut.is_dag(W_est)):
        min_val = np.unique(W_est)[1]
        W_est[W_est == min_val] = 0
        # print(f'min_val: {min_val}')
        
    ## Visualize graph 
    vis_path = args.save_folder + "vis_dag_graph.png"
    labels = ut.get_labels(args.data_name)
    ut.draw_directed_graph(W_est, vis_path, labels) 
    
    # Shape 
    data_train = None
    X100 = shap.utils.sample(X_train, 100)
    explainer = shap.Explainer(notear_predict, X100)
    shap_values = explainer(explained_data)
    shap.plots.bar(shap_values, show=False)
    plt.savefig('shap_on_notear.png')
    
def arg_parser():
    parser = argparse.ArgumentParser()
    # path
    parser.add_argument("--root_path", 
                        default="/workspace/tripx/MCS/xai_causality/run_v2/boston_housing", 
                        type=str)
    parser.add_argument("--data_path", 
                        default="/workspace/tripx/MCS/xai_causality/dataset/scaled_boston_housing.csv", 
                        type=str)
    parser.add_argument("--data_name", 
                        default="boston_housing", 
                        type=str)
    parser.add_argument("--save_folder", 
                        default="/workspace/tripx/MCS/xai_causality/baseline_xai/shap", 
                        type=str)
    # training
    parser.add_argument("--lambda_reg", 
                        default=5, 
                        type=float)
    parser.add_argument("--lambda1", 
                        default=0.005, 
                        type=float)
    parser.add_argument("--lambda2", 
                        default=0.0025, 
                        type=float)
    parser.add_argument("--ratio_test", 
                        default=0.2, 
                        type=float)
    parser.add_argument("--no_seeds", 
                        default=20, 
                        type=int,
                        help='Number of random seed')
    
    return parser.parse_args()



if __name__ == '__main__':
    args = arg_parser()
    main(args)
