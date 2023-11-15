import  sys
sys.path.append("./")
import os
import argparse
import json
from notears.locally_connected import LocallyConnected
from notears.lbfgsb_scipy import LBFGSBScipy
from notears.trace_expm import trace_expm
import torch
import torch.nn as nn
import numpy as np
import math
import pandas as pd
import wandb
import notears.utils as ut
from sklearn.metrics import classification_report, f1_score, accuracy_score
import networkx as nx 
import matplotlib.pyplot as plt

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
    # Loss = causal(X:1->d) + Lambda*classification (y = f_c(X))
    # Lambda > 1: focus classification
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
    return W_est


def main(args):
    torch.set_default_dtype(torch.double)
    np.set_printoptions(precision=3)
    # Read labels 
    # project_name = f'XAI_Regression_{args.data_name}'
    project_name = f'XAI_Regression_tuning_{args.data_name}'

    test_loss_lst = []
    for seed in range(args.no_seeds):
        ut.set_random_seed(seed)
        seed_name = 'seed_'+str(seed)
        out_folder = args.root_path + f"{seed_name}"
        if not os.path.isdir(out_folder):
            os.mkdir(out_folder)
        out_folder += "/"
        wandb.init(
            project=project_name,
            name=seed_name,
            config={
            "name": seed_name,
            "dataset": args.data_name,
            "lambda1": args.lambda1,
            "lambda2": args.lambda2,
            'lambda_reg': args.lambda_reg,
            'ratio_test': args.ratio_test},
            dir=out_folder,
            mode=args.wandb_mode)

        # Read and process data
        X, X_train, X_test = ut.read_data(args.data_path, out_folder, args.ratio_test)
        d = X.shape[1]
        
        # Modeling
        print("type X_train ", X_train.dtype)
        model = NotearsMLP(dims=[d, 10, 1], bias=True)
        W_est = notears_nonlinear(model, X_train, wandb, args.lambda_reg, args.lambda1, args.lambda2)
        # assert ut.is_dag(W_est)
        print('Estimated: \n', W_est, W_est.shape)
        np.savetxt(out_folder + 'W_est.csv', W_est, delimiter=',')
        
        # Evaluation regression on test set
        cls_predict, target = pred_reg(X_test, model)
        test_loss = 0.5 / target.shape[0] * np.sum((cls_predict - target) ** 2)
        
        ## Visualize graph 
        vis_path = out_folder + "vis_graph.png"
        labels = ut.get_labels(args.data_name)
        ut.draw_directed_graph(W_est, vis_path, labels) 
        
        print("test_loss: ", test_loss)
        wandb.log({'test_loss': test_loss})
        test_loss_lst.append(test_loss)
        wandb.finish()
        
    test_loss_lst =  np.array(test_loss_lst) 
    final_reg_results = { 'test_loss': {'mean': np.mean(test_loss_lst), 
                                        'std': np.std(test_loss_lst)},}
    json_object = json.dumps(final_reg_results, indent=4)
    # Writing to sample.json
    with open(args.root_path + "final_reg.json", "w") as outfile:
        outfile.write(json_object)
 
def arg_parser():
    parser = argparse.ArgumentParser()
    # path
    parser.add_argument("--root_path", 
                        default="/workspace/tripx/MCS/xai_causality/run/ov_cancer/", 
                        type=str)
    parser.add_argument("--data_path", 
                        default="/dataset/PANCAN/OV_gene_filter.csv", 
                        type=str)
    parser.add_argument("--data_name", 
                        default="ov_cancer", 
                        type=str)
    # training
    parser.add_argument("--lambda_reg", 
                        default=3, 
                        type=float)
    parser.add_argument("--lambda1", 
                        default=0.01, 
                        type=float)
    parser.add_argument("--lambda2", 
                        default=0.01, 
                        type=float)
    parser.add_argument("--ratio_test", 
                        default=0.2, 
                        type=float)
    parser.add_argument("--no_seeds", 
                        default=15, 
                        type=int,
                        help='Number of random seed')
    # log
    parser.add_argument("--wandb_mode", 
                        default="online", 
                        type=str)
    
    return parser.parse_args()
   
if __name__ == '__main__':
    args = arg_parser()
    main(args)
