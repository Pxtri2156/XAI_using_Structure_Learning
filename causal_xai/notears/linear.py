import  sys
sys.path.append("./")
import wandb 
import argparse

import numpy as np
import pandas as pd
import scipy.linalg as slin
import scipy.optimize as sopt
from scipy.special import expit as sigmoid
from notears import utils
import time

def benchmark_me():
    def decorator(function):
        def wraps(*args, **kwargs):
            start = time.time()
            result = function(*args, **kwargs)
            end = time.time()
            # print(f"\t\t{function.__name__}: exec time: {(end - start) *10**3}")
            wandb.log({function.__name__:(end - start) *10**3})
            return result
        return wraps
    return decorator

def notears_linear(X,wandb, lambda1, loss_type, max_iter=100, h_tol=1e-8, rho_max=1e+16, w_threshold=0.3):
    """Solve min_W L(W; X) + lambda1 ‖W‖_1 s.t. h(W) = 0 using augmented Lagrangian.

    Args:
        X (np.ndarray): [n, d] sample matrix
        lambda1 (float): l1 penalty parameter
        loss_type (str): l2, logistic, poisson
        max_iter (int): max num of dual ascent steps
        h_tol (float): exit if |h(w_est)| <= htol
        rho_max (float): exit if rho >= rho_max
        w_threshold (float): drop edge if |weight| < threshold

    Returns:
        W_est (np.ndarray): [d, d] estimated DAG
    """
    @benchmark_me()
    def _loss(W):
        """Evaluate value and gradient of loss."""
        M = X @ W
        if loss_type == 'l2':
            R = X - M
            loss = 0.5 / X.shape[0] * (R ** 2).sum()
            G_loss = - 1.0 / X.shape[0] * X.T @ R
        elif loss_type == 'logistic':
            loss = 1.0 / X.shape[0] * (np.logaddexp(0, M) - X * M).sum()
            G_loss = 1.0 / X.shape[0] * X.T @ (sigmoid(M) - X)
        elif loss_type == 'poisson':
            S = np.exp(M)
            loss = 1.0 / X.shape[0] * (S - X * M).sum()
            G_loss = 1.0 / X.shape[0] * X.T @ (S - X)
        else:
            raise ValueError('unknown loss type')
        return loss, G_loss
    
    @benchmark_me()
    def _h(W):
        """Evaluate value and gradient of acyclicity constraint."""
        E = slin.expm(W * W)  # (Zheng et al. 2018)
        h = np.trace(E) - d
        #     # A different formulation, slightly faster at the cost of numerical stability
        #     M = np.eye(d) + W * W / d  # (Yu et al. 2019)
        #     E = np.linalg.matrix_power(M, d - 1)
        #     h = (E.T * M).sum() - d
        G_h = E.T * W * 2
        return h, G_h

    @benchmark_me()
    def _adj(w):
        """Convert doubled variables ([2 d^2] array) back to original variables ([d, d] matrix)."""
        return (w[:d * d] - w[d * d:]).reshape([d, d])
    
    @benchmark_me()
    def _func(w):
        """Evaluate value and gradient of augmented Lagrangian for doubled variables ([2 d^2] array)."""
        W = _adj(w)
        loss, G_loss = _loss(W)
        h, G_h = _h(W)
        obj = loss + 0.5 * rho * h * h + alpha * h + lambda1 * w.sum()
        G_smooth = G_loss + (rho * h + alpha) * G_h
        g_obj = np.concatenate((G_smooth + lambda1, - G_smooth + lambda1), axis=None)
        return obj, g_obj

    n, d = X.shape
    w_est, rho, alpha, h = np.zeros(2 * d * d), 1.0, 0.0, np.inf  # double w_est into (w_pos, w_neg)
    bnds = [(0, 0) if i == j else (0, None) for _ in range(2) for i in range(d) for j in range(d)]
    if loss_type == 'l2':
        X = X - np.mean(X, axis=0, keepdims=True)
    for i in range(max_iter):
        w_new, h_new = None, None
        while rho < rho_max:
            start_t = time.time()
            sol = sopt.minimize(_func, w_est, method='L-BFGS-B', jac=True, bounds=bnds)
            end_t = time.time()
            print(f"Iter: {i} Exec time steps: {(end_t - start_t)*10**3} ms")
            w_new = sol.x
            h_new, _ = _h(_adj(w_new))
            if h_new > 0.25 * h:
                rho *= 10
            else:
                break
        w_est, h = w_new, h_new
        wandb.log({"h": h, 
                   "alpha": alpha, 
                   "rho": rho})
        alpha += rho * h
        if h <= h_tol or rho >= rho_max:
            break
    W_est = _adj(w_est)
    W_est[np.abs(W_est) < w_threshold] = 0
    return W_est


def main(args):
    # utils.set_random_seed(1)
    cfg = f'cfg_{str(args.cfg)}'
    out_folder = f"{args.root_path}{cfg}/linear/"
    n, d, s0, graph_type, sem_type = args.samples , args.dimensions, args.dimensions, 'ER', 'gauss'
    
    wandb.init(
        project="linear_notear",
        name=f"linear_notea_{cfg}",
        config={
        "samples": n,
        "dimension": d,},
        dir=out_folder,
        mode=args.wandb_mode)
    
    B_true = utils.simulate_dag(d, s0, graph_type)
    W_true = utils.simulate_parameter(B_true)
    np.savetxt(out_folder +'W_true.csv', W_true, delimiter=',')

    X = utils.simulate_linear_sem(W_true, n, sem_type)
    np.savetxt(out_folder +'X.csv', X, delimiter=',')
    
    start = time.time() 
    W_est = notears_linear(X, wandb, lambda1=0.1, loss_type='l2')
    end = time.time()
    print("The time of execution of above program is :",
      (end-start) * 10**3, "ms")
    print('W_est: ', W_est.shape)
    assert utils.is_dag(W_est)
    np.savetxt(out_folder +'W_est.csv', W_est, delimiter=',')
    acc = utils.count_accuracy(B_true, W_est != 0)
    print(acc)

def arg_parser():
    parser = argparse.ArgumentParser()
    # path
    parser.add_argument("--root_path", 
                        default="/workspace/tripx/MCS/xai_causality/run_v7/", 
                        type=str)
    parser.add_argument("--cfg", 
                        default=0, 
                        type=int)

    parser.add_argument("--samples", 
                        default=300, 
                        type=int)
    parser.add_argument("--dimensions", 
                        default=20, 
                        type=int)
    parser.add_argument("--wandb_mode", 
                        default="disabled", 
                        type=str)
    
    return parser.parse_args()
   
if __name__ == '__main__':
    args = arg_parser()
    main(args)