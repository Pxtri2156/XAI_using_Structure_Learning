import  sys
sys.path.append("./")
import wandb 
import argparse
import torch 
import numpy as np
import pandas as pd
import scipy.linalg as slin
import scipy.optimize as sopt
from scipy.special import expit as sigmoid
from notears import utils
from notears.rank_factorization import rank_factorization

import time
import sympy as sp

eps = 1e-8

def benchmark_me():
    def decorator(function):
        def wraps(*args, **kwargs):
            start = time.time()
            result = function(*args, **kwargs)
            end = time.time()
            print(f"\t\t{function.__name__}: exec time: {(end - start) *10**3}")
            wandb.log({function.__name__:(end - start) *10**3})
            return result
        return wraps
    return decorator

@benchmark_me()
def non_pivot_columns_indices(matrix):
    zero_diag_columns = []
    for i in range(matrix.shape[0]):
        if matrix[i, i] == 0:
            zero_diag_columns.append(i)
    return zero_diag_columns

@benchmark_me()
def matrix_factorization_binh(W):
    #-----------------------------------------------
    # FIND REDUCED ROW ECHELON FORM
    # Convert the numpy array to sympy matrix
    sympy_matrix = sp.Matrix(W)

    # Compute the reduced row echelon form
    start = time.time()
    rref_matrix = sympy_matrix.rref()[0]
    end = time.time()
    print(f"\t\t sympy_matrix exec time: {(end - start) *10**3}")

    # Convert the resulting sympy matrix back to numpy array
    B = np.array(rref_matrix).astype(np.float64)

    #-----------------------------------------------
    # FIND TRUTHLY POVIT COLUMNS
    # Find non povit columns
    non_povit_columns = np.array(non_pivot_columns_indices(B))

    #Find row with whole values are zeros
    zero_rows = np.where(~np.any(B != 0, axis=1))[0]

    # Create a boolean mask for elements to keep
    mask = np.isin(non_povit_columns, zero_rows, invert=True)

    # Apply the mask to the original array to filter out elements
    non_povit_columns = non_povit_columns[mask]

    #-----------------------------------------------
    # FIND SUB MATRIX
    C = W[:, ~np.isin(np.arange(W.shape[1]), non_povit_columns)]

    non_zero_rows = np.any(B != 0, axis=1)
    F = B[non_zero_rows]
    # if W.all() == (C @ F).all():
    #     print("Successfully factorized!")
    # else:
    #     print("Error!")
    C = torch.tensor(C)
    F = torch.tensor(F)
    return C, F

@benchmark_me()
def series(x):
    """
    compute the matrix series: \sum_{k=0}^{\infty}\frac{x^{k}}{(k+1)!}
    """
    s = torch.eye(x.size(-1), dtype=torch.double)
    t = x / 2
    k = 3
    while torch.norm(t, p=1, dim=-1).max().item() > eps:
        s = s + t
        t = torch.matmul(x, t) / k
        k = k + 1
    return s

@benchmark_me()
def cal_expm(A, B, I):
    """
    Compute the expm based on A, B
    """
    V = B.matmul(A)
    series_V = series(V)
    expm_W = I + (A.matmul(series_V)).matmul(B)
    expm_W = expm_W.numpy()
    return expm_W

@benchmark_me()    
def matrix_factorization(W, t):
    # Perform SVD on matrix W
    W = torch.from_numpy(W)
    U, S, V = torch.svd(W)
    
    # Take the first t singular values and vectors
    Ut = U[:, :t]
    St = torch.diag(S[:t])
    Vt = V[:, :t]
    
    # Construct matrix A and B
    A = Ut @ torch.sqrt(St)
    B = torch.sqrt(St) @ Vt.t()
    return A, B
  
def notears_linear(X,wandb, t, lambda1, loss_type, max_iter=100, h_tol=1e-8, rho_max=1e+16, w_threshold=0.3):
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
        # Factorization W into A1 and A2 
        ## Matrix Factorization
        # A, B = matrix_factorization(W * W, t)
        try:
            A, B = matrix_factorization_binh(W * W)
            E = cal_expm(A, B, I)
            # A, B = rank_factorization(W*W)
        except:
            A, B = matrix_factorization(W * W, t)
            E = cal_expm(A, B, I)

        # A, B = matrix_factorization(W * W, t)
        ## Non-negative Matrix Factorization 
        
        # E = slin.expm(W * W)  # (Zheng et al. 2018)
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
    I = torch.eye(d, dtype=torch.double)
    w_est, rho, alpha, h = np.zeros(2 * d * d), 1.0, 0.0, np.inf  # double w_est into (w_pos, w_neg)
    bnds = [(0, 0) if i == j else (0, None) for _ in range(2) for i in range(d) for j in range(d)]
    if loss_type == 'l2':
        X = X - np.mean(X, axis=0, keepdims=True)
    for i in range(max_iter):
        w_new, h_new = None, None
        while rho < rho_max:
            sol = sopt.minimize(_func, w_est, method='L-BFGS-B', jac=True, bounds=bnds)
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
    out_folder = f"{args.root_path}/linear/"
    n, d, s0, graph_type, sem_type = args.samples , args.dimensions, args.dimensions - 5, 'ER', 'gauss'
    t = 10
    wandb.init(
        project="scale_notear",
        name=f"linear_exp_flow",
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
    W_est = notears_linear(X, wandb, t, lambda1=0.1, loss_type='l2')
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
                        default="/workspace/tripx/MCS/xai_causality/run/run_v9", 
                        type=str)
    parser.add_argument("--samples", 
                        default=100, 
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