import  sys
sys.path.append("./")
from notears.locally_connected import LocallyConnected
# from notears.lbfgsb_scipy import LBFGSBScipy
# from notears.lbfgsb_gpu import LBFGSBGPU

from causal_xai.notears.trace_expm import trace_expm, trace_expm_gpu
import torch
import torch.nn as nn
import numpy as np
import math
import pandas as pd
import time
from causal_xai.notears.pytorch_minimize.optim import MinimizeWrapper
from tqdm import tqdm
import causal_xai.notears.utils as ut
import random

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
        # fc2: local linear layers
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
                    if i == j:
                        bound = (0, 0)
                    else:
                        bound = (0, None)
                    bounds.append(bound)
        # constraint y=0 (chú ý số chiều)
        return bounds
 
    def forward(self, x):  # [n, d] -> [n, d]
        x = self.fc1_pos(x) - self.fc1_neg(x)  # [n, d * m1]
        x = x.view(-1, self.dims[0], self.dims[1])  # [n, d, m1]
        # x = x.unsqueeze(1).repeat(1, self.dims[0], 1)
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
        # A = torch.sum(fc1_weight * fc1_weight, dim=1).t()  # [i, j]
        A = torch.sum(fc1_weight, dim=2).t()  # [i, j]
        r = int(self.dims[1]/2)
        u = fc1_weight[:r,:]
        v = fc1_weight[r:,:]
        A = torch.matmul(u.T,v)
        A = torch.where(A < 0.0, torch.tensor(0.0), A)
        h = trace_expm_gpu(A) - d  
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
        # A = torch.sum(fc1_weight * fc1_weight, dim=1).t()  # [i, j]
        r = int(self.dims[1]/2)
        u = fc1_weight[:r,:]
        v = fc1_weight[r:,:]
        A = torch.matmul(u.T,v)
        A = torch.where(A < 0.0, torch.tensor(0.0), A)
        W = torch.sqrt(A)  # [i, j]
        W = W  # [i, j]
        return W
    
def gather_flat_bounds(model):
    bounds = []
    for p in model.parameters():
        if hasattr(p, 'bounds'):
            b = p.bounds
        else:
            b = [(0, None)] * p.numel()
        bounds += b
    
    return bounds

def squared_loss(beta, output, target):
    # Use weight
    # Loss = causal(X:1->d) + Lambda*classification (y = f_c(X))
    # Lambda > 1: focus classification
    n = target.shape[0]
    loss =  beta / n * torch.sum((output - target) ** 2)
    return loss

def dual_ascent_step(model, X, device, lambda1, lambda2, beta, rho, alpha, h, rho_max):
    """Perform one step of dual ascent in augmented Lagrangian."""
    h_new = None
    model.to(device)
    flat_bounds = gather_flat_bounds(model)
    # minimizer_args = dict(method='L-BFGS-B', jac=True, bounds=flat_bounds, options={'disp': True, 'maxiter': 100})
    minimizer_args = dict(method='L-BFGS-B', jac=True, options={'disp': True, 'maxiter': 100})
    optimizer = MinimizeWrapper(model.parameters(), minimizer_args)
    # optimizer = LBFGSBGPU(model.parameters())
    # optimizer = torch.optim.LBFGS(model.parameters())
    X_torch = torch.from_numpy(X)
    X_torch = X_torch.to(device)
    # X_torch = jnp.asarray(X_torch)
    while rho < rho_max:
        def closure():
            optimizer.zero_grad()
            X_hat = model(X_torch)
            loss = squared_loss(beta, X_hat, X_torch)
            h_val = model.h_func()
            penalty = 0.5 * rho * h_val * h_val + alpha * h_val
            l2_reg = 0.5 * lambda2 * model.l2_reg()
            l1_reg = lambda1 * model.fc1_l1_reg()
            primal_obj = loss + penalty + l2_reg + l1_reg
            # primal_obj = loss + penalty
            primal_obj.backward()
            return primal_obj
        optimizer.step(closure)  
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
                      device,
                      B_true,
                      lambda1: float = 0.,
                      lambda2: float = 0.,
                      beta: float = 0.,
                      max_iter: int = 100,
                      h_tol: float = 1e-8,
                      rho_max: float = 1e+16,
                      w_threshold: float = 0.3):
    start_time = time.time()
    rho, alpha, h = 1.0, 0.0, np.inf
    for _ in range(max_iter):
        rho, alpha, h = dual_ascent_step(model, X, device, lambda1, lambda2, beta,
                                         rho, alpha, h, rho_max)
        print("h: ", h)
        print("rho: ", rho)
        if h <= h_tol or rho >= rho_max:
            break
    print("--- Training %s seconds ---" % (time.time() - start_time))

    #Tuning w_threshold
    keep_going_w_threshold = True
    while keep_going_w_threshold:
        start_time = time.time()    
        W_est = model.fc1_to_adj()

        non_zero_values = W_est[W_est != 0]
        if non_zero_values.numel() == 0: 
            print('h matrix is empty')
            return W_est
        print("Non zero values before filter by threshold: ", non_zero_values.numel())
        
        # Scale into 0 - 1 
        W_est = torch.abs(W_est)
        min_val = torch.min(W_est)
        max_val = torch.max(W_est)
        W_est = (W_est - min_val) / (max_val - min_val)

        non_zero_values = W_est[W_est != 0]
        print("Non zero values after scaling: ", non_zero_values.numel())

        
        # Filter by threshold
        W_est = torch.where(W_est < w_threshold, torch.tensor(0.0), W_est)

        # Check if W_est is an empty matrix  
        non_zero_values = W_est[W_est != 0]
        print("Non zero values after filter by threshold: ", non_zero_values.numel())

        # Remove minimum until DAGs
        if not ut.is_dag(W_est): 
            print("Remove minimum until DAGs")

        while not ut.is_dag(W_est):
            non_zero_values = W_est[W_est != 0]
            if non_zero_values.numel() == 0: 
                break 
            else: 
                min_value = torch.min(non_zero_values)
                indices = torch.where(W_est == min_value)
                W_est[indices[0][0], indices[1][0]] = 0.0
                # print(torch.count_nonzero(W_est))
        acc = ut.count_accuracy(B_true, W_est.cpu().numpy() != 0)
        print('w_threshold: ', w_threshold)
        print(acc)
        print("--- Tuning %s seconds ---" % (time.time() - start_time))

        print('-------------------------------------')
        print('0: stop; other is value for w_threshold')

        from_user = input('stop or w_threshold: ')
        if from_user == '0': 
            keep_going_w_threshold = False
        else: 
            w_threshold = float(from_user)
    return W_est

def freezing(regulators, model):
    for module in model.fc2:
        for layer in module.parameters(): 
            for idx, param in enumerate(layer):
                if idx not in regulators and param.requires_grad:
                    # print(f'Not in regulators: {idx}')
                    param.data.requires_grad = False
                else: 
                    param.data.requires_grad = True
                    print(f'In regulators: {idx}')
    return model 


def main():
    torch.set_default_dtype(torch.double)
    np.set_printoptions(precision=3)
    # seed_value = random.randint(1, 100000)
    seed_value = 123
    torch.manual_seed(seed_value)
    # ut.set_random_seed(123)
    device = torch.device("cuda")

    # net = 1
    # out_path= f"/workspace/binhtlh/projects/causality/XAI_using_Structure_Learning/run/DREAM5/NET{net}/"
    out_path= f"/workspace/binhtlh/projects/causality/XAI_using_Structure_Learning/run/synthesis/"

    # data_path= f"/dataset/DREAM5/X_B_true/X_{net}.csv"
    # gt_path= f"/dataset/DREAM5/X_B_true/B_true_{net}.csv"
    # X = pd.read_csv(data_path, header=None)
    # X = X.to_numpy().astype(float)
    # d = X.shape[1]
    # print("d: ", d)
    # print("Shape X: ", X.shape)
    # B_true = pd.read_csv(gt_path, header=None)
    # B_true = B_true.to_numpy().astype(float)
    # # print("B_true: ", B_true)
    # print("Shape B_true : ", B_true.shape)
    # # np.savetxt(out_path + 'W_true.csv', B_true, delimiter=',')

    # K = int(0.3*d) + 1
    # print("K: ", K)

    # # read regulators 
    # regulators_path = f'/dataset/DREAM5/net{net}/net{net}_transcription_factors.tsv'
    # regulators = pd.read_csv(regulators_path, delimiter='\t', header=None)
    # regulators = (regulators[0].str.replace('G','').astype(int) - 1).tolist()

    n, d, s0, graph_type, sem_type = 200, 5, 9, 'ER', 'mim'
    B_true = ut.simulate_dag(d, s0, graph_type)
    np.savetxt(out_path + 'W_true.csv', B_true, delimiter=',')

    X = ut.simulate_nonlinear_sem(B_true, n, sem_type)
    print("type X: ", type(X))
    np.savetxt(out_path + 'X.csv', X, delimiter=',')

    model = NotearsMLP(dims=[d, 6, 1], bias=True)

    W_est = notears_nonlinear(model, X, device, B_true, lambda1=0.001, lambda2=0.01, beta=0.7)
    assert ut.is_dag(W_est)
    W_est_numpy = W_est.cpu().numpy()
    np.savetxt(out_path + 'W_est.csv', W_est_numpy, delimiter=',')
    # W_df = W2DF(W_est_numpy)
    # W_df.to_csv(out_path + 'W_df.csv', index=False, header=None)

    acc = ut.count_accuracy(B_true, W_est_numpy != 0)
    print('Final result: ')
    print(acc)

if __name__ == '__main__':
    start_time = time.time()
    main()