import  sys
sys.path.append("./")
from notears.locally_connected import LocallyConnected
# from notears.lbfgsb_scipy import LBFGSBScipy
from notears.lbfgsb_gpu import LBFGSBGPU

from notears.trace_expm import trace_expm
import torch
import torch.nn as nn
import numpy as np
import math
import pandas as pd
import time
from pytorch_minimize.optim import MinimizeWrapper

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
        h = trace_expm(A.cpu()) - d  # (Zheng et al. 2018)
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
        fc1_weight = fc1_weight.view(d, -1, d)  # [j, m1, i]
        A = torch.sum(fc1_weight * fc1_weight, dim=1).t()  # [i, j]
        W = torch.sqrt(A)  # [i, j]
        W = W.cpu().detach().numpy()  # [i, j]
        return W
    
def gather_flat_bounds(model):
    bounds = []
    for p in model.parameters():
        if hasattr(p, 'bounds'):
            b = p.bounds
        else:
            b = [(None, None)] * p.numel()
        bounds += b
    
    return bounds

def squared_loss(output, target):
    # Use weight
    # Loss = causal(X:1->d) + Lambda*classification (y = f_c(X))
    # Lambda > 1: focus classification
    n = target.shape[0]
    loss = 0.5 / n * torch.sum((output - target) ** 2)
    return loss

def dual_ascent_step(model, X, device, lambda1, lambda2, rho, alpha, h, rho_max):
    """Perform one step of dual ascent in augmented Lagrangian."""
    h_new = None
    model.to(device)
    flat_bounds = gather_flat_bounds(model)
    minimizer_args = dict(method='L-BFGS-B', jac=True, bounds=flat_bounds, options={'disp':True, 'maxiter':100})
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
            loss = squared_loss(X_hat, X_torch)
            h_val = model.h_func()
            penalty = 0.5 * rho * h_val * h_val + alpha * h_val
            l2_reg = 0.5 * lambda2 * model.l2_reg()
            l1_reg = lambda1 * model.fc1_l1_reg()
            primal_obj = loss + penalty + l2_reg + l1_reg
            primal_obj.backward()
            return primal_obj
        optimizer.step(closure)  # NOTE: updates model in-place
        # # Set constraint 
        # for name, param in model.named_parameters():
        #     if "fc1" in name and "weight" in name:
        #         param.data.clamp_(0, None)
        #         start_idx = 0
        #         # print('Before: ', name, param.data)
        #         for k in range(model.dims[0]):
        #             end_indx = start_idx + model.dims[1]
        #             # print("Blocks " + str(k), param[start_idx:end_indx,k])
        #             param[start_idx:end_indx,k].data.clamp_(0, 0)
        #             start_idx=end_indx
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
                      lambda1: float = 0.,
                      lambda2: float = 0.,
                      max_iter: int = 100,
                      h_tol: float = 1e-8,
                      rho_max: float = 1e+16,
                      w_threshold: float = 0.3):
    rho, alpha, h = 1.0, 0.0, np.inf
    for _ in range(max_iter):
        rho, alpha, h = dual_ascent_step(model, X, device, lambda1, lambda2,
                                         rho, alpha, h, rho_max)
        if h <= h_tol or rho >= rho_max:
            break
    print("h: ", h)
    W_est = model.fc1_to_adj()
    W_est[np.abs(W_est) < w_threshold] = 0
    return W_est

def main():
    data_path="/dataset/DREAM5/X_B_true/X_1.csv"
    gt_path="/dataset/DREAM5/X_B_true/B_true_1.csv"
    out_path="/workspace/tripx/MCS/xai_causality/run/run_v8/"
    
    torch.set_default_dtype(torch.double)
    np.set_printoptions(precision=3)

    import notears.utils as ut
    ut.set_random_seed(123)
    device = torch.device("cuda")

    # n, d, s0, graph_type, sem_type = 200, 100, 50, 'ER', 'mim'
    # B_true = ut.simulate_dag(d, s0, graph_type)
    X = pd.read_csv(data_path, header=None)
    X = X.to_numpy().astype(float)
    d = X.shape[1]
    print("d: ", d)
    print("Shape X: ", X.shape)
    B_true = pd.read_csv(gt_path, header=None)
    B_true = B_true.to_numpy().astype(float)
    # print("B_true: ", B_true)
    print("Shape B_true : ", B_true.shape)
    np.savetxt(out_path + 'W_true.csv', B_true, delimiter=',')

    # X = ut.simulate_nonlinear_sem(B_true, n, sem_type)
    
    # print("type X: ", type(X))
    
    np.savetxt(out_path + 'X.csv', X, delimiter=',')

    model = NotearsMLP(dims=[d, 100, 1], bias=True)
    W_est = notears_nonlinear(model, X, device, lambda1=0.01, lambda2=0.01)
    assert ut.is_dag(W_est)
    np.savetxt(out_path + 'W_est.csv', W_est, delimiter=',')
    acc = ut.count_accuracy(B_true, W_est != 0)
    print(acc)

if __name__ == '__main__':
    main()
