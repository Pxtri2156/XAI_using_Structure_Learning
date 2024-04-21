import torch
import numpy as np
import scipy.linalg as slin
import notears.utils as ut

torch.set_printoptions(precision=3)

K=50
class TraceExpm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # detach so we can cast to NumPy
        E = slin.expm(input.detach().numpy())
        f = np.trace(E)
        E = torch.from_numpy(E)
        ctx.save_for_backward(E)
        return torch.as_tensor(f, dtype=input.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        E, = ctx.saved_tensors
        grad_input = grad_output * E.t()
        return grad_input


class TraceExpmGPUV0(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # detach so we can cast to NumPy
        E = torch.linalg.matrix_exp(input)
        f = torch.trace(E)
        ctx.save_for_backward(E)
        return torch.as_tensor(f, dtype=input.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        E, = ctx.saved_tensors
        grad_input = grad_output * E.t()
        return grad_input
    
class TraceExpmGPU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # detach so we can cast to NumPy
        # K = torch.linalg.matrix_rank(input)
        I = torch.eye(input.shape[0], dtype=torch.double, device=input.device) # need to improve
        A, B = ut.matrix_factorization_svd(input, K)
        E = ut.cal_expm(A, B, I)

        # E = torch.linalg.matrix_exp(input) 
        f = torch.sum(torch.diagonal(E))
        ctx.save_for_backward(E)
        return torch.as_tensor(f, dtype=input.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        E, = ctx.saved_tensors
        grad_input = grad_output * E.t()
        return grad_input

trace_expm = TraceExpm.apply
trace_expm_gpu = TraceExpmGPU.apply
trace_expm_gpu_v0 = TraceExpmGPUV0.apply

def main():
    input = torch.randn(20, 20, dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(trace_expm, input)

    input = torch.tensor([[1, 2], [3, 4.]], requires_grad=True)
    tre = trace_expm(input)
    f = 0.5 * tre * tre
    print('f\n', f.item())
    f.backward()
    print('grad\n', input.grad)


if __name__ == '__main__':
    main()
