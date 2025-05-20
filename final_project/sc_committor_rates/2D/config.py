import torch
import numpy as np

class CommittorNet(torch.nn.Module):
    def __init__(self, dim):
        super(CommittorNet, self).__init__()
        self.dim = dim
        block = [torch.nn.Linear(dim, 50),
                      torch.nn.Tanh(),
                      torch.nn.Linear(50, 1),]
        self.Block = torch.nn.Sequential(*block)
    
    def forward(self, x):
        prediction = self.Block(x)
        return prediction.squeeze()
device = torch.device('cpu')


## TC
# run_name = "TC_mod"
# def V(x):
#     return 1*(3*torch.exp(-x[:,0]**2 - (x[:,1] - (1/3))**2) - 3*torch.exp(-x[:,0]**2 - (x[:,1]-(5/3))**2) - 5*torch.exp(-(x[:,0]-1)**2 - x[:,1]**2) - 5*torch.exp(-(x[:,0]+1)**2 - x[:,1]**2) + 0.2*x[:,0]**4 +0.2*(x[:,1]-(1/3))**4)

# dim = 2
# n_reporter_steps = torch.tensor([3]).to(device)
# batch_size = 11


# beta = torch.tensor([6.67]).to(device) # Inverse kT for our system
# gamma = torch.tensor([1]).to(device) # Friction coefficient for Langevin dynamics
# step_size = torch.tensor([1e-2]).to(device) # The step size for each step of Langevin dynamics

# # Set hyperparameters for the optimization
# n_windows = 11
# cutoff = torch.tensor([0.1]).to(device)
# a_center = torch.tensor([-1., 0.]).to(device)
# b_center = torch.tensor([1., 0.]).to(device)
# cutoff = torch.tensor([0.2]).to(device)
# x = torch.linspace(-2,2,200)
# y = torch.linspace(-1.5,2.5,200)




## MB
# run_name = "MB_mod"
# def V(x):
#     # equation 26, Muller-Brown potential
#     A = torch.tensor([-20, -10, -17, 1.5])
#     a = torch.tensor([-1, -1, -6.5, 0.7])
#     b = torch.tensor([0, 0, 11, 0.6])
#     c = torch.tensor([-10, -10, -6.5, 0.7])
#     x0 = torch.tensor([1, 0, -0.5, -1])
#     y0 = torch.tensor([0, 0.5, 1.5, 1])
#     def _gau(x, idx): # Defining a multidimensional Gaussian
#         return A[idx]*torch.exp(a[idx]*torch.square(x[:,0] - x0[idx]) + b[idx]*(x[:,0] - x0[idx])*(x[:,1] - y0[idx]) +
#         c[idx]*torch.square(x[:,1] - y0[idx]))
    
#     return _gau(x, 0) + _gau(x,1) +\
#             _gau(x, 2) + _gau(x,3)

# dim = 2

# n_reporter_steps = torch.tensor([5]).to(device)
# batch_size = 11

# beta = torch.tensor([1.0]).to(device) # Inverse kT for our system
# sampling_beta = torch.tensor([0]).to(device) # We can sample at a higher temperature
# gamma = torch.tensor([1]).to(device) # Friction coefficient for Langevin dynamics
# step_size = torch.tensor([1e-3]).to(device) # The step size for each step of Langevin dynamics
# n_windows = 11

# a_center = torch.tensor([-0.5, 1.5]).to(device)
# b_center = torch.tensor([0.5, 0.]).to(device)
# cutoff = torch.tensor([0.1]).to(device)
# x = torch.linspace(-2,1.5,100)
# y = torch.linspace(-1,2.5,100)

## Tilted Cosine
run_name = "holes"
dim = 2
s = .32
b = 5.0
def V(x):
    # x: shape (N, 2)
    k = 1.0             # cosine frequency
    a = 7.0            # amplitude (barrier height ≈ 13 kT)
    lambda_ = 1.0       # linear tilt in x₁
    mu = 0.05           # confining curvature in x₁
    nu = 0.05           # confining curvature in x₂

    x1 = x[:, 0]
    x2 = x[:, 1]

    periodic = -a * torch.sin(k * x1) * torch.cos(k * x2)
    tilt = lambda_ * x1
    confine = mu * x1**2 + nu * x2**2

    return periodic + tilt + confine  # shape (N,)


n_reporter_steps = torch.tensor([5]).to(device)
batch_size = 11

beta = torch.tensor([1.0]).to(device) # Inverse kT for our system
sampling_beta = torch.tensor([0]).to(device) # We can sample at a higher temperature
gamma = torch.tensor([1]).to(device) # Friction coefficient for Langevin dynamics
step_size = torch.tensor([1e-3]).to(device) # The step size for each step of Langevin dynamics
n_windows = 11

a_center = torch.tensor([-1.57,0.]).to(device)
b_center = torch.tensor([1.57,0.]).to(device)
cutoff = torch.tensor([0.1]).to(device)
x = torch.linspace(-3.,3.,100)
y = torch.linspace(-3.,3.,100)