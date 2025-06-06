import torch
import numpy as np
import os
from utils import *
from global_utils import max_K, CommittorNet, mpath

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
# run_name = "holes_2_2.5"
# dim = 2
# s = .32
# b = 5.0
# def V(x):
#     # x: shape (N, 2)
#     centers = torch.tensor([[-1.5, 0.0],
#                             [ 0.0, 0.0],
#                             [ 1.5, 0.0]], device=x.device)  # shape (3, 2)

#     a = 9.2     # depth of Gaussian wells (≈ barrier height)
#     s = 0.4     # width
#     c = 2.5     # confining strength

#     diff = x[:, None, :] - centers[None, :, :]  # (N, 3, 2)
#     sq_dist = torch.sum(diff**2, dim=-1)        # (N, 3)
#     wells = torch.exp(-sq_dist / (2 * s**2))    # (N, 3)
#     total = wells.sum(dim=1)                    # (N,)

#     confine = c * torch.sum(x**2, dim=1)        # (N,)
#     return -a * total + confine                 # (N,)


# n_reporter_steps = torch.tensor([5]).to(device)
# batch_size = 11

# beta = torch.tensor([1.0]).to(device) # Inverse kT for our system
# sampling_beta = torch.tensor([0]).to(device) # We can sample at a higher temperature
# gamma = torch.tensor([1]).to(device) # Friction coefficient for Langevin dynamics
# step_size = torch.tensor([1e-3]).to(device) # The step size for each step of Langevin dynamics
# n_windows = 11

# a_center = torch.tensor([-1.5,0.]).to(device)
# b_center = torch.tensor([0.,0.]).to(device)
# cutoff = torch.tensor([0.1]).to(device)
# x = torch.linspace(-3.,3.,100)
# y = torch.linspace(-3.,3.,100)


## Improved General Transition Well
##<CHANGE THESE PARAMETERS>
key = "linear"
key_param = 5
run_name = "wells_" + key

run_name = run_name + "_" + str(key_param)
a_i = 0
b_i = 4
run_name = run_name + f"_a{a_i}_b{b_i}"
validation_mode = False # this is for checking FEM solutions and plotting committors, must be run in the fenics conda env not the md_sims one


# there should be as many heights as there are wells
low_height = torch.tensor([0.0])
mid_height = torch.tensor([0.2])
heights = [low_height,mid_height,mid_height,mid_height,low_height]

##<\CHANGE THESE PARAMETERS>
if validation_mode:
    from fem_utils import *

nice_name = ""
if key == "linear":
    nice_name = f"{key_param} Linear Wells (A={a_i+1}, B={b_i+1})"
elif key == "triangle":
    nice_name= f"{key_param} Wells"

dim = 2
b_default = -20.
def make_V_from_centers(centers,heights=None):

    def V(x, b=b_default):
        # x [N,dim]
        diff = x[:, None, :] - centers[None, :, :]      # (N, M, dim)
        wells_k = torch.sum(diff**2, dim=-1)            # (N, M)
        if heights is not None:
            wells_k = wells_k + heights.unsqueeze(0)
        return -torch.logsumexp(b*wells_k,dim=1)
    return V

def line_centers(K, spacing=1.5):
    # K wells evenly spaced along x-axis
    x = torch.linspace(-(K-1)/2, (K-1)/2, K) * spacing
    return torch.stack([x, torch.zeros_like(x)], dim=1)  # shape (K,2)

def triangle_centers(K,radius=.9):
    # 3 wells at 120° intervals in 2D
    angles = torch.tensor([0, 2*np.pi/3, 4*np.pi/3])
    angles = angles[:K]
    x = radius * torch.cos(angles)
    y = radius * torch.sin(angles)
    return torch.stack([x, y], dim=1)  # shape (3,2)

def square_center(s=1.5, epsilon=0.):
    return torch.stack(
        (torch.tensor([0.,0.]),
        torch.tensor([s,0.]),
        torch.tensor([0.,s+epsilon]),
        torch.tensor([s,s])),
    )

center_dict = {
    "linear": line_centers(K=key_param, spacing = 1.5),
    "triangle": triangle_centers(K=key_param, radius=1.0),
    "square": square_center(s=1.5),
    "dist_square": square_center(s=1.5,epsilon=.4)
}

kheights = torch.stack(heights).squeeze()
kcenters = center_dict[key]

V = make_V_from_centers(kcenters,kheights)

def bounds(centers, margin=1, spacing=100):
    # centers: shape (N, 2)
    x_min = centers[:, 0].min().item() - margin
    x_max = centers[:, 0].max().item() + margin
    y_min = centers[:, 1].min().item() - margin
    y_max = centers[:, 1].max().item() + margin

    x = torch.linspace(x_min, x_max, spacing)
    y = torch.linspace(y_min, y_max, spacing)
    return x, y

n_reporter_steps = torch.tensor([5]).to(device)
batch_size = 11

beta = torch.tensor([1.0]).to(device) # Inverse kT for our system
sampling_beta = torch.tensor([0]).to(device) # We can sample at a higher temperature
gamma = torch.tensor([1]).to(device) # Friction coefficient for Langevin dynamics
step_size = torch.tensor([1e-3]).to(device) # The step size for each step of Langevin dynamics
n_windows = 11

a_center = kcenters[a_i].to(device)
b_center = kcenters[b_i].to(device)

# a_center = torch.tensor([-1.5,0.]).to(device)
# b_center = torch.tensor([0.,0.]).to(device)
cutoff = torch.tensor([0.1]).to(device)
x,y = bounds(kcenters)
gen_V_contour(x,y,V,nice_name,mpath(run_name+"_contour.png"),a_center,b_center,None)


if validation_mode:
    X,Y = torch.meshgrid(x,y)
    # generates NN commitor npy file
    print(run_name + f"_K{max_K}.pt")
    if os.path.exists(mpath(run_name + f"_K{max_K}.pt")):
        evaluate_committor_on_grid(mpath(run_name + f"_K{max_K}.pt"), X, Y, CommittorNet)
    else:
        print("Failed to generate _gridnn file. .pt weights model not found")
        exit()
    print("Generating Contour(s)")
    for i in range(max_K):
        fname = mpath(run_name + f"_K{max_K}_{i}_gridnn.npy")
        file_found = os.path.exists(fname)
        if not file_found:
            print(f"{fname} not found")
            continue
        print(f"Generating {fname}")
        gen_V_contour(x,y,V,nice_name,mpath(run_name+f"_K{max_K}_contour_com.png"),a_center=a_center,b_center=b_center, committor_file=fname)
    print("Running gridfem")
    # if not os.path.exists(mpath(run_name + "_gridfem")):
    V_expr_run = SoftMinPotential(kcenters,kheights,b_default)
    compute_committor(
        V_expr_run,
        a_center=a_center,
        b_center=b_center,
        x_min=float(x[0]),
        x_max=float(x[-1]),
        y_min=float(y[0]),
        y_max=float(y[-1]),
        mesh_Nx=300,
        mesh_Ny=300,
        nx=len(x),
        ny=len(y),
        radius=float(cutoff),
        out_file=mpath(run_name+"_gridfem")
    ) # generates FEM solution npy file
    print("Comparing Committors")
    for i in range(max_K):
        compare_committors(
            a=np.load(mpath(run_name + f"_K{max_K}_{i}_gridnn.npy")),
            b=np.load(mpath(run_name + "_gridfem.npy")),
            X=X,
            Y=Y,
            filename=mpath(run_name + f"_K{max_K}_{i}_grid_comparison.pdf"),
            title_1=f"A: NN Committor {i}",
            title_2="B: FEM Committor",
        )
    exit()






