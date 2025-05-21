import dolfin
from dolfin import UserExpression
import math
import torch
from fem_utils import *
# this is just a demo for computing the MB potential
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

## MB
V_expr_mb = MullerBrown(degree=2)
vals_mb = compute_committor(
    V_expr_mb,
    a_center=(-.5,1.5),
    b_center=(.5,0.),
    x_min = -2.,
    y_min = -1.,
    x_max=1.5,
    y_max=2.5,
    mesh_Nx=300,
    mesh_Ny=300,
    nx=100,
    ny=100,
    radius=.1,
    out_file="mb_computed_committor"
)

x = torch.linspace(-2,1.5,100)
y = torch.linspace(-1,2.5,100)
X, Y = torch.meshgrid(x, y)
a = np.load('MB_committor.npy').T
a = np.clip(a,0.,1.)
b = np.load('mb_computed_committor.npy').reshape(100,100)
b = np.clip(b,0.,1.)
compare_committors(a, b, X, Y, 'mb_comparison.pdf',title_1="Original Committor (from repo)",title_2="Committor Recomputed via FEM")
