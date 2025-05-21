import dolfin
from dolfin import UserExpression
import math
import torch
from fem_utils import compute_committor
from config import *

# Define a custom potential expression, e.g. a sum of Gaussians
class MyPotential(UserExpression):
    def __init__(self, centers, b, **kwargs):
        super().__init__(**kwargs)
        self.centers = centers  # list of (x,y)
        self.b = b
    def eval(self, value, x):
        acc = 0.0
        for cx, cy in self.centers:
            dx, dy = x[0]-cx, x[1]-cy
            acc += math.exp(self.b*(dx*dx+dy*dy))
        value[0] = -math.log(acc)
    def value_shape(self): return ()


class MullerBrown(UserExpression):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # parameters from Equation 26
        self.A  = [-20.0, -10.0, -17.0,  1.5]
        self.a  = [ -1.0,  -1.0,  -6.5,  0.7]
        self.b  = [  0.0,   0.0,  11.0,  0.6]
        self.c  = [-10.0, -10.0,  -6.5,  0.7]
        self.x0 = [  1.0,   0.0,  -0.5, -1.0]
        self.y0 = [  0.0,   0.5,   1.5,  1.0]

    def eval(self, value, x):
        total = 0.0
        for i in range(4):
            dx = x[0] - self.x0[i]
            dy = x[1] - self.y0[i]
            total += self.A[i] * math.exp(
                self.a[i]*dx*dx +
                self.b[i]*dx*dy +
                self.c[i]*dy*dy
            )
        value[0] = total

    def value_shape(self):
        return ()

# MB

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
    out_file="mb_computed_committor.npy"
)               

# # softmax
# centers = torch.tensor([[-1.5,0.0],[0.0,0.0],[1.5,0.0]])
# V_expr = MyPotential(centers.cpu().tolist(), b=-20.0, degree=2)

# # TODO actually check this

# # Compute committor
# coords, vals = compute_committor(
#     V_expr, a_center, b_center,
#     x_min=-3, x_max=3, y_min=-3, y_max=3,
#     nx=100, ny=100,
#     radius=0.5,
#     output_file="committor.npy"
# )
