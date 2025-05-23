from dolfin import *
import numpy as np
import torch
import matplotlib.pyplot as plt
import math
from dolfin import parameters
parameters["form_compiler"]["quadrature_degree"] = 8

def compute_committor(V_expr,
                      a_center, b_center,
                      x_min, x_max, y_min, y_max,
                      mesh_Nx, mesh_Ny,          # fine FEM mesh cells
                      nx, ny,                    # coarse output grid
                      out_file,
                      radius=0.5,
                      beta=1.0,
                    ):
    """
    Solve  div(ρ ∇q)=0  with ρ=exp(-βV)  on a fine mesh;
    Dirichlet: q=1 in disk around a_center, q=0 in disk around b_center.
    Down-sample onto an (nx×ny) rectangle and save [X, Y, q] to out_file.
    """

    # --- build fine FEM mesh -------------------------------------------------
    mesh = RectangleMesh(Point(x_min, y_min), Point(x_max, y_max),
                         mesh_Nx, mesh_Ny, "crossed")
    # refine around A
    cell_markers = MeshFunction("bool", mesh, mesh.topology().dim())
    cell_markers.set_all(False)
    for cell in cells(mesh):
        mp = cell.midpoint()
        if mp.distance(Point(a_center[0], a_center[1])) < 2.0 * radius:
            cell_markers[cell] = True
    mesh = refine(mesh, cell_markers)

    V_space = FunctionSpace(mesh, "CG", 2)

    # --- Dirichlet regions ---------------------------------------------------
    A_sub = CompiledSubDomain("pow(x[0]-xa,2)+pow(x[1]-ya,2) < r*r",
                              xa=a_center[0], ya=a_center[1], r=radius)
    B_sub = CompiledSubDomain("pow(x[0]-xb,2)+pow(x[1]-yb,2) < r*r",
                              xb=b_center[0], yb=b_center[1], r=radius)
    bc_A = DirichletBC(V_space, Constant(1.0), A_sub)
    bc_B = DirichletBC(V_space, Constant(0.0), B_sub)

    # --- assemble weighted Laplace problem -----------------------------------
    q = TrialFunction(V_space)
    v = TestFunction(V_space)

    V_func = interpolate(V_expr, V_space)
    rho    = project(exp(-beta * V_func), V_space)

    a_form = inner(rho * grad(q), grad(v)) * dx
    L_form = Constant(0.0) * v * dx

    q_sol = Function(V_space)
    solve(a_form == L_form, q_sol, [bc_A, bc_B])

    # # --- rate computation with higher precision  -----------------------

    # # 1) Project onto a higher‐order space (e.g. quadratic) for better gradients
    # P2 = FunctionSpace(mesh, "CG", 2)
    # q_high = interpolate(q_sol, P2)
    # grad_q = project(grad(q_high), VectorFunctionSpace(mesh, "CG", 2))

    # # 2) Recompute rho on P2 as well (optional but consistent)
    # V_high = interpolate(V_expr, P2)
    # rho_high = project(exp(-beta * V_high), P2)

    # # 3) Mark ∂A facets as before
    # facet_markers = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
    # A_sub.mark(facet_markers, 1)
    # dsA = Measure("ds", domain=mesh, subdomain_data=facet_markers,
    #               metadata={"quadrature_degree": 8})

    # n = FacetNormal(mesh)

    # # 4) Define the high‐precision flux form
    # flux_form = - (1.0/beta) * rho_high * dot(grad_q, n) * dsA(1)

    # # 5) Assemble with an explicit compiler hint for quadrature
    # k = assemble(flux_form,
    #              form_compiler_parameters={"quadrature_degree": 8})

    # print("High‐precision surface flux k =", k)

    # --- sample on regular nx×ny grid ----------------------------------------
    xs = np.linspace(x_min, x_max, nx)
    ys = np.linspace(y_min, y_max, ny)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    Q = np.empty((nx, ny))

    for i in range(nx):
        for j in range(ny):
            Q[i, j] = q_sol(Point(X[i, j], Y[i, j]))

    Q = 1.0 - np.clip(Q, 0.0, 1.0)
    np.save(out_file + ".npy", Q)
    return Q

def compare_committors(a, b, X, Y, filename, eps=1e-8, title_1 ='Committor (Exact Computed)', title_2 = 'Committor (Computed)'):
    """
    Compare two committor fields a and b on the same mesh defined by X, Y.
    Produces a 2x2 figure:
      [ a      | b       ]
      [ diff   | logdiff ]
    Saves as a PDF to filename.
    
    - a, b: 2D numpy arrays same shape as X, Y
    - X, Y: meshgrid arrays
    - filename: output PDF path
    - eps: small constant to avoid log(0)
    """
    diff = a - b
    logdiff = np.log10(np.abs(diff) + eps)

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    im0 = axs[0,0].contourf(X, Y, a, levels=50, cmap='viridis')
    axs[0,0].set_title(title_1)
    fig.colorbar(im0, ax=axs[0,0])

    im1 = axs[0,1].contourf(X, Y, b, levels=50, cmap='viridis')
    axs[0,1].set_title(title_2)
    fig.colorbar(im1, ax=axs[0,1])

    im2 = axs[1,0].contourf(X, Y, diff, levels=50, cmap='RdBu')
    axs[1,0].set_title('Difference (A - B)')
    fig.colorbar(im2, ax=axs[1,0])

    im3 = axs[1,1].contourf(X, Y, logdiff, levels=50, cmap='magma')
    axs[1,1].set_title('Log10(|Difference| + eps)')
    fig.colorbar(im3, ax=axs[1,1])

    for ax in axs.flat:
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


class SoftMinPotential(UserExpression):
    def __init__(self, centers, heights, b, **kwargs):
        super().__init__(**kwargs)
        self.centers = centers  # list of (x,y)
        self.heights = heights
        self.b = b
    def eval(self, value, x):
        acc = 0.0
        for i in range(len(self.centers)):
            cx, cy = self.centers[i]
            dx, dy = x[0]-cx, x[1]-cy
            acc += math.exp(self.b*(dx*dx+dy*dy) + self.heights[i])
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