from dolfin import *
import numpy as np
import torch

def compute_committor(V_expr,
                      a_center, b_center,
                      x_min, x_max, y_min, y_max,
                      mesh_Nx, mesh_Ny,          # fine FEM mesh cells
                      nx, ny,                    # coarse output grid
                      radius=0.5,
                      beta=1.0,
                      out_file="committor.npy"):
    """
    Solve  div(ρ ∇q)=0  with ρ=exp(-βV)  on a fine mesh;
    Dirichlet: q=1 in disk around a_center, q=0 in disk around b_center.
    Down-sample onto an (nx×ny) rectangle and save [X, Y, q] to out_file.
    """

    # --- build fine FEM mesh -------------------------------------------------
    mesh = RectangleMesh(Point(x_min, y_min), Point(x_max, y_max),
                         mesh_Nx, mesh_Ny, "crossed")
    V_space = FunctionSpace(mesh, "CG", 1)

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

    # --- sample on regular nx×ny grid ----------------------------------------
    xs = np.linspace(x_min, x_max, nx)
    ys = np.linspace(y_min, y_max, ny)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    Q = np.empty((nx, ny))

    for i in range(nx):
        for j in range(ny):
            Q[i, j] = q_sol(Point(X[i, j], Y[i, j]))

    np.save(out_file, Q)
    return Q


import matplotlib.pyplot as plt
import numpy as np

def compare_committors(a, b, X, Y, filename, eps=1e-8):
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
    axs[0,0].set_title('Committor (Exact Computed)')
    fig.colorbar(im0, ax=axs[0,0])

    im1 = axs[0,1].contourf(X, Y, b, levels=50, cmap='viridis')
    axs[0,1].set_title('Committor (Computed)')
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