import matplotlib.pyplot as plt
import torch
import os
import numpy as np
from matplotlib import cm


def gen_V_contour(x, y, V_func, nicename, filename, a_center, b_center, committor_file=None, **kwargs):
    X, Y = torch.meshgrid(torch.tensor(x), torch.tensor(y), indexing='ij')
    grid = torch.stack([X.flatten(), Y.flatten()], dim=1)
    Z = V_func(grid, **kwargs).reshape(X.shape).detach().cpu().numpy()
    
    fig, ax = plt.subplots(figsize=(6, 5))
    cs = ax.contourf(X.numpy(), Y.numpy(), Z, cmap='PiYG', levels=np.linspace(Z.min(), 15, 35))
    cbar1 = fig.colorbar(cs, ax=ax)
    cbar1.set_label("Potential")

    # Optional: overlay committor
    if committor_file is not None:
        print(committor_file)
        C_data = np.load(committor_file)
        C = C_data[2] if C_data.shape[0] == 3 else C_data
        levels = np.linspace(0.1, 0.9, 9)
        cmap = cm.get_cmap("spring")
        cs2 = ax.contour(X.numpy(), Y.numpy(), C, levels=levels, cmap=cmap, linewidths=0.8)

        # Add second colorbar for committor
        norm = plt.Normalize(vmin=0.1, vmax=0.9)
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # required for colorbar
        cbar2 = fig.colorbar(sm, ax=ax, ticks=levels)
        cbar2.set_label("NN Committor")

    ax.plot(a_center[0], a_center[1], 'o', color='red', label='A')
    ax.plot(b_center[0], b_center[1], 'o', color='green', label='B')
    ax.text(0.02, 0.98, 'A', color='red', transform=ax.transAxes,
            fontsize=12, verticalalignment='top')
    ax.text(0.07, 0.98, 'B', color='green', transform=ax.transAxes,
            fontsize=12, verticalalignment='top')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(nicename + ' Potential')
    fig.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()




def evaluate_committor_on_grid(pt_file: str, X: np.ndarray, Y: np.ndarray, model_class):

    # Flatten the grid to shape (N, 2)
    points = np.stack([X.ravel(), Y.ravel()], axis=1)
    inputs = torch.tensor(points, dtype=torch.float32).cpu()

    # Load model
    model = model_class(dim=2)
    model.load_state_dict(torch.load(pt_file, map_location='cpu'))
    model.eval()

    # Run inference
    with torch.no_grad():
        output = torch.sigmoid(model(inputs)).cpu().numpy().reshape(X.shape)

    # Save output
    out_file = os.path.splitext(pt_file)[0] + "_gridnn.npy"
    np.save(out_file, output)
