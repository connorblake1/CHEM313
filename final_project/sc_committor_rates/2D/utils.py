import matplotlib.pyplot as plt
import torch
import numpy as np

def gen_V_contour(x, y, V_func, filename, a_center, b_center, **kwargs):
    # x, y: 1D arrays
    # V_func: callable (N,2) → (N,)
    X, Y = torch.meshgrid(torch.tensor(x), torch.tensor(y), indexing='ij')
    grid = torch.stack([X.flatten(), Y.flatten()], dim=1)  # (N², 2)
    Z = V_func(grid, **kwargs).reshape(X.shape).detach().cpu()

    plt.figure(figsize=(6,5))
    cs = plt.contourf(X.numpy(), Y.numpy(), Z.numpy(), cmap='PiYG',
                      levels=np.linspace(Z.min(),15, 35))
    plt.colorbar(cs)
    
    # Plot red and green markers
    plt.plot(a_center[0], a_center[1], 'o', color='red', label='A')
    plt.plot(b_center[0], b_center[1], 'o', color='green', label='B')
    
    # Text legend in upper left
    plt.text(0.02, 0.98, 'A', color='red', transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top')
    plt.text(0.07, 0.98, 'B', color='green', transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(filename.strip(".png") + ' Potential')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()