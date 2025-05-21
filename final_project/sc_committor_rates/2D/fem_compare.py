import torch
from fem_utils import compare_committors
import numpy as np

## MB
x = torch.linspace(-2,1.5,100)
y = torch.linspace(-1,2.5,100)
X, Y = torch.meshgrid(x, y)

a = 1-np.load('MB_committor.npy').T
b = np.load('mb_computed_committor.npy').reshape(100,100)
b = np.clip(b,0.,1.)
print(a.shape,b.shape)
compare_committors(a, b, X, Y, 'mb_comparison.pdf')
