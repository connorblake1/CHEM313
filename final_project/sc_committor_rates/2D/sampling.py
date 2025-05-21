import torch
import numpy as np
import pymbar
import matplotlib.pyplot as plt
from global_utils import cnmsam, max_K

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
torch.set_default_dtype(torch.float64)
torch.set_default_tensor_type(torch.DoubleTensor)

def V(x):
    return 3*torch.exp(-x[:,0]**2 - (x[:,1] - (1/3))**2) - 3*torch.exp(-x[:,0]**2 - (x[:,1]-(5/3))**2) - 5*torch.exp(-(x[:,0]-1)**2 - x[:,1]**2) - 5*torch.exp(-(x[:,0]+1)**2 - x[:,1]**2) + 0.2*x[:,0]**4 +0.2*(x[:,1]-(1/3))**4
    
def dist(x, y):
    return torch.sqrt(torch.sum(torch.square(x-y), axis = -1))
    
def Langevin_step(x, V, beta, gamma, step_size):
    x = torch.clone(x).detach()
    x.requires_grad = True
    gradient = torch.autograd.grad(outputs=V(x),
                                    inputs=x, grad_outputs = torch.ones_like(V(x)),
                                    create_graph = True)[0]
    step = -(1/gamma)*gradient*step_size.unsqueeze(-1) + torch.sqrt(2/(beta*gamma))*torch.normal(torch.zeros(x.size())).to(device)*torch.sqrt(step_size.unsqueeze(-1))
    return x + step

def take_reporter_steps(input_xs, V, beta, gamma, step_size, n_trajectories, n_steps, a_center, b_center, cutoff, adaptive = False):
    xs = torch.repeat_interleave(input_xs, n_trajectories, axis = 0).to(device)
    step_sizes = torch.repeat_interleave(step_size, xs.size()[0], axis = 0)
    steps = torch.clone(step_sizes)
    zeros = torch.zeros(xs.size()[0]).to(device)
    step_sizes = torch.where(dist(xs, a_center) < cutoff, zeros, step_sizes)
    step_sizes = torch.where(dist(xs, b_center) < cutoff, zeros, step_sizes) # Set step size to 0 if in basin
    for q in range(n_steps):
        xs = Langevin_step(xs, V, beta, gamma, step_sizes)
        steps += step_sizes
        step_sizes = torch.where(dist(xs, a_center) < cutoff, zeros, step_sizes)
        step_sizes = torch.where(dist(xs, b_center) < cutoff, zeros, step_sizes) # Set step size to 0 if in basin
        # Truncate reporter length once one reporter reaches a basin:
        if adaptive:
            for i in range(len(input_xs)):
                if step_sizes[i*n_trajectories:(i+1)*n_trajectories].nonzero().size()[0] < n_trajectories:
                    step_sizes[i*n_trajectories:(i+1)*n_trajectories] = zeros[i*n_trajectories:(i+1)*n_trajectories]
        
        if step_sizes.nonzero().size() == torch.Size([0,1]):
            break
    return xs.reshape([-1, n_trajectories, xs.size()[-1]]), (steps/step_size).cpu().detach().numpy()

def take_reporter_steps_multi(input_xs, V, beta, gamma, step_size, n_trajectories, n_steps, center, centers, cutoff, adaptive = False):
    xs = torch.repeat_interleave(input_xs, n_trajectories, axis = 0).to(device)
    step_sizes = torch.repeat_interleave(step_size, xs.size()[0], axis = 0)
    steps = torch.clone(step_sizes)
    zeros = torch.zeros(xs.size()[0]).to(device)
    for k in range(centers.shape[0]):
        step_sizes = torch.where(dist(xs,centers[k]) < cutoff, zeros, step_sizes)
    for q in range(n_steps):
        xs = Langevin_step(xs, V, beta, gamma, step_sizes)
        steps += step_sizes
        for k in range(centers.shape[0]):
            step_sizes = torch.where(dist(xs,centers[k]) < cutoff, zeros, step_sizes)
        # Truncate reporter length once one reporter reaches a basin:
        if adaptive:
            for i in range(len(input_xs)):
                if step_sizes[i*n_trajectories:(i+1)*n_trajectories].nonzero().size()[0] < n_trajectories:
                    step_sizes[i*n_trajectories:(i+1)*n_trajectories] = zeros[i*n_trajectories:(i+1)*n_trajectories]
        if step_sizes.nonzero().size() == torch.Size([0,1]):
            break
    return xs.reshape([-1, n_trajectories, xs.size()[-1]]), (steps/step_size).cpu().detach().numpy()

def calculate_committor_estimates(xs, net, a_center, b_center, cutoff, n_trajectories, i_a, i_b, cmask):
    print("xs",xs.shape,"ac,bc",a_center.shape,b_center.shape)

    zeros = torch.zeros(xs.size()[0]).to(device)
    ones = torch.ones(xs.size()[0]).to(device)
    a_estimates = cnmsam(net,xs,cmask)[...,i_a]
    xs_for_var = torch.reshape(xs, [int(xs.size()[0]/n_trajectories), n_trajectories, 2])
    a_estimates = torch.where(dist(xs, a_center) < cutoff, zeros, a_estimates)
    a_estimates = torch.where(dist(xs, b_center) < cutoff, ones, a_estimates)
    a_estimates = torch.reshape(a_estimates, [int(xs.size()[0]/n_trajectories), n_trajectories])
    final_a_estimates = torch.mean(a_estimates, axis = 1)
    final_a_var = (torch.sum(torch.var(xs_for_var, axis = 1), axis = -1))
    final_a_means = torch.mean(xs_for_var, axis = 1)
    
    b_estimates = cnmsam(net,xs,cmask)[...,i_b]
    b_estimates_for_var = torch.reshape(b_estimates, [int(xs.size()[0]/n_trajectories), n_trajectories])
    b_estimates = torch.where(dist(xs, a_center) < cutoff, ones, b_estimates)
    b_estimates = torch.where(dist(xs, b_center) < cutoff, zeros, b_estimates)
    b_estimates = torch.reshape(b_estimates, [int(xs.size()[0]/n_trajectories), n_trajectories])
    final_b_estimates = torch.mean(b_estimates, axis = 1)
    final_b_var = torch.sum(torch.var(xs_for_var, axis = 1))
    return final_a_estimates, final_b_estimates, final_a_var.detach(), final_a_means.detach()

def calculate_committor_estimates_multi(xs, net, centers, cutoff, n_trajectories, cmask):
    # xs.shape = [N,dim]
    print("MULTI")
    print("xs",xs.shape)
    N = xs.shape[0]
    K = max_K
    def hot_code(i):
        mask = torch.zeros([N,K]).to(device)
        mask[:,i] = 1
        return mask

    estimates = cnmsam(net,xs,cmask) # [N,K] dims
    xs_for_var = torch.reshape(xs, [int(N/n_trajectories), n_trajectories, 2])
    for k in range(estimates.shape[1]):
        dist_mat = xs.unsqueeze(1) - centers.unsqueeze(0) # [N,K,dim]
        dists = dist_mat.norm(dim=2)
        estimates = torch.where(dists < cutoff, hot_code(k), estimates)
    estimates = estimates.permute([1,0])
    estimates = torch.reshape(estimates,[K,int(N/n_trajectories), n_trajectories])
    final_estimates = torch.mean(estimates, axis=2)
    final_estimates = final_estimates.permute([1,0]) # [short, K]
    final_means = torch.mean(xs_for_var, axis=1)
    print("out_estimeates",final_estimates.shape)
    return final_estimates, final_means

def flux_sample(V, beta, gamma, step_size, a_center, basin_cutoff, n_crossings, stride = 1):
    # always going from A->B
    x = a_center.unsqueeze(0)
    n_steps = 0
    crossings = 0
    in_basin = True
    from_A = True
    escape_confs = []
    escape_times = []
    last_crossing = 0
    just_left_flag = False
    while crossings < n_crossings + 1:
        just_left_flag = False
        for i in range(stride):
            x = Langevin_step(x, V, beta, gamma, step_size)
        n_steps += stride
        # print(x)
        if torch.sqrt(torch.sum(torch.square(x - a_center))) > basin_cutoff and in_basin:
            just_left_flag = True
            #print("Leaving Basin")
            in_basin = False
            crossings += 1
            print(crossings)
            escape_confs.append(x.squeeze())
            escape_times.append(n_steps)
            n_steps = 0
            from_A = False
        # if torch.sqrt(torch.sum(torch.square(x - b_center))) < basin_cutoff:
        #    print("Actual Transition")
        #    x = a_center.unsqueeze(0)
        #    n_steps = 0
        if torch.sqrt(torch.sum(torch.square(x - a_center))) < basin_cutoff and from_A == False:
            #print("Re-entering Basin")
            from_A = True
            in_basin = True

    times = torch.tensor(escape_times).cpu() * step_size.cpu()
    confs = torch.stack(escape_confs).cpu()  
    return times, confs
