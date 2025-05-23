import sampling
import training
import torch
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import numpy as np
from config import V, dim, a_center, b_center, n_windows, cutoff, n_reporter_steps, batch_size, run_name, x, y, beta, gamma, step_size, nice_name, kcenters, kheights
import json
import os
from json import JSONDecodeError
from sklearn.cluster import DBSCAN
from global_utils import max_K, cnmsam, dist, append_rate, CommittorNet, mpath

# For plotting
matplotlib.rcParams['text.usetex'] = False
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list("mycmap", ['#1B346C','#01ABE9','#F1F8F1','#F54B1A'])
cmap2 = LinearSegmentedColormap.from_list("mycmap2", ['#000000','#ffffff'])
cmap4 = LinearSegmentedColormap.from_list("mycmap4", ['#ffffff','#F54B1A'])
cmap5 = LinearSegmentedColormap.from_list("mycmap5", ['#000000','#000000'])
cmap6 = LinearSegmentedColormap.from_list("mycmap5", ['#ffffff','#ffffff'])
matplotlib.colormaps.register(name="mycmap",cmap=cmap)
matplotlib.colormaps.register(name="mycmap2",cmap=cmap2)
matplotlib.colormaps.register(name="mycmap4",cmap=cmap4)
matplotlib.colormaps.register(name="mycmap5",cmap=cmap5)
matplotlib.colormaps.register(name="mycmap6",cmap=cmap6)
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
torch.set_default_dtype(torch.float64)
torch.set_default_tensor_type(torch.DoubleTensor)
# Set a deterministic RNG seed
torch.manual_seed(42)

# Initialize evenly-spaced configurations that linearly interpolate the basin centers
init_data = torch.zeros((n_windows, dim)) 
for d in range(dim):
    init_data[:,d] = torch.linspace(a_center[d], b_center[d], n_windows)

# net = CommittorNet(dim=dim).to(device).double()
net2 = CommittorNet(dim=dim).to(device).double()




print("Initial representation of the committor has been trained!")

# Run the optimization
n_trials = 1
n_opt_steps = 700
# optimizer = torch.optim.Adam(net.parameters(), lr = 1e-2)
optimizer2 = torch.optim.Adam(net2.parameters(), lr = 1e-2)

xs = init_data.to(device)
us = torch.linspace(0, 1, n_windows).to(device)
n_reporter_trajectories = torch.tensor([100]).to(device)
# For plotting:
X, Y = torch.meshgrid(x, y)
grid_input = torch.cat((X.unsqueeze(-1),Y.unsqueeze(-1)), dim = -1).to(device)
print(V(grid_input.reshape(-1,dim)))
print(V(grid_input.reshape(-1,dim)).size())
print(X.size())
V_surface = V(grid_input.reshape((-1, dim))).reshape(X.size())
V_surface_numpy = V_surface.cpu().detach().numpy()
V_surface_min = V_surface_numpy.min()
# Calculate Flux out of the basins

##<CHANGE THESE PARAMETERS>

K = max_K
# cmask = torch.arange(max_K) < 2
cmask = torch.ones(max_K)
# centers_k = torch.stack((a_center,b_center,kcenters[1],kcenters[2],kcenters[3]),dim=0)
centers_k = torch.stack((a_center,b_center),dim=0)
##<\CHANGE THESE PARAMETERS>


run_name = run_name + f"_K{K}"
#<block Z>


escape_confs_list = []
escape_times_list = []
print("Calculating Flux...")
for k in range(max_K): # TODO put back at 1k
    times, confs = sampling.flux_sample(V, beta, gamma, step_size, centers_k[k], cutoff, 1000, stride=1)
    escape_confs_list.append(confs.clone().reshape([-1,dim]).to(device).double().detach())
    escape_times_list.append(times)
escape_confs_k = torch.stack(escape_confs_list)
escape_times_k = torch.stack(escape_times_list)

K, N, _ = escape_confs_k.shape
running_exit_confs_k = [[] for _ in range(K)]
exit_indices_k    = torch.stack([torch.randperm(N) for _ in range(K)])



transit_history_k = [[] for _ in range(K)]
transit_k         = [-1]*K
last_transit_k    = torch.zeros(K, dtype=torch.long)
means_k           = []
times_k           = [[] for _ in range(K)]
losses_k          = []
log_losses_k      = []
index_counters_k  = torch.zeros(K, dtype=torch.long)
k_running_short_reporters = torch.zeros([K,n_reporter_steps,dim])

# <split>
# a_escape_times, a_escape_confs = sampling.flux_sample(V, beta, gamma, step_size, a_center, cutoff, 100, stride = 1) 
# b_escape_times, b_escape_confs = sampling.flux_sample(V, beta, gamma, step_size, b_center, cutoff, 100, stride = 1)
# a_escape_confs_list = a_escape_confs.clone().reshape([-1,dim]).to(device).double().detach()
# b_escape_confs_list = b_escape_confs.clone().reshape([-1,dim]).to(device).double().detach()
# running_a_exit_confs = []
# running_b_exit_confs = []
# a_exit_indices = torch.randperm(len(a_escape_confs_list))
# b_exit_indices = torch.randperm(len(b_escape_confs_list))
# i_a = 0
# i_b = 1
# a_transit_history = [0]
# b_transit_history = [0]
# a_transit = False
# last_a_transit = 0
# b_transit = False
# last_b_transit = 0
# a_means = []
# a_times = []
# b_means = []
# b_times = []
# losses = []
# log_losses = []
# a_index_counter = 0
# b_index_counter = 0
#<\block Z>



# START RUNNING
for step in range(n_opt_steps):
    # optimizer.zero_grad()
    optimizer2.zero_grad()
    # net.zero_grad()
    net2.zero_grad()

    #<block A>
    xs_list = []
    for k in range(K):
        if step == 0 or transit_k[k] != -1:
            xs_k = escape_confs_list[k][exit_indices_k[k, index_counters_k[k]] ]
            xs_k = xs_k.unsqueeze(0)
            transit_k[k] = -1
            running_exit_confs_k[k].append(xs_k.cpu().numpy())
        else:
            weights_k = cnmsam(net2, k_running_short_reporters[k].squeeze().reshape([-1,dim]), cmask)[..., k][last_transit_k[k]:].detach()

            weights_k = torch.where(
                dist(k_running_short_reporters[k].squeeze().reshape([-1,dim])[last_transit_k[k]:], centers_k[k]) < cutoff,
                torch.tensor(0.0, device=weights_k.device),
                weights_k
            )
            idx_k = torch.sort(weights_k, descending=True)[1][:1]
            xs_k = k_running_short_reporters[k].squeeze().reshape([-1,dim])[last_transit_k[k]:][idx_k]  # shape (1, dim)
        xs_list.append(xs_k)
    xs_k = torch.stack(xs_list, dim=0)
    # print("A: xs_k",xs_k.shape)
    # print("A:", xs_k)
    # <split>
    # if step == 0 or a_transit == True:
    #     a_xs = a_escape_confs_list[a_exit_indices[a_index_counter]]
    #     a_xs = a_xs.unsqueeze(0)
    #     a_transit = False
    #     running_a_exit_confs.append(a_xs.cpu().numpy())
    # else:
    #     # a_reporter_energies = V(a_running_short_reporters.reshape([-1,dim])[last_a_transit:])  # noqa: F821
    #     a_weights = cnmsam(net,a_running_short_reporters.squeeze().reshape([-1, dim]),cmask)[...,i_a][last_a_transit:].detach() # noqa: F821
    #     a_weights = torch.where(dist(a_running_short_reporters.squeeze().reshape([-1, dim])[last_a_transit:], a_center) < cutoff, 0., a_weights) # noqa: F821
    #     _, a_indices = torch.sort(a_weights, descending = True)
    #     a_indices = a_indices[:1]
    #     a_xs = a_running_short_reporters.squeeze().reshape([-1, dim])[last_a_transit:][a_indices]# noqa: F821
    # if step == 0 or b_transit == True:
    #     b_xs = b_escape_confs_list[b_exit_indices[b_index_counter]]
    #     b_xs = b_xs.unsqueeze(0)
    #     b_transit = False
    #     running_b_exit_confs.append(b_xs.cpu().numpy())
    # else:
    #     # b_reporter_energies = V(b_running_short_reporters.reshape([-1,dim])[last_b_transit:])# noqa: F821
    #     b_weights = cnmsam(net,b_running_short_reporters.squeeze().reshape([-1, dim]),cmask)[...,i_b][ last_b_transit:].detach()# noqa: F821
    #     b_weights = torch.where(dist(b_running_short_reporters.squeeze().reshape([-1, dim])[ last_b_transit:], b_center) < cutoff, 0., b_weights)# noqa: F821
    #     _, b_indices = torch.sort(b_weights, descending = True)
    #     b_indices = b_indices[:1]
    #     b_xs = b_running_short_reporters.squeeze().reshape([-1, dim])[last_b_transit:][b_indices]# noqa: F821
    # print("A: a_xs",a_xs.shape)
    #<\block A>

    #<block B>
    k_short_reporters_list = []
    for k in range(K):
        k_short_reporters, k_short_times = sampling.take_reporter_steps_multi(xs_k[k], V, beta, gamma, step_size, n_reporter_trajectories, n_reporter_steps, centers_k[k], centers_k, cutoff, adaptive= True)
        k_short_reporters_list.append(k_short_reporters)
        times_k[k].append(np.mean(k_short_times))
    k_short_reporters = torch.stack(k_short_reporters_list,dim=0)
    # print("B:", k_short_reporters)
    # print("B: k short reporters",k_short_reporters.shape) #st
    # print("B: len times_k",len(times_k),len(times_k[0]))
    # print()
    # print(k_short_reporters)
    # <split>
    # a_short_reporters, a_short_times = sampling.take_reporter_steps(a_xs, V, beta, gamma, step_size, n_reporter_trajectories, n_reporter_steps, a_center, b_center, cutoff, adaptive = True)
    # b_short_reporters, b_short_times = sampling.take_reporter_steps(b_xs, V, beta, gamma, step_size, n_reporter_trajectories, n_reporter_steps, a_center,  b_center, cutoff, adaptive = True) # CB: why are A and B not swapped? Ohh it's because the committor is not swap invariant
    # a_times.append(np.mean(a_short_times))
    # b_times.append(np.mean(b_short_times))
    # # print("B: a_short_reporters",a_short_reporters.shape) #st
    # # print("B: a_times",len(a_times)) #st
    # # print()
    # # print(a_short_reporters,b_short_reporters)
    #<\block B>

    # Keep a memory of sampled configurations
    #<block C>
    if step == 0:
        k_running_xs = xs_k.detach()
        running_xs_all = k_running_xs.flatten(0,1)
        k_running_short_reporters = k_short_reporters
        running_short_reporters_all = k_running_short_reporters.flatten(0,1)
    else:
        k_running_xs = torch.cat((k_running_xs.detach(), xs_k.detach()), dim=1)
        k_running_short_reporters = torch.cat((k_running_short_reporters.detach(), k_short_reporters),dim=1)
        running_short_reporters_all = k_running_short_reporters.flatten(0,1)
        running_xs_all = k_running_xs.flatten(0,1)
    # print("C: xs_k",xs_k.shape)
    # print("C: k_short_reporters",k_short_reporters.shape)
    # print(k_short_reporters)
    # print("C: k_running_short_reporters",k_running_short_reporters.shape)
    # print("C: running_short_reporters_all",running_short_reporters_all.shape)
    # print("C: k_running_xs",k_running_xs.shape)
    # print("C: running_xs_all",running_xs_all.shape)
    # print()
    # <split>
    # if step == 0:
    #     a_running_xs = a_xs.detach()
    #     b_running_xs = b_xs.detach()
    #     running_xs = torch.cat((a_running_xs, b_running_xs), axis = 0)
    #     a_running_short_reporters = a_short_reporters
    #     b_running_short_reporters = b_short_reporters
    #     running_short_reporters = torch.cat((a_running_short_reporters, b_running_short_reporters), axis = 0)
        
    # else:
    #     a_running_xs = torch.cat((a_running_xs.detach(), a_xs.detach()))
    #     b_running_xs = torch.cat((b_running_xs.detach(), b_xs.detach()))
    #     a_running_short_reporters = torch.cat((a_running_short_reporters.detach(), a_short_reporters), axis = 0)
    #     b_running_short_reporters = torch.cat((b_running_short_reporters.detach(), b_short_reporters), axis = 0)
    #     running_short_reporters = torch.cat((a_running_short_reporters, b_running_short_reporters), axis = 0)
    #     running_xs = torch.cat((a_running_xs, b_running_xs), axis = 0)
    # print("C: a_xs",a_xs.shape)
    # print("C: a_short_reporters",a_short_reporters.shape)
    # print("C: a_running_short_reporters",a_running_short_reporters.shape)
    # print("C: running_short_reporters",running_short_reporters.shape)
    # print("C: a_running_xs",a_running_xs.shape)
    # print("C: running xs",running_xs.shape)
    # <\block C>
    
    #<block D>
    with torch.no_grad():
        short_targets, short_means = sampling.calculate_committor_estimates_multi(running_short_reporters_all.reshape([-1,dim]), net2, centers_k, cutoff, n_reporter_trajectories, cmask)
    # print("D: short_targets",short_targets.shape, "running_xs_all",running_xs_all.shape)
    # print(short_targets)
    # print(short_means)
    # print("D: short_means",short_means.shape)
    batch_size = running_xs_all.size()[0]
    for m in range(100):
        permutation = torch.randperm(batch_size)
        for i in range(0, batch_size, batch_size):
            # print(m)
            optimizer2.zero_grad()
            net2.zero_grad()
            indices = permutation[i:i+batch_size]
            log_loss = training.half_log_loss_multi(
                net2,
                running_xs_all.to(device)[indices],
                short_targets.to(device)[indices],
                cmask
            )
            total_loss = log_loss
            total_loss.backward()
            optimizer2.step() 
    # <split>
    # for j in range(1): # Can go through multiple optimization steps per data collection step, if needed
    #     with torch.no_grad():
    #         a_short_targets, b_short_targets, a_short_var, a_short_means = sampling.calculate_committor_estimates(running_short_reporters.reshape([-1, dim]), net, a_center, b_center, cutoff, n_reporter_trajectories, i_a, i_b, cmask)            
    #         # print(a_short_targets, b_short_targets)
    #         # print(o_short_targets)

    #     batch_size = running_xs.size()[0]

    #     # print("D: a_short_targets", a_short_targets.shape)
    #     # print("D: a_short_means", a_short_means.shape)
    #     for m in range(100):
    #         permutation = torch.randperm(running_xs.size()[0])
    #         for i in range(0, len(running_xs), batch_size):
    #             # print(m)
    #             optimizer.zero_grad()
    #             net.zero_grad()
    #             indices = permutation[i:i+batch_size]
    #             log_loss,_ = training.half_log_loss(
    #                 net,
    #                 running_xs.to(device)[indices],
    #                 a_short_targets.to(device)[indices],
    #                 b_short_targets.to(device)[indices],
    #                 None,
    #                 None,
    #                 i_a,
    #                 i_b,
    #                 cmask
    #             )
    #             total_loss = log_loss
# 
                # shuffled = running_xs.to(device)[indices]
                # stacked = torch.stack((a_short_targets[indices],b_short_targets[indices]),dim=1).to(device)
                # log_loss = training.half_log_loss_multi(
                #     net, 
                #     shuffled,
                #     stacked,
                #     cmask
                # )
                # total_loss = 2*log_loss

                # total_loss.backward()
                # optimizer.step()
    # print()
    # print()
    # #<\block D>\

    # Estimate rates
    #<block E>
    with torch.no_grad():
        exit_tensor = torch.tensor(escape_confs_k.numpy())
        exit_predictions = cnmsam(net2, exit_tensor, cmask) # [K,N,K]
        exit_predictions = exit_predictions.diagonal(dim1=0, dim2=2) # [N,K]
        mean_exit_predictions = torch.mean(exit_predictions,dim=0)
        # print("E: exit_preds", exit_predictions.shape)
        rate_estimates = 1/torch.mean(escape_times_k, dim=1)*mean_exit_predictions
        means_k.append(rate_estimates)
        # print("means_k",means_k)
        # TODO export to dict
    # print("E: exit_tensor", exit_tensor.shape)
    # print("E: escape_times", escape_times_k.shape)
    print(f"Step {step}: Rate Estimate = {rate_estimates.detach().cpu().numpy()}; Loss = {float(log_loss.item()):.3f}")#, Log Loss 2 = {log_loss_2.item()}") # TODO

    # <split>
    # with torch.no_grad():
    #     a_exit_tensor = torch.tensor(np.array(a_escape_confs)).squeeze()
    #     b_exit_tensor = torch.tensor(np.array(b_escape_confs)).squeeze()
    #     a_rate_estimates = 1/torch.mean(a_escape_times)*torch.mean(cnmsam(net,a_exit_tensor,cmask)[...,i_a])
    #     b_rate_estimates = 1/torch.mean(b_escape_times)*torch.mean(cnmsam(net,b_exit_tensor,cmask)[...,i_b])

    #     a_rate_mean = torch.mean(a_rate_estimates)
    #     a_local = float(a_rate_mean.cpu().detach().numpy())
    #     a_means.append(a_rate_mean.cpu().detach().numpy())
    #     b_rate_mean = torch.mean(b_rate_estimates)
    #     b_local = float(b_rate_mean.cpu().detach().numpy())
    #     b_means.append(b_rate_mean.cpu().detach().numpy())
    #     append_rate(run_name=run_name, a_rate=a_local, b_rate=b_local)


    # # Report to the command line
    # print(f"Step {step}: Rate Estimate = {a_rate_mean.item()}; Loss = {log_loss.item()}")#, Log Loss 2 = {log_loss_2.item()}")
    # print(a_index_counter, b_index_counter)

    # print("E: a_exit_tensor", a_exit_tensor.shape)
    # print("E: a_escape_times", a_escape_times.shape)
    # print("E: a_rate_estimates", a_rate_estimates.shape)
    # print("E: a_means", len(a_means))
    # <\block E>

    # Check whether or not a sampling chain has reached the basin
    #<block F>
    for k in range(K):
        for j in range(K):
            if k == j:
                continue
            # print(k_running_short_reporters)
            # print(k_running_short_reporters.shape)
            disp_vecs = k_running_short_reporters[k].squeeze().reshape([-1,dim])[last_transit_k[k]:] -centers_k[j]
            basin_dists = torch.sqrt(torch.sum(torch.square(disp_vecs),dim=1))
            if basin_dists.shape[0] == 0:
                continue
            basin_min = torch.min(basin_dists)
            if basin_min < cutoff: # k->j transition
                print(f"TRANSIT {k}->{j}")
                transit_k[k] = j
                last_transit_k[k] = len(k_running_short_reporters[k].squeeze().reshape([-1,dim]))
                index_counters_k[k] += 1
                transit_history_k[k].append(j)
    # <split>
    # if torch.min(torch.sqrt(torch.sum(torch.square(a_running_short_reporters.reshape([-1, dim])[last_a_transit:] - b_center), axis = -1))) < cutoff:
    #     print("A Transit!")
    #     a_transit = True
    #     last_a_transit = len(a_running_short_reporters.reshape([-1, dim]))
    #     a_index_counter += 1
    #     a_transit_history.append(int(len(running_xs.reshape([-1, dim]))/2))
        
    # if torch.min(torch.sqrt(torch.sum(torch.square(b_running_short_reporters.reshape([-1, dim])[last_b_transit:] - a_center), axis = -1))) < cutoff:
    #     print("B Transit!")
    #     b_transit = True
    #     last_b_transit = len(b_running_short_reporters.reshape([-1, dim]))
    #     b_index_counter += 1
    #     b_transit_history.append(int(len(running_xs.reshape([-1, dim]))/2))
    #<\block F>
    # plotting
    if step % 10 == 9:
        #<block G>
        plt.tight_layout()
        fig, axs  = plt.subplot_mosaic([['a', 'b'], ['a', 'b']], width_ratios = [1., 1.])
        fig.set_size_inches(15.2, 4.8)
        axs['a'].contourf(X,Y, V_surface_numpy, levels=np.linspace(V_surface_min, 15, 35), cmap = 'mycmap',zorder=0)
        
        cmaps  = ["winter", "cool",  "copper", "Wistia", "brg"]
        levels = np.linspace(0.1, 0.9, 10)
        norm   = Normalize(vmin=levels.min(), vmax=levels.max())
        data_k   = 1. - cnmsam(net2, grid_input, cmask).cpu().detach().numpy()
        for k in range(2):
            cs  = axs['a'].contour(
                        X, Y, data_k[...,k],
                        levels=levels,
                        cmap=cmaps[k],
                        norm=norm
                    )
            mappable = ScalarMappable(norm=norm, cmap=cmaps[k])
            cbar     = fig.colorbar(mappable, ax=axs['a'])
            cbar.set_label(rf'$p_{{{k}}}$', fontsize=14)

        data_reshaped = torch.reshape(running_xs_all, [-1, 2]).detach().numpy()
        
        if max_K == 2:
            with torch.no_grad():
                c0 = cnmsam(net2,torch.tensor(data_reshaped),cmask)[...,0]
            axs['a'].scatter(data_reshaped[:,0], data_reshaped[:,1], c = c0, cmap = 'mycmap2', alpha = 1)
 
        for k in range(K):
            circle = plt.Circle(
                    centers_k[k],
                    cutoff,
                    linewidth=2,
                    color='red',
                    fill=True,
                    # label=f'$k={k}$'
                )
            axs['a'].add_patch(circle)
            xk, yk = centers_k[k]
            axs['a'].text(xk, yk,f'${k}$', ha='center', va='center')

        axs['a'].set_xlabel(r"x", size = 16)
        axs['a'].set_ylabel(r"y", size = 16)
        axs['a'].set_xlim(x[0],x[-1])
        axs['a'].set_ylim(y[0],y[-1])
        # axs['a'].legend()
                
        dt = 2 * (n_reporter_trajectories * n_reporter_steps).cpu().detach().numpy()*step_size.cpu().detach().numpy()
        out_means_k = torch.stack((means_k),dim=0)
        for k in range(K):
            cmap = plt.get_cmap(cmaps[k])
            color = cmap(1.0)
            axs['b'].plot(
                np.arange(step+1) * dt,
                out_means_k[: step+1,k],
                color=color,
                label=f'$k={k}$'
            )
        axs['b'].legend(fontsize=16)


        axs['b'].set_yscale('log')
        axs['b'].set_xlabel(r"Sampling Time ($\tau$)", size = 16)
        axs['b'].set_ylabel(r"Rate ($\tau^{-1}$)", size = 16)
        axs['b'].set_ylim(1e-8, 1e1)
        axs['b'].tick_params(axis='y', labelsize=14)
        axs['b'].tick_params(axis='x', labelsize=14)
        plt.tight_layout()
        fig.savefig(mpath(run_name + "_all.pdf"))
        plt.close()
        print(transit_history_k)
        # <split>
        # plt.tight_layout()
        # fig, axs  = plt.subplot_mosaic([['a', 'b'], ['a', 'b']], width_ratios = [1., 1.])
        # fig.set_size_inches(15.2, 4.8)
        # axs['a'].contourf(X,Y, V_surface_numpy, levels=np.linspace(V_surface_min, 15, 35), cmap = 'mycmap',zorder=0)
        
        # fig.colorbar(axs['a'].contour(X, Y, cnmsam(net,grid_input,cmask)[...,i_a].cpu().detach().numpy(), levels = np.linspace(0.1, 0.9, 9), cmap ="spring"), ax = axs['a'], ticks = np.linspace(0, 1, 11))
        # axs['a'].scatter(torch.reshape(running_xs, [-1, 2]).detach().numpy()[:,0], torch.reshape(running_xs, [-1, 2]).detach().numpy()[:,1], c = a_short_var, cmap = 'mycmap2', alpha = 1)
        # # axs['a'].scatter(torch.reshape(a_running_xs, [-1, 2]).detach().numpy()[-1,0], torch.reshape(a_running_xs, [-1, 2]).detach().numpy()[-1,1], color = 'blue', alpha = 1)
        # # axs['a'].scatter(torch.reshape(b_running_xs, [-1, 2]).detach().numpy()[-1,0], torch.reshape(b_running_xs, [-1, 2]).detach().numpy()[-1,1], color = 'green', alpha = 1)
        # axs['a'].add_patch(plt.Circle(a_center, cutoff, linewidth = 2, color = 'red', fill = True))
        # axs['a'].add_patch(plt.Circle(b_center, cutoff, linewidth = 2, color = 'green', fill = True))
        # axs['a'].plot([], [], color='red', label='A')
        # axs['a'].plot([], [], color='green', label='B')
        # axs['a'].set_xlabel(r"x", size = 16)
        # axs['a'].set_ylabel(r"y", size = 16)
        # axs['a'].set_xlim(x[0],x[-1])
        # axs['a'].set_ylim(y[0],y[-1])
        # axs['a'].legend()

        # axs['b'].plot(np.arange(step+1)*2*((n_reporter_trajectories*n_reporter_steps).cpu().detach().numpy() + 1)*step_size.cpu().detach().numpy(), np.array(a_means), c = '#1B346C')
        # axs['b'].plot(np.arange(step+1)*2*((n_reporter_trajectories*n_reporter_steps).cpu().detach().numpy() + 1)*step_size.cpu().detach().numpy(), np.array(b_means), c = '#F54B1A')
        # axs['b'].set_yscale('log')
        # axs['b'].set_xlabel(r"Sampling Time ($\tau$)", size = 12)
        # axs['b'].set_ylabel(r"Rate ($\tau^{-1}$)", size = 12)
        # axs['b'].legend([r'Rate Estimate A to B', r'Rate Estimate B to A', r'Analytical Rate'], prop={'size': 12})
        # axs['b'].set_ylim(1e-8, 1e1)
        # plt.tight_layout()
        # fig.savefig(mpath(run_name + ".pdf"))
        # plt.close()
        #<\block G>
        
# COMPUTE DBSCANS ETC
# net.eval()
# with torch.no_grad():
#     q_xs = torch.sigmoid(net(running_xs)).cpu().numpy().reshape([-1,1])
# labels = DBSCAN(eps=.02, min_samples=10).fit_predict(q_xs)
# unique_labels = set(labels)
# K = len(unique_labels)-1

# for k in unique_labels:
#     if k == -1:
#         continue
#     class_member_mask = (labels == k)
#     q_k = q_xs[class_member_mask]



# Plot the final committor
# <block H>
plt.contourf(X,Y, V_surface, levels=np.linspace(V_surface_min, 15, 35), cmap = 'mycmap')
plt.contour(X, Y, net2(grid_input).detach()[:,:,0].numpy(), levels = np.linspace(0.1, 0.9, n_windows), cmap = 'mycmap2')
plt.savefig(mpath(run_name+"_committor.pdf"))
#plt.close()
print("SAVING")
torch.save(net2.state_dict(), mpath(run_name+ ".pt"))
torch.save(running_xs_all, mpath(run_name+"_rxs.pt"))
torch.save(out_means_k, mpath(run_name + "_k.pt"))
torch.save(centers_k, mpath(run_name + "_ctrs.pt"))

d = {str(i): transit_history_k[i] for i in range(len(transit_history_k))}
with open(mpath(run_name+'_exits.json'), 'w') as f:
    json.dump(d, f)
# <split>
# plt.contourf(X,Y, V_surface, levels=np.linspace(V_surface_min, 15, 35), cmap = 'mycmap')
# plt.contour(X, Y, net(grid_input).detach()[:,:,0].numpy(), levels = np.linspace(0.1, 0.9, n_windows), cmap = 'mycmap2')
# plt.savefig(mpath(run_name+"_committor.pdf"))
# #plt.close()
# print("SAVING")
# torch.save(net.state_dict(), mpath(run_name+ ".pt"))
# torch.save(running_xs, mpath(run_name+"_rxs.pt"))
# <\block H>



