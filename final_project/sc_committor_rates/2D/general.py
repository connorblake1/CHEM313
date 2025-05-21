import sampling
import training
import torch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from config import V, dim, a_center, b_center, n_windows, cutoff, n_reporter_steps, batch_size, CommittorNet, run_name, x, y, beta, gamma, step_size, nice_name, mpath
import json
import os
from json import JSONDecodeError


RATES_FILE = "rates_db.json"


def append_rate(a_rate, b_rate):
    # Try to load the existing database; if it doesn't exist or is empty/corrupt, start fresh
    if os.path.exists(RATES_FILE) and os.path.getsize(RATES_FILE) > 0:
        try:
            with open(RATES_FILE, "r") as f:
                db = json.load(f)
        except JSONDecodeError:
            db = {}
    else:
        db = {}

    # Append to the list for this run
    db.setdefault(run_name, []).append((a_rate, b_rate))

    # Write back atomically
    tmp_file = RATES_FILE + ".tmp"
    with open(tmp_file, "w") as f:
        json.dump(db, f, indent=2)
    os.replace(tmp_file, RATES_FILE)


# For plotting
matplotlib.rcParams['text.usetex'] = False
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list("mycmap", ['#1B346C','#01ABE9','#F1F8F1','#F54B1A'])
cmap2 = LinearSegmentedColormap.from_list("mycmap2", ['#ffffff','#000000','#ffffff'])
cmap3 = LinearSegmentedColormap.from_list("mycmap3", ['#1B346C','#ffffff','#F54B1A'])
cmap4 = LinearSegmentedColormap.from_list("mycmap4", ['#ffffff','#F54B1A'])
cmap5 = LinearSegmentedColormap.from_list("mycmap5", ['#000000','#000000'])
cmap6 = LinearSegmentedColormap.from_list("mycmap5", ['#ffffff','#ffffff'])
matplotlib.colormaps.register(name="mycmap",cmap=cmap)
matplotlib.colormaps.register(name="mycmap2",cmap=cmap2)
matplotlib.colormaps.register(name="mycmap3",cmap=cmap3)
matplotlib.colormaps.register(name="mycmap4",cmap=cmap4)
matplotlib.colormaps.register(name="mycmap5",cmap=cmap5)
matplotlib.colormaps.register(name="mycmap6",cmap=cmap6)
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
torch.set_default_dtype(torch.float64)
torch.set_default_tensor_type(torch.DoubleTensor)
# Set a deterministic RNG seed
torch.manual_seed(42)

# Define our two-channel potential

def dist(x, y):
    return torch.sqrt(torch.sum(torch.square(x-y), axis = -1))

# Initialize evenly-spaced configurations that linearly interpolate the basin centers
init_data = torch.zeros((n_windows, dim)) 
for d in range(dim):
    init_data[:,d] = torch.linspace(a_center[d], b_center[d], n_windows)

net = CommittorNet(dim=dim).to(device).double()

print("Initial representation of the committor has been trained!")

# Run the optimization
n_trials = 1
n_opt_steps = 700
optimizer = torch.optim.Adam(net.parameters(), lr = 1e-2)
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

print("Calculating Flux...")
a_escape_times, a_escape_confs = sampling.flux_sample(V, beta, gamma, step_size, a_center, b_center, cutoff, 1000, stride = 1)
b_escape_times, b_escape_confs = sampling.flux_sample(V, beta, gamma, step_size, b_center, a_center, cutoff, 1000, stride = 1)

a_escape_confs_list = a_escape_confs.clone().reshape([-1,dim]).to(device).double().detach()
b_escape_confs_list = b_escape_confs.clone().reshape([-1,dim]).to(device).double().detach()

escape_confs = torch.cat([a_escape_confs, b_escape_confs], axis = 0).reshape(-1,dim)

running_a_exit_confs = []
running_b_exit_confs = []
a_exit_indices = torch.randperm(len(a_escape_confs_list))
b_exit_indices = torch.randperm(len(b_escape_confs_list))
a_transit_history = [0]
b_transit_history = [0]
for trial in range(1): # Can run multiple trials, if you'd like
    a_transit = False
    last_a_transit = 0
    b_transit = False
    last_b_transit = 0
    a_means = []
    a_times = []
    b_means = []
    b_times = []
    losses = []
    log_losses = []
    a_index_counter = 0
    b_index_counter = 0
    for step in range(n_opt_steps):
        optimizer.zero_grad()
        net.zero_grad()

        # Sample some configurations
        
        if step == 0 or a_transit == True:
            a_xs = a_escape_confs_list[a_exit_indices[a_index_counter]]
            a_xs = a_xs.unsqueeze(0)
            a_transit = False
            running_a_exit_confs.append(a_xs.cpu().numpy())
        
        else:
            a_reporter_energies = V(a_running_short_reporters.reshape([-1,dim])[last_a_transit:])  # noqa: F821
            a_weights = torch.sigmoid(net(a_running_short_reporters.squeeze().reshape([-1, dim])))[last_a_transit:].detach() # noqa: F821
            a_weights = torch.where(dist(a_running_short_reporters.squeeze().reshape([-1, dim])[last_a_transit:], a_center) < cutoff, 0, a_weights) # noqa: F821
            _, a_indices = torch.sort(a_weights, descending = True)
            a_indices = a_indices[:1]
            a_xs = a_running_short_reporters.squeeze().reshape([-1, dim])[last_a_transit:][a_indices]# noqa: F821
        
        if step == 0 or b_transit == True:
            b_xs = b_escape_confs_list[b_exit_indices[b_index_counter]]
            b_xs = b_xs.unsqueeze(0)
            b_transit = False
            running_b_exit_confs.append(b_xs.cpu().numpy())
        else:
            b_reporter_energies = V(b_running_short_reporters.reshape([-1,dim])[last_b_transit:])# noqa: F821
            b_weights = torch.sigmoid(-net(b_running_short_reporters.squeeze().reshape([-1, dim])))[ last_b_transit:].detach()# noqa: F821
            b_weights = torch.where(dist(b_running_short_reporters.squeeze().reshape([-1, dim])[ last_b_transit:], b_center) < cutoff, 0, b_weights)# noqa: F821
            _, b_indices = torch.sort(b_weights, descending = True)
            b_indices = b_indices[:1]
            b_xs = b_running_short_reporters.squeeze().reshape([-1, dim])[last_b_transit:][b_indices]# noqa: F821
    
        a_short_reporters, a_short_times = sampling.take_reporter_steps(a_xs, V, beta, gamma, step_size, n_reporter_trajectories, n_reporter_steps, a_center, b_center, cutoff, adaptive = True)
        b_short_reporters, b_short_times = sampling.take_reporter_steps(b_xs, V, beta, gamma, step_size, n_reporter_trajectories, n_reporter_steps, a_center,  b_center, cutoff, adaptive = True) # CB: why are A and B not swapped? Ohh it's because the committor is not swap invariant
        a_times.append(np.mean(a_short_times))
        b_times.append(np.mean(b_short_times))

        # Keep a memory of sampled configurations
        if step == 0:
            a_running_xs = a_xs.detach()
            b_running_xs = b_xs.detach()
            running_xs = torch.cat((a_running_xs, b_running_xs), axis = 0)
            a_running_short_reporters = a_short_reporters
            b_running_short_reporters = b_short_reporters
            running_short_reporters = torch.cat((a_running_short_reporters, b_running_short_reporters), axis = 0)
        else:
            a_running_xs = torch.cat((a_running_xs.detach(), a_xs.detach()))
            b_running_xs = torch.cat((b_running_xs.detach(), b_xs.detach()))
            a_running_short_reporters = torch.cat((a_running_short_reporters.detach(), a_short_reporters), axis = 0)
            b_running_short_reporters = torch.cat((b_running_short_reporters.detach(), b_short_reporters), axis = 0)
            running_short_reporters = torch.cat((a_running_short_reporters, b_running_short_reporters), axis = 0)
            running_xs = torch.cat((a_running_xs, b_running_xs), axis = 0)
        
        for j in range(1): # Can go through multiple optimization steps per data collection step, if needed
            with torch.no_grad():
                a_short_targets, b_short_targets, a_short_var, a_short_means = sampling.calculate_committor_estimates(running_short_reporters.reshape([-1, dim]), net, a_center, b_center, cutoff, n_reporter_trajectories)
            batch_size = running_xs.size()[0]
            for m in range(100):
                permutation = torch.randperm(running_xs.size()[0])
                for i in range(0, len(running_xs), batch_size):
                    print(m)
                    optimizer.zero_grad()
                    net.zero_grad()
                    indices = permutation[i:i+batch_size]
                    loss, individual_losses =  1*training.half_loss(net, running_xs.to(device)[indices], a_short_targets.to(device)[indices], b_short_targets.to(device)[indices])
                    log_loss, individual_log_loss = 1*training.half_log_loss(net, running_xs.to(device)[indices], a_short_targets.to(device)[indices], b_short_targets.to(device)[indices], a_short_targets.to(device)[indices], b_short_targets.to(device)[indices])
                    total_loss = 0*loss + 1*log_loss
                    total_loss.backward()
                    optimizer.step()
        
        # Estimate rates
        with torch.no_grad():
            a_exit_tensor = torch.tensor(np.array(a_escape_confs)).squeeze()
            b_exit_tensor = torch.tensor(np.array(b_escape_confs)).squeeze()
            a_rate_estimates = 1/torch.mean(a_escape_times)*torch.mean(torch.sigmoid(net(a_exit_tensor)))
            b_rate_estimates = 1/torch.mean(b_escape_times)*torch.mean(1 - torch.sigmoid(net(b_exit_tensor)))

            a_rate_mean = torch.mean(a_rate_estimates)
            a_local = float(a_rate_mean.cpu().detach().numpy())
            a_means.append(a_rate_mean.cpu().detach().numpy())
            b_rate_mean = torch.mean(b_rate_estimates)
            b_local = float(b_rate_mean.cpu().detach().numpy())
            b_means.append(b_rate_mean.cpu().detach().numpy())
            append_rate(a_local, b_local)
        
        # Report to the command line
        print(f"Step {step}: Rate Estimate = {a_rate_mean.item()}; Loss = {log_loss.item()}")#, Log Loss 2 = {log_loss_2.item()}")
        print(a_index_counter, b_index_counter)
        

        # Check whether or not a sampling chain has reached the basin
        if torch.min(torch.sqrt(torch.sum(torch.square(a_running_short_reporters.reshape([-1, dim])[last_a_transit:] - b_center), axis = -1))) < cutoff:
            print("A Transit!")
            a_transit = True
            last_a_transit = len(a_running_short_reporters.reshape([-1, dim]))
            a_index_counter += 1
            a_transit_history.append(int(len(running_xs.reshape([-1, dim]))/2))
            
        if torch.min(torch.sqrt(torch.sum(torch.square(b_running_short_reporters.reshape([-1, dim])[last_b_transit:] - a_center), axis = -1))) < cutoff:
            print("B Transit!")
            b_transit = True
            last_b_transit = len(b_running_short_reporters.reshape([-1, dim]))
            b_index_counter += 1
            b_transit_history.append(int(len(running_xs.reshape([-1, dim]))/2))

        # plotting
        if step % 10 == 9:
            plt.tight_layout()
            fig, axs  = plt.subplot_mosaic([['a', 'b'], ['a', 'b']], width_ratios = [1., 1.])
            fig.set_size_inches(15.2, 4.8)
            axs['a'].contourf(X,Y, V_surface_numpy, levels=np.linspace(V_surface_min, 15, 35), cmap = 'mycmap',zorder=0)
            
            fig.colorbar(axs['a'].contour(X, Y, torch.sigmoid(net(grid_input)).cpu().detach().numpy(), levels = np.linspace(0.1, 0.9, 9), cmap ="spring"), ax = axs['a'], ticks = np.linspace(0, 1, 11))
            axs['a'].scatter(torch.reshape(running_xs, [-1, 2]).detach().numpy()[:,0], torch.reshape(running_xs, [-1, 2]).detach().numpy()[:,1], c = a_short_var, cmap = 'mycmap2', alpha = 1)
            # axs['a'].scatter(torch.reshape(a_running_xs, [-1, 2]).detach().numpy()[-1,0], torch.reshape(a_running_xs, [-1, 2]).detach().numpy()[-1,1], color = 'blue', alpha = 1)
            # axs['a'].scatter(torch.reshape(b_running_xs, [-1, 2]).detach().numpy()[-1,0], torch.reshape(b_running_xs, [-1, 2]).detach().numpy()[-1,1], color = 'green', alpha = 1)
            axs['a'].add_patch(plt.Circle(a_center, cutoff, linewidth = 2, color = 'red', fill = True))
            axs['a'].add_patch(plt.Circle(b_center, cutoff, linewidth = 2, color = 'green', fill = True))
            # axs['a'].text(a_center[0]-.04,a_center[1]-.04, "A", weight = 'bold', size = 15, color = 'white')
            # axs['a'].text(b_center[0]-.04,b_center[1]-.04, "B", weight = 'bold', size = 15, color = 'white')
            axs['a'].plot([], [], color='red', label='A')
            axs['a'].plot([], [], color='green', label='B')
            axs['a'].set_xlabel(r"x", size = 16)
            axs['a'].set_ylabel(r"y", size = 16)

            #axs['a'].contour(X, Y, V_surface.cpu().detach().numpy(), levels = np.linspace(0., 0, 1), cmap = 'mycmap5')
            # axs['a'].contourf(X,Y, V_surface.cpu().detach().numpy(), levels = np.linspace(0, 1e20, 2), cmap = 'mycmap6', zorder = 100)
            axs['a'].set_xlim(x[0],x[-1])
            axs['a'].set_ylim(y[0],y[-1])
            axs['a'].legend()

            

            axs['b'].plot(np.arange(step+1)*2*((n_reporter_trajectories*n_reporter_steps).cpu().detach().numpy() + 1)*step_size.cpu().detach().numpy(), np.array(a_means), c = '#1B346C')
            axs['b'].plot(np.arange(step+1)*2*((n_reporter_trajectories*n_reporter_steps).cpu().detach().numpy() + 1)*step_size.cpu().detach().numpy(), np.array(b_means), c = '#F54B1A')
            # axs['b'].plot(np.arange(len(a_means))*2*((n_reporter_trajectories*n_reporter_steps).cpu().detach()s.numpy() + 1)*step_size.cpu().detach().numpy(), np.ones_like(a_means)*9.18e-8*2, '--', c = 'black')
            axs['b'].set_yscale('log')
            axs['b'].set_xlabel(r"Sampling Time ($\tau$)", size = 12)
            axs['b'].set_ylabel(r"Rate ($\tau^{-1}$)", size = 12)
            axs['b'].legend([r'Rate Estimate A to B', r'Rate Estimate B to A', r'Analytical Rate'], prop={'size': 12})
            axs['b'].set_ylim(1e-8, 1e1)
            plt.tight_layout()
            fig.savefig(mpath(run_name + ".pdf"))
            plt.close()
        

# Plot the final committor
plt.contourf(X,Y, V_surface, levels=np.linspace(V_surface_min, 15, 35), cmap = 'mycmap')
plt.contour(X, Y, net(grid_input).detach().numpy(), levels = np.linspace(0.1, 0.9, n_windows), cmap = 'mycmap2')
plt.savefig(mpath(run_name+"_committor.pdf"))
#plt.close()
print("SAVING")
torch.save(net.state_dict(), mpath(run_name+ ".pt"))


