import torch
import time
import sampling
from global_utils import cnmsam

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float64)
torch.set_default_tensor_type(torch.DoubleTensor)
    
def half_loss(net, data, a_targets, b_targets, i_a, i_b, cmask):
    a_predictions = cnmsam(net, data, cmask)[...,i_a]
    b_predictions = cnmsam(net, data, cmask)[...,i_b]
    individual_losses = (torch.square(a_predictions - a_targets) + (torch.square(b_predictions - b_targets)))
    loss = torch.mean((torch.square(a_predictions - a_targets) + (torch.square(b_predictions - b_targets))))
    return loss, individual_losses
    
def half_log_loss(net, data, a_targets, b_targets, a_long_targets, b_long_targets, i_a, i_b, cmask):
    a_predictions = cnmsam(net,data,cmask)[...,i_a]
    b_predictions = cnmsam(net,data,cmask)[...,i_b]
    a_indices = torch.where(a_targets != 0)
    b_indices = torch.where(b_targets != 0)
    
    a_predictions, a_targets = a_predictions[a_indices], a_targets[a_indices]
    b_predictions, b_targets = b_predictions[b_indices], b_targets[b_indices]
    
    a_indices = torch.where(a_targets != 1)
    b_indices = torch.where(b_targets != 1)
    
    a_predictions, a_targets = a_predictions[a_indices], a_targets[a_indices]
    b_predictions, b_targets = b_predictions[b_indices], b_targets[b_indices]

    return (torch.mean(torch.square(torch.log(a_predictions) - torch.log(a_targets)))) + (torch.mean(torch.square(torch.log(b_predictions) - torch.log(b_targets)))), None #individual_losses
    
def half_log_loss_multi(net, data, targets, cmask):
    predictions = cnmsam(net, data, cmask) # [N,K]
    mask = ((targets != 0) & (targets != 1)).all(dim=1)
    predictions, targets = predictions[mask], targets[mask]
    return torch.mean(torch.square(torch.log(predictions) - torch.log(targets)))