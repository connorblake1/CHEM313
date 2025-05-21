import torch

max_K = 5

def masked_softargmax(x,mask):
    x = x.masked_fill(~mask.to(torch.bool), float('-inf'))
    return torch.softmax(x,dim=-1)

def cnmsam(net, t, mask):  # call net masked soft arg max
    return masked_softargmax(net(t),mask)

def dist(x, y):
    return torch.sqrt(torch.sum(torch.square(x-y), axis = -1))
