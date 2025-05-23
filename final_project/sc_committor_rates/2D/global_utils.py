import torch
import os
import json
from json import JSONDecodeError

max_K = 2
class CommittorNet(torch.nn.Module):
    def __init__(self, dim, K=max_K):
        super(CommittorNet, self).__init__()
        self.dim = dim
        self.K = K
        block = [
            torch.nn.Linear(dim, 50),
            torch.nn.Tanh(),
            torch.nn.Linear(50,50), 
            torch.nn.Tanh(),
            torch.nn.Linear(50,50), 
            torch.nn.Tanh(),
            torch.nn.Linear(50, max_K)
        ]
        self.Block = torch.nn.Sequential(*block)
    
    def forward(self, x):
        prediction = self.Block(x)
        return prediction

def mpath(name):
    return os.path.join("run_data",name)

def masked_softargmax(x,mask):
    x = x.masked_fill(~mask.to(torch.bool), float('-inf'))
    return torch.softmax(x,dim=-1)

def cnmsam(net, t, mask):  # call net masked soft arg max
    p_k = masked_softargmax(net(t),mask) # prob goes to basin k, sum to 1
    return 1 - p_k # P(next basin != k), small and what we want

def dist(x, y):
    return torch.sqrt(torch.sum(torch.square(x-y), axis = -1))

RATES_FILE = "rates_db.json"
def append_rate(run_name, a_rate, b_rate):
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
