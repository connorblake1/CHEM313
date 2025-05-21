import torch
import os
import json
from json import JSONDecodeError

max_K = 2

def masked_softargmax(x,mask):
    x = x.masked_fill(~mask.to(torch.bool), float('-inf'))
    return torch.softmax(x,dim=-1)

def cnmsam(net, t, mask):  # call net masked soft arg max
    return masked_softargmax(net(t),mask)

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
