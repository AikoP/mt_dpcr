import sys
import os
sys.path.insert(1, os.getcwd())  # to load from any submodule in the repo

import torch
import time
from utils import dpcrUtils as utils

device = utils.getDevice()


a = torch.randn(2, 6700, 3).to(device)

def cdist(a, b = None):

    """
        a: (C1 x ... x Cn x N x d) tensor
        b: (C1 x ... x Cn x M x d) tensor [optional]
    """

    aa = torch.sum(torch.square(a), dim=-1, keepdim=True)   # size: (C1 x ... x Cn x N x 1)

    if b != None:
        bb = torch.sum(torch.square(b), dim=-1, keepdim=True)   # size: (C1 x ... x Cn x M x 1)
        inner = torch.matmul(2.0 * a, b.transpose(-1, -2))      # size: (C1 x ... x Cn x N x M)
        return aa - inner + bb.transpose(-1, -2)                # size: (C1 x ... x Cn x N x M)

    else:
        inner = torch.matmul(2.0 * a, a.transpose(-1, -2))      # size: (C1 x ... x Cn x N x N)
        return aa - inner + aa.transpose(-1, -2)                # size: (C1 x ... x Cn x N x N)    
    

def dist_plus(a):
    inner = torch.bmm(a, a.transpose(-1, -2))    # size: (N x N)
    aa = torch.sum(a**2, dim=-1, keepdim=True)      # size: (N x 1)
    # aa = torch.sum(torch.square(a), dim=-1, keepdim=True)
    return aa - 2 * inner + aa.transpose(-1, -2)


def dist_new(a):
    return torch.sum(torch.square(a.unsqueeze(-2) - a.unsqueeze(-3)), dim=-1)  # size: (N x N x d)

def dist_opt(a):
    if a.size(0) * a.size(1) > 13000:
        return dist(a)
    else:
        return dist_new(a)

# start = time.time()
# for i in range(20):
#     D_opt = dist_opt(a)
# end = time.time()
# print ("opt:", end - start, " s")

# del D_opt

start = time.time()
for i in range(10):
    # D = dist(a)
    D = dist_opt(a)
    # D = dist_opt(a)
    if D.size(0) > 0:
        continue
end = time.time()
print ("classic:", end - start, " s")

# del D_old

# start = time.time()
# for i in range(10):
#     D_new = dist_new(a)
# end = time.time()
# print ("new:", end - start, " s")

# del D_new


