import numpy as np
import random
import time
import torch
import itertools

def getBins(sizes, b=10):

    """
        sample_list:    a python list of integres that represent the number of points in a sample
        b:              number of bins

        returns:    a python list of tensors that represent the ids of samples

    """

    N = len(sizes)  # total number of samples
    k = N / b       # estimated samples per bin

    sorted_sample_ids = np.arange(N)[np.argsort(sizes)] # list of sample indices, sorted by size of samples    

    limits      = [ (int(np.ceil(i * k)), int(np.ceil((i+1) * k)))  for i in range(b)]  # (index) limits of each bin
    bins        = [ []                                              for i in range(b)]  # the bins

    for (i, limit) in enumerate(limits):
        
        for index in sorted_sample_ids[limit[0]:limit[1]]:
            
            bins[i].append(index)     

        bins[i] = torch.from_numpy(np.array(bins[i])).long()  

    return bins


def getSubSamplesFromBins(bins, sizes):
    
    subsampled_bins = []

    for bin in bins:

        # get minumum sample size from current bin
        subsample_size = np.min(np.array(sizes))

        # initialize subsampled bin
        subsampled_bin = torch.zeros((len(bin), subsample_size, bin[0].size(1)))

        for (i, sample) in enumerate(bin):
  
            subsampled_bin[i] = sample[torch.randperm(sample.size(0))[:subsample_size]]
            
            # # compute sample ratio (percentage of how many points from sample are kept)
            # subsample_ratio = subsample_size / sample.size(0)

            # # compute number of points in sample that are from either class 0 or class 1
            # c0_samples = np.sum(sample[:,class_index] == False)
            # c1_samples = np.sum(sample[:,class_index] == True)

            # # compute number of points per class in subsample (note: c1 is preferred, since it usually is the smaller class)
            # c1_subsamples = int(np.ceil(c1_samples * subsample_ratio))
            # c0_subsamples = subsample_size - c1_subsamples

            # # compute subsamples
            # subsampled_bin[i, :c0_subsamples, :] = sample[sample[:,class_index] == False][np.random.randint(0, c0_samples-1, c0_subsamples)]
            # subsampled_bin[i, c0_subsamples:, :] = sample[sample[:,class_index] == True][np.random.randint(0, c1_samples-1, c1_subsamples)]

        subsampled_bins.append(subsampled_bin)
        
    return subsampled_bins


def drawBinBatches(bin_sizes, batchsize=10):

    binbatches = []
    
    for i in range(len(bin_sizes)):
        
        batch_count = int(np.floor(bin_sizes[i] / batchsize))
        
        for batch in np.random.choice(bin_sizes[i], (batch_count, batchsize), replace=False):
            
            binbatches.append((i, batch))
            
    random.shuffle(binbatches)
        
    return binbatches


def knn(x, k, q=5000):

    """
        Computes the k-nearest neighbors of an (n x d) array  of n points in d dimensions

        Input:

            x - (n x d) array
            k - the number of nearest neighbors to compute
            q - number of points to compute in parallel (due to memory limits)

        Output:

            topk - (n x k) array of nearest neighbor indices

    """
    
    N = x.size(0)

    if (N <= q):
        
        xx = torch.sum(x**2, dim=1, keepdim=True)
        D = xx.transpose(0, 1) - 2.0 * torch.matmul(x, x.transpose(0, 1)) + xx
        
        return D.topk(k=k+1, dim=1, largest=False)[1][:,1:]
    
    else:
        
        topk = torch.zeros(N, k, dtype=torch.long, device=x.device)

        for i in range(0, N, q):

            aa = torch.sum(x[i:i+q]**2, dim=1, keepdim=True)
            bb = torch.sum(x**2, dim=1, keepdim=True)
            ab = torch.matmul(x[i:i+q], x.transpose(0, 1))
            
            D = aa - 2.0 * ab + bb.transpose(0, 1)
            
            topk[i:i+q, :] = D.topk(k=k+1, dim=1, largest=False)[1][:,1:]
        
        return topk

def matchPoints(x, y, perms=None, y_perms=None):

    """

    Input:

        x - (N x k x d) tensor of points
        y - (N x k x d) tensor of points

            N - number of cases to match
            k - number of points to match per case
            d - dimension of single points

        perms   - (k! x k) tensor of permutations to test (optional)
        y_perms - (N x k! x k x d) tensor of permutations of input y

    """

    N = x.size(0)
    k = x.size(1)
    d = x.size(2)

    if y_perms is not None or perms is not None:
        assert y_perms is not None and perms is not None

    if y_perms is None:

        if perms is None:
            perms = torch.from_numpy(np.asarray(list(itertools.permutations(np.arange(k))))).long().to(x.device)
 
        # create all permutations of 'sample points' in y
        y_perms = torch.zeros((N, perms.size(0), k, d), device=x.device)
        for (i, p) in enumerate(perms):
            y_perms[:,i] = y[:, p, :]

    # compute permutation that minimizes distance from 'sample points' in x to y

    D = x[:,None,:,:] - y_perms     # size: (N, k!, k, d)
    torch.mul(D, D, out=D)
    D = torch.sum(D, dim=3)         # size: (N, k!, k)
    torch.sqrt(D, out=D)
    D = torch.sum(D, dim=2)         # size: (N, k!)

    return perms[D.argmin(dim=1)]   # size: (N)

def fac(n):

    k = 1

    for i in range(2,n+1):
        k *= i 

    return k


def mergeClusters(pts, t, max_iter = 5):
    
    if (max_iter < 0):
        print ("reached max iterations!")
        return pts

    N = pts.size(0)

    aa = torch.sum(pts**2, dim=1, keepdim=True)
    bb = torch.sum(pts**2, dim=1, keepdim=True)
    ab = torch.matmul(pts, pts.transpose(0, 1))

    D = aa - 2.0 * ab + bb.transpose(0, 1)

    del aa
    del bb
    del ab

    idx_cluster = torch.arange(N)[torch.sum(D < t**2, dim=1) > 1]
    
    # if no clusters are found
    if (idx_cluster.size(0) == 0):
        print ("terminated mergeClusters! (%d iterations)" % (5 - max_iter))
        del D
        torch.cuda.empty_cache()
        return pts    
    
    idx_non_cluster = torch.arange(N)[torch.sum(D < t**2, dim=1) < 2]

    E_N = (torch.max(D) + 1) * torch.eye(N, device = pts.device)

    # compute for each point the index of the nearest point
    m = torch.argmin(D + E_N, dim = 1)

    del D
    
    # mark backrefs
    for i in range(m.size(0)):
        if m[i] > -1 and m[m[i]] == i:
            m[m[i]] = -1

    # keep only those points, where the distance to the nearest neighbor is less than t
    m = m[idx_cluster]
    
    # compute means of the cluster pairs
    clusters = 0.5 * (pts[idx_cluster[[m > -1]]] + pts[m[[m > -1]]])
    
    # generate new point cloud
    new_pts = torch.cat([pts[idx_non_cluster], clusters], dim=0)

    torch.cuda.empty_cache()
    return mergeClusters(new_pts, t, max_iter = max_iter - 1)

def exportPLY(data, colors, path, name = 'model'):

    """
    input: 
        data - (N,d) numpy array with point coordinates
        colors - (N,3) numpy array with rgb color values (0-255)
        path - output path
        name - output name
    """

    N = data.shape[0]

    assert(N == colors.shape[0])

    f = open(path + "\\" + name + ".ply", "w")

    f.write("ply\n")
    f.write("format ascii 1.0\n")
    f.write("element vertex %d\n" % (N))
    f.write("property float32 x\n")
    f.write("property float32 y\n")
    f.write("property float32 z\n")
    f.write("property uchar red\n")
    f.write("property uchar green\n")
    f.write("property uchar blue\n")
    f.write("end_header\n")

    for i in range(N):
        v = data[i]
        c = colors[i]
        f.write(' '.join(map(str, v)) + " ")
        f.write(' '.join(map(str, c)) + "\n")
        
    f.close()

if __name__ == '__main__':

    print ("utils working!")