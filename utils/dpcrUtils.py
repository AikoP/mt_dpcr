import torch
import numpy as np
import random
import itertools

def cdist(a, b = None):

    """
        a: (C1 x ... x Cn x N x d) tensor
        b: (C1 x ... x Cn x M x d) tensor [optional]
    """

    aa = torch.sum(torch.square(a), dim=-1, keepdim=True)   # size: (C1 x ... x Cn x N x 1)

    if b != None:
        bb = torch.sum(torch.square(b), dim=-1, keepdim=True)                               # size: (C1 x ... x Cn x M x 1)
        inner = torch.matmul(2.0 * a, b.transpose(-1, -2))                                  # size: (C1 x ... x Cn x N x M)
        return (aa - inner + bb.transpose(-1, -2)).clamp(min=torch.finfo(a.dtype).eps)      # size: (C1 x ... x Cn x N x M)

    else:
        inner = torch.matmul(2.0 * a, a.transpose(-1, -2))                                  # size: (C1 x ... x Cn x N x N)
        return (aa - inner + aa.transpose(-1, -2)).clamp(min=torch.finfo(a.dtype).eps)      # size: (C1 x ... x Cn x N x N)   

def drawBinBatches(bin_sizes, batchsize=10):

    binbatches = []
    
    for i in range(len(bin_sizes)):
        
        batch_count = int(np.floor(bin_sizes[i] / batchsize))
        
        for batch in np.random.choice(bin_sizes[i], (batch_count, batchsize), replace=False):
            
            binbatches.append((i, batch))
            
    random.shuffle(binbatches)
        
    return binbatches

def getBins(sizes, b=10):

    """
        sample_list:    a python list of integres that represent the number of points in a sample
        b:              number of bins

        returns:    a python list of tensors that represent the ids of samples

    """

    N = len(sizes)  # total number of samples
    k = N / b       # estimated samples per bin

    # sorted_sample_ids = np.arange(N)[np.argsort(sizes)] # list of sample indices, sorted by size of samples
    sorted_sample_ids = np.argsort(sizes) # list of sample indices, sorted by size of samples    

    limits      = [ (int(np.ceil(i * k)), int(np.ceil((i+1) * k)))  for i in range(b)]  # (index) limits of each bin
    bins        = [ []                                              for i in range(b)]  # the bins

    for (i, limit) in enumerate(limits):
        
        for index in sorted_sample_ids[limit[0]:limit[1]]:
            
            bins[i].append(index)     

        bins[i] = torch.from_numpy(np.array(bins[i])).long()  

    return bins

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


def matchPointsBatched(x, y, perms=None, y_perms=None):

    """

    Input:

        x - (B x N x k x d) tensor of points
        y - (B x N x k x d) tensor of points

            B - batch size
            N - number of cases to match
            k - number of points to match per case
            d - dimension of single points

        perms   - (k! x k) tensor of permutations to test (optional) (same for each batch, for each point)
        y_perms - (B x N x k! x k x d) tensor of permutations of input y

    """

    B = x.size(0)
    N = x.size(1)
    k = x.size(2)
    d = x.size(3)

    if y_perms is not None or perms is not None:
        assert y_perms is not None and perms is not None

    if y_perms is None:

        if perms is None:
            perms = torch.from_numpy(np.asarray(list(itertools.permutations(np.arange(k))))).long().to(x.device)
 
        # for each sample in batch, create all permutations of 'sample points' in y
        y_perms = torch.zeros((B, N, perms.size(0), k, d), device=x.device)
        for (i, p) in enumerate(perms):
            y_perms[:,:,i] = y[:,:, p, :]

    # compute permutation that minimizes distance from 'sample points' in x to y
    D = x[:, :, None, :, :] - y_perms               # size: (B, N, k!, k, d)
    D.square_()
    D = D.sum(dim = -1) # size: (B, N, k!, k)
    D.sqrt_()
    D = D.sum(-1)     # size: (B x N x k!)

    return perms[D.argmin(dim=-1)]


def fac(n):

    k = 1

    for i in range(2,n+1):
        k *= i 

    return k

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

def getDevice(io = None, quiet = False):

    device = torch.device("cpu")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        if not quiet:
            out = "Using %d CUDA devices (%s)" % (torch.cuda.device_count(), torch.cuda.get_device_name(0))
            if io != None:
                io.cprint(out)
            else:
                print (out)

    else:
        if not quiet:
            out = "Using CPU"
            if io != None:
                io.cprint(out)
            else:
                print (out)

    return device


def getRotation(d, theta, device=torch.device('cpu'), dtype=torch.float):
    """
        input:
        d       - number of dimensions (must be integer > 1)
        theta   - angles to rotate in each plane spanned by two orthonormal vectors (must be a list of (d^2 - d) / 2 entries))

        returns: a (d x d) rotation matrix 
    """

    assert d > 1
    assert d**2 - d == 2 * len(theta)

    rotation = torch.eye(d, device=device, dtype=dtype)

    r = 0

    for i in range(d):
        for j in range(d):
            if j == i:
                break
            sin_theta = np.sin(theta[r])
            cos_theta = np.cos(theta[r])
            rot = torch.eye(d, device=device, dtype=dtype)
            rot[i,i] = cos_theta
            rot[j,j] = cos_theta
            rot[i,j] = sin_theta
            rot[j,i] = -sin_theta
            rotation = rotation.matmul(rot)
            r += 1

    return rotation

def getRandomRotation(d, device=torch.device('cpu'), dtype=torch.float):
    return getRotation(d, np.random.uniform(0.0, high=2*np.pi, size = (d**2 - d) // 2).tolist(), device=device, dtype=dtype)