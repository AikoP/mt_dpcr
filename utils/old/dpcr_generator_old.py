import numpy as np
import scipy as sp
import scipy.spatial
import threading
import time
import torch
import utils.dpcr_utils as utils


def growHidden(P, E_0, gamma = 1.2, iterations = 1):

    # P     : List of N points with shape (N, d)
    # E_0   : List of 0/1 or True/False with shape (N,1) or (N) that indicates if a point is visible
    # gamma : parameter that decides how large the 'edge circle' is around hidden points

    assert(E_0.shape[0] == P.shape[0])

    E = np.copy(E_0).astype(bool)
    H = np.zeros(P.shape[0], dtype = bool)    # mask of hidden points

    for i in range(iterations):

        # break if there are no edge points left
        if not np.any(E):
            break

        H_prev = np.copy(H).astype(bool)

        # print (H_prev == False)
        # print (E)

        # Compute pairwise distances between edge points and all *visible* points
        dists = sp.spatial.distance.cdist(P[E], P[H_prev == False])

        # eliminate zero-distances to self
        dists[dists == 0] = np.max(dists) + 1

        # hide edge points
        H[E] = True    

        # find all visible points within the edge thresholds (the new edge)
        E = np.zeros(P.shape[0]).astype(bool)
        # print (dists.shape)
        # print (E.shape)
        # print (E[H_prev == False].shape)
        # print (np.any(dists <= (gamma * np.min(dists, axis = 1))[:, None], axis = 0).shape)
        E[H_prev == False] = np.any(dists <= (gamma * np.min(dists, axis = 1))[:, None], axis = 0)

        # remove all hidden points from the edge set (E_n+1 = E_n \ H_n)
        # E[H] = False    

    return E, H

def connectPoints(points, N, noise=0):

    if (N <= points.shape[0]):
        return points

    lengths = np.linalg.norm(points - np.vstack((points[1:], points[0])), axis = 1)
    total_length = np.sum(lengths)

    counts = np.ceil(lengths / total_length * N).astype(int)
    total_count = np.sum(counts)

    connected = np.zeros((total_count, points.shape[1]))
    cum_count = 0

    for i in range(counts.shape[0]):
        connected[cum_count:cum_count + counts[i]] = np.linspace(points[i], points[(i+1)%points.shape[0]], counts[i], endpoint = False)
        cum_count += counts[i]

    if (noise != 0):
        connected += np.random.normal(0, noise, connected.shape)

    return connected

def genRegularNgon(N = 4, p = 100, off = 0.1, noise = 0.03):

    t = np.linspace(0, 2*np.pi, N, endpoint = False)
    points = np.array([np.cos(t), np.sin(t)])
    points = np.swapaxes(points, 0, 1)

    if (off > 0):
        points += np.random.normal(0, off, points.shape)

    return connectPoints(points, p, noise = noise)

def getNearestHidden(pts, hidden):

    return np.copy(hidden[np.argmin(sp.spatial.distance.cdist(pts, hidden), axis = 1)])


def genCircle(p = 100, noise = 0):

    t = np.linspace(0, 2*np.pi, p, endpoint = False)
    points = np.array([np.cos(t), np.sin(t)])
    points = np.swapaxes(points, 0, 1)

    return points

def getTrainingArray(N, resolution, max_iter, gamma = 1.1):

    trainArr = N * [None]

    for i in range(N):

        ngon = genRegularNgon(N = np.random.randint(3,6), p = resolution, off = 0.3, noise = 0)

        mask = np.zeros(ngon.shape[0]).astype(bool)
        mask[np.random.randint(0, ngon.shape[0]-1, 3)] = True

        E, H = growHidden(ngon, mask, gamma = gamma, iterations = np.random.randint(1, max_iter))

        trainArr[i] = np.concatenate((ngon, E[:, None], H[:, None]), axis = 1)

        pts = ngon[H == False]
        edge_mask = E[H == False]

        nearest_hidden = np.zeros(pts.shape)
        nearest_hidden[edge_mask] = getNearestHidden(pts[edge_mask], ngon[H]) - pts[edge_mask]

        trainArr[i] = np.concatenate((pts, nearest_hidden, edge_mask[:,None]), axis = 1)

    return trainArr


def getTrainingArrayFromModel(model, N, max_iter, gamma = 1.1):

    trainArr = N * [None]
    
    for i in range(N):

        mask = np.zeros(model.shape[0]).astype(bool)
        mask[np.random.randint(0, model.shape[0]-1, 5)] = True

        E, H = growHidden(model, mask, gamma = gamma, iterations = np.random.randint(1, max_iter))

        trainArr[i] = np.concatenate((model, E[:, None], H[:, None]), axis = 1)

        pts = model[H == False]
        edge_mask = E[H == False]

        nearest_hidden = np.zeros(pts.shape)

        nearest_hidden[edge_mask] = getNearestHidden(pts[edge_mask], model[H]) - pts[edge_mask]

        trainArr[i] = np.concatenate((pts, nearest_hidden, edge_mask[:,None]), axis = 1)

    return trainArr

def getTrainingArrayFromModelThread(tid, output, model, N, max_iter, gamma):

    output[tid] = getTrainingArrayFromModel(model, N, max_iter, gamma = gamma)

def getTrainingArrayFromModelThreaded(model, N, max_iter, gamma = 1.1, thread_count = 8):

    samples_per_thread = int(N / thread_count)
        
    outputs = thread_count * [None]

    threads = list()
    for tid in range(thread_count):
        t = threading.Thread(target=getTrainingArrayFromModelThread, args=(tid, outputs, model, samples_per_thread, max_iter, gamma,))
        threads.append(t)
        t.start()

    for thread in threads:
        thread.join()

    trainArr = []

    start = time.time()

    for output in outputs:
        trainArr += output

    # print ("Joining time:", time.time() - start)

    return trainArr


def getTrainingArrayIDsFromModel(model, N, max_iter, gamma = 1.1):

    trainArr = N * [None]
    
    for i in range(N):

        mask = np.zeros(model.shape[0]).astype(bool)
        mask[np.random.randint(0, model.shape[0]-1, 5)] = True

        E, H = growHidden(model, mask, gamma = gamma, iterations = np.random.randint(1, max_iter))

        trainArr[i] = torch.from_numpy(np.argwhere(H == False).flatten()).long()

    return trainArr

def getTrainingArrayIDsFromModelThread(tid, output, model, N, max_iter, gamma):

    output[tid] = getTrainingArrayIDsFromModel(model, N, max_iter, gamma = gamma)

def getTrainingArrayIDsFromModelThreaded(model, N, max_iter, gamma = 1.1, thread_count = 8):

    samples_per_thread = int(np.floor(N / thread_count))
        
    outputs = thread_count * [None]

    threads = list()
    for tid in range(thread_count):
        t = threading.Thread(target=getTrainingArrayIDsFromModelThread, args=(tid, outputs, model, samples_per_thread, max_iter, gamma,))
        threads.append(t)
        t.start()

    for thread in threads:
        thread.join()

    trainArr = []

    start = time.time()

    for output in outputs:
        trainArr += output

    if len(trainArr) < N:
        trainArr += getTrainingArrayIDsFromModel(model, N - len(trainArr), max_iter, gamma = gamma)

    return trainArr


def growNeighborhoods(P, V_0, k=6, n=4, noise=0):

    """
    P:     torch tensor of N points with size (N, d)
    V_0:   torch tensor of 0/1 with size (N,1) or (N) that indicates if a point is visible (=1) or not (=0)
    k:     number of nearest neighbors to expand per iteration
    n:     number of iterations applied
    noise: control the proportion of how many random points of the mask are changed back to visible per iteration (max = 1.0)

    returns:    a list L (of length n+1) of torch.uint8 tensors of size (2, N) where:
                    L[i][0] are the visible indices (indices where the visibility mask in that iteration is 1)
                    L[i][1] is the edge mask E_i relative to the indices (defines which of the visible points given by the indices are edge points)
                    
                for iteration i (i = 0,...,n) respectively

    """

    N = P.size(0)
    d = P.size(1)

    assert(V_0.size(0) == N)
    
    nn = utils.knn(P, P.size(0) - 1, q=5000)

    device = V_0.device

    # compute first edge mask E_0
    E_0 = V_0.expand(N, -1).gather(1, nn[:,:k]).sum(dim=1) < k
    # E_0[V_0 == 0] = 0   # <- not needed

    L = (n+1) * [None]
    L[0] = torch.stack([torch.arange(N, device=device)[V_0 == 1], E_0[V_0 == 1].long()], dim=1)

    V_prev = V_0

    for i in range(1, n+1):
        
        # compute visibility mask V_i
        
        V_i = torch.ones(N, device=P.device, dtype=torch.uint8)
        # V_prev = M[i-1, 0]
        
        V_i[V_prev == 0] = 0
        V_i[nn[V_prev == 0, :k].reshape((-1,)).unique()] = 0
        
        # add noise
        if noise > 0:

            idx = torch.arange(N, dtype=torch.long)[V_i == 0]
            nc = int(noise * idx.size(0))
            random_idx = idx[torch.randperm(idx.size(0))[:nc]]

            V_i[random_idx] = 1
            
        # compute edge mask E_i
        E_i = V_i.expand(N, -1).gather(1, nn[:,:k]).sum(dim=1) < k
        # E_i[V_i == 0] = 0     # <- not needed

        L[i] = torch.stack([torch.arange(N, device=device)[V_i == 1], E_i[V_i == 1].long()], dim=1)

        V_prev = V_i

    return L


def getData(pts_tensor, size):

    device = pts_tensor.device
    N = pts_tensor.size(0)

    train_list = []

    for i in range(size):
        
        h = np.random.randint(3, 16)    # number of initially hidden points (3-15)
        n = np.random.randint(3, 11)    # number of interations (3-10)
        
        # create initial mask of k hidden points (3-15)
    
        V_0 = torch.ones(N, dtype=torch.uint8).to(device)
        V_0[torch.randint(N, (h,)).to(device)] = 0
        
        train_list += growNeighborhoods(pts_tensor, V_0, k = 6, n = n, noise = 0.1)

    # return torch.cat(train_list)

    return train_list


