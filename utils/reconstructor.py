import sys
sys.path.insert(1, '../')  # to load from any submodule in the repo

import torch
import numpy as np
import time

import utils.dpcrUtils as utils

def mergeClusters(pts, iter_idx, t, r = 5, recency_threshold = 1, verbose=False):

    """
        input:
            - pts: (n x d) tensor
            - iter_idx: (n) tensor of integers, indicating the iteration where a point from 'pts' has been added
            - t: float
            - r: integer, remaining merge iterations. The resursion stops if  r = 0
            - recency_threshold: integer, indicating how many iterations are considered 'recent' and thus defines which points are modifiable (i.e. non-fixed) 
    """

    if r < 1:
        if verbose:
            print ("reached max iterations!")
        return pts, iter_idx

    n = pts.size(0)

    max_iter = int(torch.max(iter_idx).item())
    
    if max_iter < 1:
        return pts, iter_idx

    recency_threshold = min(max_iter, recency_threshold)

    fixed_pts_ids = torch.arange(n)[iter_idx <= (max_iter - recency_threshold)]
    recent_pts_ids = torch.arange(n)[iter_idx > (max_iter - recency_threshold)]

    fixed_pts = pts[fixed_pts_ids]
    recent_pts = pts[recent_pts_ids]

    if recent_pts.size(0) < 1:
        if verbose:
            print ("no points to merge")
        return pts, iter_idx

    if verbose:
        print ("Iterations left: ", r)

    n = recent_pts.size(0)

    D = utils.cdist(recent_pts)
    D = D + (2 * t**2 + torch.max(D)) * torch.eye(n, device=recent_pts.device)

    # get indices of all points that must be merged (relative to all recent points)
    cluster_idx = torch.arange(n)[torch.sum(D < t**2, dim=1) > 0]

    # if no clusters are found
    if cluster_idx.numel() < 1:
        if verbose:
            print ("no clusters - terminated mergeClusters! (%d iterations)" % (5 - r))
        return pts, iter_idx
    
    # get indices of all points that must NOT be merged (relative to all recent points)
    non_cluster_idx = torch.arange(n)[torch.sum(D < t**2, dim=1) < 1]

    # for each (recent) point marked as a cluster, get the index of the nearest (recent) point (relative to all recent points)
    m = torch.argmin(D, dim = 1)

    # references should only be one-directional, so mark excess refs with -1
    bi_dirs = torch.arange(n, device=m.device) == m[m]   # mask that marks all points with backrefs
    uni_idx = torch.arange(n, device=m.device).to(m.device) < m       # mask that makes the selection of ref directions unique
    m[bi_dirs * uni_idx] = -1

    # mark all non-cluster pts with -1
    m[non_cluster_idx] = -1

    # compute means of the cluster pairs
    clusters = 0.5 * (recent_pts[torch.arange(n)[m > -1]] + recent_pts[m[m > -1]])

    # the iteration index of the new cluster points is set to the iteration index of one of the merged points
    clusters_iter_idx = iter_idx[recent_pts_ids][m[m > -1]]

    # remove clustered points
    cluster_mask = torch.ones(pts.size(0), dtype=torch.bool)
    cluster_mask[recent_pts_ids[cluster_idx]] = False

    pts = pts[cluster_mask]
    iter_idx = iter_idx[cluster_mask]

    # add clusters
    pts = torch.cat([pts, clusters])
    iter_idx = torch.cat([iter_idx, clusters_iter_idx])

    return mergeClusters(pts, iter_idx, t, r = r - 1)

def cleanPoints(pts, iter_idx, verbose=False, recency_threshold = 1):

    """
        input:
            - sample: (N x d) tensor
            - new_pts (m x d) tensor
            - pts: (n x d) tensor
            - iter_idx: (n) tensor of integers, indicating the iteration where a point from 'pts' has been added
            - recency_threshold: integer, indicating how many iterations are considered 'recent' and thus defines which points are modifiable (i.e. non-fixed) 
    """

    n = pts.size(0)
    assert n == iter_idx.size(0)

    max_iter = int(torch.max(iter_idx).item())

    if max_iter < 1:
        return pts, iter_idx

    recency_threshold = min(max_iter, recency_threshold)

    fixed_pts_ids = torch.arange(n)[iter_idx <= (max_iter - recency_threshold)]
    recent_pts_ids = torch.arange(n)[iter_idx > (max_iter - recency_threshold)]

    fixed_pts = pts[fixed_pts_ids]
    recent_pts = pts[recent_pts_ids]

    if recent_pts.size(0) == 0:
        return pts, iter_idx

    # compute internal distances among fixed points
    # min_rad for each point is equal to 80% of the distance to its (current) nearest neighbor (note we are working with squared distances)
    internal_dist = utils.cdist(fixed_pts)                      # size: (N  x N)
    internal_dist += (2 * torch.max(internal_dist)) * torch.eye(internal_dist.size(0), device = internal_dist.device)  # size: (N  x N) [remove zeros from self-distances]
    min_rad = (0.8 ** 2) * internal_dist.min(dim=1)[0]          # size: (N)
    
    # compute distance from recent points to fixed points
    # compute mask of {0,1} where 0 indicates that a 'recent' point should be culled (because it is too close to a fixed point)
    culling_mask = utils.cdist(recent_pts, fixed_pts) < min_rad[None,:]     # size: (m x N)
    culling_mask = ~culling_mask.max(dim=1)[0]                           # size: (m)

    # proximity culling
    non_culled_ids = torch.cat([fixed_pts_ids, recent_pts_ids[culling_mask]])
    pts = pts[non_culled_ids]
    iter_idx = iter_idx[non_culled_ids]

    if pts.size(0) < 1:
        return pts, iter_idx

    # merge clusters for new points
    pts, iter_idx = mergeClusters(pts, iter_idx, torch.sqrt(min_rad).mean().item(), recency_threshold = recency_threshold, verbose=verbose)

    return pts, iter_idx

def reconstruct(data, predictor, detector, corrector = None, max_iters = 10, t = 0, verbose=False, device = torch.device('cpu'), corrector_stage=1, returnTimes=False):

    """
        input:
            - data -> (N x d) tensor of d-dimensional input data to be reconstructed
            - predictor -> a predictor model that predicts a certain number of neighbors for each input data point
            - detector -> a detector model that predicts for each input point if it is on an 'edge'
            - corrector (optional) -> a corrector model that predicts for each input point a displacement vector, to correct its position. Used to correct positions of newly created vertices

        output:
            - running_data -> (N x d) tensor of the final reconstruction state
            - new_points_list -> a python list of (N x d) tensors that each represent the new points added in each iteration
    """

    prediction_time = 0
    detection_time = 0
    corrector_time = 0

    with torch.no_grad():

        start = time.time()

        d = data.size(1)

        iter_idx = torch.zeros(data.size(0))

        # running_data = data.clone().detach()    # size: (N x d)

        for i in range(max_iters):
            
            if verbose:
                print ("\nIteration %d:\n" % (i+1))

            input = data.unsqueeze(0).transpose(1,2)  # size: (1 x d x N)
            
            prediction_start = time.time()
            p = predictor(input).squeeze(0).transpose(0,1).reshape((-1, 6, 3))      # size: (N x k x d)
            if (p.size(0) > 0):
                prediction_time += time.time() - prediction_start

            detection_start = time.time()
            e = detector(input).squeeze(0).transpose(0,1)                           # size: (N x 2)
            if (e.size(0) > 0):
                detection_time = time.time() - detection_start


            del input

            # apply softmax and recall modifier threshold t to edge prediction output and class 'probabilities'
            e = torch.exp(e)
            e /= torch.sum(e, dim=1)[:, None]
            e[:,0] += t
            e = e.argmax(dim=1)     # size: (N x 1)

            # break if no new points are added
            if (torch.sum(e) < 1):
                if verbose:
                    print ("No edge points detected")
                break

            # apply edge mask, add predicted neighbor directions (p) to the given input points (rs) and reshape to get list of points
            newPoints = (data[e == 1, None, :] + p[e == 1]).reshape((-1, d))     # size: (N x d)

            del p, e

            if verbose:
                print ("New candidates: ", newPoints.size(0))

            if (newPoints.size(0) < 1):
                break

            data = torch.cat([data, newPoints])
            iter_idx = torch.cat([iter_idx, (i+1) * torch.ones(newPoints.size(0))])

            data, iter_idx = cleanPoints(data, iter_idx, recency_threshold=1, verbose = verbose)

            corrector_start = time.time()

            if corrector != None:

                pre_correction_sum = -1

                # execute correction steps
                for z in range(10):

                    # compute mask for recent points
                    recent_mask = iter_idx > (i+1 - min(i+1, corrector_stage))

                    error_corrections = corrector(data.unsqueeze(0).transpose(1,2)).transpose(1,2).squeeze(0)[recent_mask] # (N x d)

                    correction_sum = torch.sum(torch.abs(error_corrections)).item()

                    print ("corrections sum: %.30f" % (correction_sum))

                    correction_change = np.abs(correction_sum - pre_correction_sum) / pre_correction_sum
                    if (pre_correction_sum > 0 and correction_change < 0.1):
                        print ("breaking correction loop after %d --> correction change: %.5f" % (z+1, correction_change))
                        break

                    pre_correction_sum = correction_sum
                    
                    # print ("diff: %.30f" % (torch.sum(torch.abs(error_corrections[recent_mask] - data[recent_mask]))))
                    # print ("data checksum %.30f" % (torch.sum(torch.abs(data[recent_mask]))))

                    data[recent_mask] += error_corrections

                    # clean 'recent' points
                    data, iter_idx = cleanPoints(data, iter_idx, recency_threshold=corrector_stage, verbose = verbose)

            if (data.size(0) > 0):
                corrector_time = time.time() - corrector_start

            if verbose:
                print ("New Pts:", data[iter_idx == i+1].size(0))
                # print ("New Size:", running_data.size(0))
                print ("New Size:", data.size(0))

            if (data[iter_idx == i+1].size(0) < 1):
                break

            if i == max_iters - 1 and verbose:
                print ("Terminating reconstruction after %d iterations" % (max_iters))
            
        data = data.to(device)

        if returnTimes:
            return prediction_time, detection_time, corrector_time


        new_points_list = []
        for i in range(1,int(torch.max(iter_idx).item())+1):
            new_points_list.append(data[iter_idx == i].to(device))

        torch.cuda.empty_cache()

        if verbose:
            print ("Reconstruction finished in %.2f s" % (time.time() - start))

        # return running_data, new_points_list
        return data, new_points_list