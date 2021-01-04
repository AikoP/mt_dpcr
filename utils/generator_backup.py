import sys
import os
sys.path.insert(1, os.getcwd())  # to load from any submodule in the repo

from models.models import getModel, loadModel

# from utils import dpcrUtils as utils
import utils.dpcrUtils as utils
import utils.reader as reader
import utils.reconstructor as reconstructor

import numpy as np
import os
import time
import torch
import argparse

# import matplotlib
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

pathDict = {
    'multi_abc': {
        # 'base': 'D:/Data/ABC_reduced',
        'base': 'data/multi_model_training/abc',
        'train': [],
        'val': [],
        # 'train_dir': 'D:/Data/ABC_reduced/training',
        'train_dir': 'data/multi_model_training/abc/training',
        # 'val_dir': 'D:/Data/ABC_reduced/validation',
        'val_dir': 'data/multi_model_training/abc/validation',
    },
    'single_cube': {
        'base': 'data/single_model_training/simple_shapes',
        'train': ['data/single_model_training/simple_shapes/cube.obj'],
        'val': ['data/single_model_training/simple_shapes/cube.obj'],
    },
    'single_ball': {
        'base': 'data/single_model_training/simple_shapes',
        'train': ['data/single_model_training/simple_shapes/ball.obj'],
        'val': ['data/single_model_training/simple_shapes/ball.obj'],
    },
    'single_tetrahedron': {
        'base': 'data/single_model_training/simple_shapes',
        'train': ['data/single_model_training/simple_shapes/tetrahedron.obj'],
        'val': ['data/single_model_training/simple_shapes/tetrahedron.obj'],
    },
    'single_bunny': {
        'base': 'data/single_model_training/bunny',
        'train': ['data/single_model_training/bunny/bunny.obj'],
        'val': ['data/single_model_training/bunny/bunny.obj'],
    },
    'single_armadillo': {
        'base': 'data/single_model_training/armadillo',
        'train': ['data/single_model_training/armadillo/armadillo.obj'],
        'val': ['data/single_model_training/armadillo/armadillo.obj'],
    },
    'multi_faces': {
        'base': 'data/multi_model_training/faces',
        'train': [
            'data/multi_model_training/faces/face_01_clean.obj',
            'data/multi_model_training/faces/face_02_clean.obj',
            'data/multi_model_training/faces/face_03_clean.obj',
            'data/multi_model_training/faces/face_04_clean.obj',
            'data/multi_model_training/faces/face_05_clean.obj',
            'data/multi_model_training/faces/face_06_clean.obj',
            'data/multi_model_training/faces/face_07_clean.obj',
            'data/multi_model_training/faces/face_08_clean.obj',
            'data/multi_model_training/faces/face_09_clean.obj',
            'data/multi_model_training/faces/face_10_clean.obj',
        ],
        'val': [
            'data/multi_model_training/faces/face_11_clean.obj',
            'data/multi_model_training/faces/face_12_clean.obj'
        ],
    },
    'multi_cuboids': {
        'base': 'data/multi_model_training/simple_shapes',
        'train': [
            'data/multi_model_training/simple_shapes/cuboid_01.obj',
            'data/multi_model_training/simple_shapes/cuboid_02.obj',
            'data/multi_model_training/simple_shapes/cuboid_03.obj',
        ],
        'val': [
            'data/multi_model_training/simple_shapes/cuboid_01.obj',
            'data/multi_model_training/simple_shapes/cuboid_02.obj',
            'data/multi_model_training/simple_shapes/cuboid_03.obj',
        ],
    },
    'multi_ellipsoids': {
        'base': 'data/multi_model_training/simple_shapes',
        'train': [
            'data/multi_model_training/simple_shapes/ellipsoid_01.obj',
            'data/multi_model_training/simple_shapes/ellipsoid_02.obj',
            'data/multi_model_training/simple_shapes/ellipsoid_03.obj',
        ],
        'val': [
            'data/multi_model_training/simple_shapes/ellipsoid_01.obj',
            'data/multi_model_training/simple_shapes/ellipsoid_02.obj',
            'data/multi_model_training/simple_shapes/ellipsoid_03.obj',
        ],
    },
    'multi_polyhedrons': {
        'base': 'data/multi_model_training/simple_shapes',
        'train': [
            'data/multi_model_training/simple_shapes/polyhedron_01.obj',
            'data/multi_model_training/simple_shapes/polyhedron_02.obj',
            'data/multi_model_training/simple_shapes/polyhedron_03.obj',
        ],
        'val': [
            'data/multi_model_training/simple_shapes/polyhedron_01.obj',
            'data/multi_model_training/simple_shapes/polyhedron_02.obj',
            'data/multi_model_training/simple_shapes/polyhedron_01.obj',
        ],
    },
    'multi_single_simple_shapes': {
        'base': 'data/single_model_training/simple_shapes',
        'train': [
            'data/single_model_training/simple_shapes/cube.obj',
            'data/single_model_training/simple_shapes/ball.obj',
            'data/single_model_training/simple_shapes/tetrahedron.obj',
        ],
        'val': [
            'data/single_model_training/simple_shapes/cube.obj',
            'data/single_model_training/simple_shapes/ball.obj',
            'data/single_model_training/simple_shapes/tetrahedron.obj',
        ],
    },
    'multi_simple_shapes': {
        'base': 'data/multi_model_training/simple_shapes',
        'train': [
            'data/multi_model_training/simple_shapes/cuboid_01.obj',
            'data/multi_model_training/simple_shapes/cuboid_02.obj',
            'data/multi_model_training/simple_shapes/cuboid_03.obj',
            'data/multi_model_training/simple_shapes/ellipsoid_01.obj',
            'data/multi_model_training/simple_shapes/ellipsoid_02.obj',
            'data/multi_model_training/simple_shapes/ellipsoid_03.obj',
            'data/multi_model_training/simple_shapes/polyhedron_01.obj',
            'data/multi_model_training/simple_shapes/polyhedron_02.obj',
            'data/multi_model_training/simple_shapes/polyhedron_03.obj',
        ],
        'val': [
            'data/multi_model_training/simple_shapes/cuboid_01.obj',
            'data/multi_model_training/simple_shapes/cuboid_02.obj',
            'data/multi_model_training/simple_shapes/cuboid_03.obj',
            'data/multi_model_training/simple_shapes/ellipsoid_01.obj',
            'data/multi_model_training/simple_shapes/ellipsoid_02.obj',
            'data/multi_model_training/simple_shapes/ellipsoid_03.obj',
            'data/multi_model_training/simple_shapes/polyhedron_01.obj',
            'data/multi_model_training/simple_shapes/polyhedron_02.obj',
            'data/multi_model_training/simple_shapes/polyhedron_03.obj',
        ],
    },
    'multi_scans': {
        'base': 'data/multi_model_training/scans',
        'train': [
            'data/multi_model_training/scans/amb_00.obj',
            'data/multi_model_training/scans/amb_01.obj',
            'data/multi_model_training/scans/amb_02.obj',
            'data/multi_model_training/scans/amb_03.obj',
            'data/multi_model_training/scans/amb_04.obj',
            'data/multi_model_training/scans/amb_05.obj',
            'data/multi_model_training/scans/amb_06.obj',
            'data/multi_model_training/scans/amb_07.obj',
            'data/multi_model_training/scans/amb_08.obj',
            'data/multi_model_training/scans/amb_09.obj',
            'data/multi_model_training/scans/amb_10.obj',
            'data/multi_model_training/scans/amb_11.obj',
            'data/multi_model_training/scans/amb_12.obj',
            'data/multi_model_training/scans/amb_13.obj',
            'data/multi_model_training/scans/amb_14.obj',
            'data/multi_model_training/scans/amb_15.obj',
            'data/multi_model_training/scans/amb_16.obj',
            'data/multi_model_training/scans/amb_17.obj',
            'data/multi_model_training/scans/amb_18.obj',
            'data/multi_model_training/scans/amb_19.obj',
            'data/multi_model_training/scans/amb_20.obj',
            'data/multi_model_training/scans/amb_21.obj',
            'data/multi_model_training/scans/amb_22.obj',
            'data/multi_model_training/scans/amb_23.obj',
            'data/multi_model_training/scans/amb_24.obj',
            'data/multi_model_training/scans/amb_25.obj',
            'data/multi_model_training/scans/amb_26.obj',
            'data/multi_model_training/scans/amb_27.obj',
            'data/multi_model_training/scans/amb_28.obj',
            'data/multi_model_training/scans/amb_29.obj',
            'data/multi_model_training/scans/amb_30.obj',
            'data/multi_model_training/scans/amb_31.obj',
            'data/multi_model_training/scans/amb_32.obj',
            'data/multi_model_training/scans/amb_33.obj',
            'data/multi_model_training/scans/amb_34.obj',
            'data/multi_model_training/scans/amb_35.obj',
            'data/multi_model_training/scans/amb_36.obj',
            'data/multi_model_training/scans/amb_37.obj',
            'data/multi_model_training/scans/amb_38.obj',
            'data/multi_model_training/scans/amb_39.obj',
        ],
        'val': [
            'data/multi_model_training/scans/amb_00.obj',
            'data/multi_model_training/scans/amb_01.obj',
            'data/multi_model_training/scans/amb_02.obj',
            'data/multi_model_training/scans/amb_03.obj',
            'data/multi_model_training/scans/amb_04.obj',
            'data/multi_model_training/scans/amb_05.obj',
            'data/multi_model_training/scans/amb_06.obj',
            'data/multi_model_training/scans/amb_07.obj',
            'data/multi_model_training/scans/amb_08.obj',
            'data/multi_model_training/scans/amb_09.obj',
            'data/multi_model_training/scans/amb_10.obj',
            'data/multi_model_training/scans/amb_11.obj',
            'data/multi_model_training/scans/amb_12.obj',
            'data/multi_model_training/scans/amb_13.obj',
            'data/multi_model_training/scans/amb_14.obj',
            'data/multi_model_training/scans/amb_15.obj',
            'data/multi_model_training/scans/amb_16.obj',
            'data/multi_model_training/scans/amb_17.obj',
            'data/multi_model_training/scans/amb_18.obj',
            'data/multi_model_training/scans/amb_19.obj',
            'data/multi_model_training/scans/amb_20.obj',
            'data/multi_model_training/scans/amb_21.obj',
            'data/multi_model_training/scans/amb_22.obj',
            'data/multi_model_training/scans/amb_23.obj',
            'data/multi_model_training/scans/amb_24.obj',
            'data/multi_model_training/scans/amb_25.obj',
            'data/multi_model_training/scans/amb_26.obj',
            'data/multi_model_training/scans/amb_27.obj',
            'data/multi_model_training/scans/amb_28.obj',
            'data/multi_model_training/scans/amb_29.obj',
            'data/multi_model_training/scans/amb_30.obj',
            'data/multi_model_training/scans/amb_31.obj',
            'data/multi_model_training/scans/amb_32.obj',
            'data/multi_model_training/scans/amb_33.obj',
            'data/multi_model_training/scans/amb_34.obj',
            'data/multi_model_training/scans/amb_35.obj',
            'data/multi_model_training/scans/amb_36.obj',
            'data/multi_model_training/scans/amb_37.obj',
            'data/multi_model_training/scans/amb_38.obj',
            'data/multi_model_training/scans/amb_39.obj',
        ],
    }
}

def getPaths():
    return pathDict

def growNeighborhoods(knn, V_0, n=4, noise=0):

    """
    knn:   (N x k) tensor of torch.long that indicate the nearest neighbors for each point p1, ..., pN
    V_0:   torch tensor of 0/1 with size (N,1) or (N) that indicates if a point is visible (=1) or not (=0)
    n:     number of iterations applied
    noise: control the proportion of how many random points of the mask are changed back to visible per iteration (max = 1.0)

    returns:    a list L (of length n+1) of torch.uint8 tensors of size (2, N) where:
                    L[i][0] are the visible indices (indices where the visibility mask in that iteration is 1)
                    L[i][1] is the edge mask E_i relative to the indices (defines which of the visible points given by the indices are edge points)
                    
                for iteration i (i = 0,...,n) respectively

    """

    N = knn.size(0)

    assert(V_0.size(0) == N)

    # compute first edge mask E_0
    # E_0 = V_0.expand(N, -1).gather(1, knn).sum(dim=1) < k
    E_0 = torch.zeros(V_0.size())
    E_0[knn[V_0 == 0].flatten().unique()] = 1
    E_0[V_0 == 0] = 0

    L = [ torch.stack([torch.arange(N)[V_0 == 1], E_0[V_0 == 1].long()], dim=1) ]

    V_i = V_0

    for i in range(n):
        
        # compute next visibility mask V_i
        V_i[knn[V_i == 0].reshape((-1,)).unique()] = 0
        
        # add noise
        if noise > 0:

            idx = torch.arange(N, dtype=torch.long)[V_i == 0]
            nc = int(noise * idx.size(0))
            random_idx = idx[torch.randperm(idx.size(0))[:nc]]

            V_i[random_idx] = 1
            
        # compute edge mask E_i
        # E_i = V_i.expand(N, -1).gather(1, knn).sum(dim=1) < k
        # E_i[V_i == 0] = 0     # <- not needed
        E_i = torch.zeros(V_0.size())
        E_i[knn[V_i == 0].flatten().unique()] = 1
        E_i[V_0 == 0] = 0

        if torch.sum(V_i).item() < 21 or torch.sum(E_i).item() < 1 or torch.sum(E_i == 0).item() < 1:
            print ("prevented additional iteration!")
            break

        L.append(torch.stack([torch.arange(N)[V_i == 1], E_i[V_i == 1].long()], dim=1))

    return L

def getSampleList(knn, size, noise = 0, k = 6, hidden_range = (3, 16), iteration_range = (3,11)):

    N = knn.size(0)

    train_list = []

    for i in range(size):
        
        if hidden_range[0] == hidden_range[1]:
            h = hidden_range[0]
        else:
            h = np.random.randint(hidden_range[0], hidden_range[1])    # number of initially hidden points

        if iteration_range[0] == iteration_range[1]:
            n = iteration_range[0]
        else:
            n = np.random.randint(iteration_range[0], iteration_range[1])    # number of interations
        
        # create initial mask of h hidden points
        V_0 = torch.ones(N, dtype=torch.long)
        V_0[torch.randint(N, (h,), dtype=torch.long)] = 0
        
        train_list += growNeighborhoods(knn, V_0, n = n, noise = noise)

    return train_list

def readModels(model_paths, device = torch.device('cpu')):

    start_id = 0

    models = []
    knns = []

    for (i, model_path) in enumerate(model_paths):

        # if i > 100:
        #     break
        
        print ("Processing model %d of %d.." % (i+1, len(model_paths)))
        
        start = time.time()
        
        # read directly to compute device
        model, _ = reader.readOBJ(model_path, returnType='torch', device=device)

        # mean centering & scaling to fit 2x2x2 bounding box
        model = model - torch.mean(model, dim = 0)
        model = model / torch.max(torch.abs(model))

        knns.append(utils.knn(model, 6).to('cpu') + start_id)
        models.append(model.to('cpu'))
        
        start_id += model.size(0)

        print ("   ..done! (%.1fs)" % (time.time() - start))

    return models, knns

def getSamples(knns, size, hidden_range = (3, 16), iteration_range = (3,11)):

    """
        generates samples for each model

        knns            - list of (N x k) tensors that indicate k nearest neighbor indices
        size            - number of neighborhood growing processes per model
        hidden_range    - valid range of initially hidden points
        iteration_range - valid range of iterations for each neighborhood growing processes

        returns a pair (samples, sample_stats):
            samples - python list of tensors (samples)
            sample_stats - a dict containing sample data split by model (from knn)

    """

    start_id = 0

    sample_stats = {
        'sample_sizes': [],
        'sample_mask_sizes': []
    }

    samples = []

    for knn in knns: 

        for sample in getSampleList(knn, size, hidden_range = hidden_range, iteration_range = iteration_range):
            sample[:, 0] += start_id
            sample_stats['sample_sizes'].append(sample.size(0))
            sample_stats['sample_mask_sizes'].append(torch.sum(sample[:, 1]).item())
            samples.append(sample)
    
        start_id += knn.size(0)

    return samples, sample_stats

def getBins(samples, clearMemory = True):

    train_bins = utils.getBins([sample.size(0) for sample in samples], b=50)
    train_bins_sub = [[] for bin in train_bins]
    bin_stats = []

    print("\nBinning Results:")

    for (i, bin) in enumerate(train_bins):

        size_list = [samples[sample_id].size(0) for sample_id in bin]

        if len(size_list) == 0:
            continue

        min_lim, max_lim = min(size_list), max(size_list)

        bin_stats.append({
            'count': train_bins[i].size(0),
            'min_size': min_lim,
            'max_size': max_lim
        })

        print("Bin %03d: %4d samples (sized %4d to %4d) [%.4f min. sample rate]" % ((i+1), train_bins[i].size(0), min_lim, max_lim, min_lim / max_lim))
        
        train_bins_sub[i] = torch.zeros((bin.size(0), min_lim, 2), dtype=torch.long)
        
        for (j, sid) in enumerate(bin):

            sid = sid.item()

            sample = samples[sid]
            Emask = sample[:, 1]
            
            # compute sample ratio (percentage of how many points from sample are kept)
            subsample_ratio = min_lim / size_list[j]

            # compute number of points in sample that are from either class 0 or class 1
            c1_samples = torch.sum(Emask).item()
            c0_samples = size_list[j] - c1_samples

            # compute exact number of points per class in subsample (note: c1 is preferred, since it usually is the smaller class)
            c1_subsamples = int(np.ceil(c1_samples * subsample_ratio))
            c0_subsamples = min_lim - c1_subsamples

            # get random id's of c1 and c0 subsamples
            idx = torch.arange(0, size_list[j], dtype=torch.long)
            rand_c1_idx = idx[Emask == 1][torch.randperm(c1_samples)[:c1_subsamples]]
            rand_c0_idx = idx[Emask == 0][torch.randperm(c0_samples)[:c0_subsamples]]
            
            # create sub-sample mask
            sub_sampled_idx = torch.zeros(size_list[j], dtype=torch.long)
            sub_sampled_idx[rand_c1_idx] = 1
            sub_sampled_idx[rand_c0_idx] = 1

            train_bins_sub[i][j, :, :] = sample[sub_sampled_idx == 1, :]

            # reference to the old sample can be removed to free up memory
            if clearMemory:
                samples[sid] = None

    return train_bins_sub, bin_stats

def getTrainingSet(
        train_model_paths, validation_model_paths,
        train_size, test_size, validation_size,
        seed = 61846,
        device = torch.device('cpu'),
        hidden_range = (3, 16),
        iteration_range = (3,11)
    ):

    """
        Generates a training set

        train_model_paths        - python list of file paths, that represent the location of .obj files (for train/test data)
        validation_model_paths   - python list of file paths, that represent the location of .obj files (for validation data)
        train_size          - number of train cases to generate per model (each case amounts to roughly 8 samples on average)
        test_size           - number of test cases to generate per model
        validation_size     - number of validation cases to generate per model

    """

    torch.manual_seed(seed)
    np.random.seed(seed)

    #####################################################################################
    # Create Train Data #################################################################

    print ("\nReading train models.. \n")
    train_models, train_knns = readModels(train_model_paths, device=device)

    print ("\nGenerating train samples for %d models.." % (len(train_models)))
    start = time.time()
    train_samples, train_sample_stats = getSamples(train_knns, train_size, hidden_range = hidden_range, iteration_range = iteration_range)
    print ("   ..done! (%.1fs) ---> #samples: %d" % (time.time() - start, len(train_samples)))

    #####################################################################################
    # Binning ###########################################################################
        
    start = time.time()
    train_bins, bin_stats = getBins(train_samples, clearMemory = True)
    print ("\nTotal Binning Time: %.1fs\n" % (time.time() - start))

    #####################################################################################
    # Test Data Generation ##############################################################

    print ("\nGenerating test samples for %d models.." % (len(train_models)))
    start = time.time()
    test_samples, test_sample_stats = getSamples(train_knns, test_size, hidden_range = hidden_range, iteration_range = iteration_range)
    print ("   ..done! (%.1fs) ---> #samples: %d\n" % (time.time() - start, len(test_samples)))

    # package some of the data to tensors..

    train_model_tensor = torch.cat(train_models)
    del train_models

    train_knns_tensor = torch.cat(train_knns)
    del train_knns

    #####################################################################################
    # Validation Data Generation ########################################################

    print ("\nReading validation models.. \n")
    val_models, val_knns = readModels(validation_model_paths, device=device)

    print ("\nGenerating train samples for %d models.." % (len(val_models)))
    start = time.time()
    val_samples, val_sample_stats = getSamples(val_knns, validation_size, hidden_range = hidden_range, iteration_range = iteration_range)
    print ("   ..done! (%.1fs) ---> #samples: %d" % (time.time() - start, len(val_samples)))

    # package some of the data to tensors..

    val_models_tensor = torch.cat(val_models)
    del val_models

    val_knns_tensor = torch.cat(val_knns)
    del val_knns

    return {
        "pts": train_model_tensor,
        "knn": train_knns_tensor,
        "train_bins": train_bins,
        "train_bin_stats": bin_stats,
        "test_samples": test_samples,
        "val_pts": val_models_tensor,
        "val_knn": val_knns_tensor,
        "val_samples": val_samples,
        'train_sample_stats': train_sample_stats,
        'test_sample_stats': test_sample_stats,
        'val_sample_stats': val_sample_stats
    }

def getSyntheticDeformTrainData(args):

    train_models, validation_models = pathDict[args.dataset]['train'], pathDict[args.dataset]['val']

    os.makedirs(pathDict[args.dataset]['base'], exist_ok=True)

    hidden_range = (args.h_min, args.h_max)
    iteration_range = (args.n_min, args.n_max)

    device = utils.getDevice()

    train_data = []

    for (i, model_path) in enumerate(train_models):
        
        print ("Processing model %d of %d.." % (i+1, len(train_models)))
        
        start = time.time()
        
        model, _ = reader.readOBJ(model_path)
        model = torch.from_numpy(model).float().to(device)
        
        # mean centering & scaling to fit 2x2x2 bounding box
        model = model - torch.mean(model, dim = 0)
        model = model / torch.max(torch.abs(model))

        knn = utils.knn(model, k=6)
        
        for j in range(args.train_size):

            # get list of hole growing steps (first element is the complete tensor)
            samples = getSampleList(model, 1, hidden_range = hidden_range, iteration_range = iteration_range)

            for sample in samples:

            # sample = samples[-1]

                edge_ids = sample[sample[:,1] == 1, 0]

                defs_anchor_ids = edge_ids[torch.randperm(edge_ids.size(0))[:int(edge_ids.size(0) / 10)]]

                def_sample = model.clone()

                mask = torch.zeros(model.size(0), device = device, dtype=torch.bool)
                mask[defs_anchor_ids] = True

                m = 0.05
                dir = np.random.choice([-1,1])

                for k in range(10):

                    norm_fac = torch.sqrt(torch.square(def_sample[mask]).sum(dim=1)) # (n)
                    def_sample[mask] = def_sample[mask] + dir * ((m - m * k/5) / norm_fac)[:,None] * def_sample[mask]

                    next_ids = knn[mask].flatten()

                    mask[mask] = False
                    mask[next_ids] = True

                # the complete sample contains inputs and corresponding targets 
                train_sample = torch.cat([def_sample[sample[:,0]][:,:,None], model[sample[:,0]][:,:,None]], dim=2).cpu()  # (N x d x 2)

                train_data.append(train_sample)

                del def_sample

                if len(train_data) % 100 == 0:
                    print (" > generated %3d samples (%.1fs).." % (len(train_data), time.time() - start))

        del model
        
        print ("   ..done! (%.1fs)" % (time.time() - start))
        
    print ("\n#Train Samples: %d" % len(train_data))

    data = {
        "train_bins": None,
        "test_samples": None,
        "val_samples": None,
        'sample_stats': {
            'sample_sizes': [sample.size(0) for sample in train_data]
        }
    }

    #####################################################################################
    # Binning ###########################################################################
        
    start = time.time()

    train_bins = utils.getBins([sample.size(0) for sample in train_data], b=50)
    train_bins_sub = [[] for bin in train_bins]
    bin_stats = []

    print("\nBinning Results:")

    for (i, bin) in enumerate(train_bins):

        size_list = [train_data[sample_id].size(0) for sample_id in bin]

        if len(size_list) == 0:
            continue

        min_lim, max_lim = min(size_list), max(size_list)

        bin_stats.append({
            'count': bin.size(0),
            'min_size': min_lim,
            'max_size': max_lim
        })

        print("Bin %03d: %4d samples (sized %4d to %4d) [%.4f min. sample rate]" % ((i+1), bin.size(0), min_lim, max_lim, min_lim / max_lim))
        
        train_bins_sub[i] = torch.zeros((bin.size(0), min_lim, train_data[0].size(1), 2), dtype=torch.float)
        
        for (j, sample_id) in enumerate(bin):

            sample = train_data[sample_id.item()]

            # select a min_lim random points
            idx = torch.randperm(sample.size(0))[:min_lim]  
            
            train_bins_sub[i][j, :, :, :] = sample[idx]
            
    data["train_bins"] = train_bins_sub
    data["train_bin_stats"] = bin_stats
            
    print ("\nTotal Binning Time: %.1fs\n" % (time.time() - start))

    #####################################################################################
    # Test Data Generation ##############################################################

    print ("Generating TEST Data -------------------\n")

    test_data = []

    for (i, model_path) in enumerate(train_models):
        
        print ("Processing model %d of %d.." % (i+1, len(train_models)))
        
        start = time.time()
        
        model, _ = reader.readOBJ(model_path)
        model = torch.from_numpy(model).float().to(device)
        
        # mean centering & scaling to fit 2x2x2 bounding box
        model = model - torch.mean(model, dim = 0)
        model = model / torch.max(torch.abs(model))

        knn = utils.knn(model, k=6)
        
        for j in range(args.test_size):

            # get list of hole growing steps (first element is the complete tensor)
            samples = getSampleList(model, 1, hidden_range = hidden_range, iteration_range = iteration_range)

            for sample in samples:

            # sample = samples[-1]

                edge_ids = sample[sample[:,1] == 1, 0]

                defs_anchor_ids = edge_ids[torch.randperm(edge_ids.size(0))[:int(edge_ids.size(0) / 10)]]

                def_sample = model.clone()

                mask = torch.zeros(model.size(0), device = device, dtype=torch.bool)
                mask[defs_anchor_ids] = True

                m = 0.05
                dir = np.random.choice([-1,1])

                for k in range(10):

                    norm_fac = torch.sqrt(torch.square(def_sample[mask]).sum(dim=1)) # (n)
                    def_sample[mask] = def_sample[mask] + dir * ((m - m * k/5) / norm_fac)[:,None] * def_sample[mask]

                    next_ids = knn[mask].flatten()

                    mask[mask] = False
                    mask[next_ids] = True

                # the complete sample contains inputs and corresponding targets 
                test_sample = torch.cat([def_sample[sample[:,0]][:,:,None], model[sample[:,0]][:,:,None]], dim=2).cpu()  # (N x d x 2)
                
                test_data.append(test_sample)

                del def_sample

                if len(test_data) % 100 == 0:
                    print (" > generated %3d samples (%.1fs).." % (len(test_data), time.time() - start))

        del model
        
        print ("   ..done! (%.4fs)" % (time.time() - start))
        
    data["test_samples"] = test_data
        
    print ("\n#Test Samples: %d\n" % len(test_data))

    #####################################################################################
    # Validation Data Generation ########################################################

    print ("Generating VALIDATION Data -------------\n")
        
    val_data = []

    for (i, model_path) in enumerate(validation_models):
        
        print ("Processing model %d of %d.." % (i+1, len(train_models)))
        
        start = time.time()
        
        model, _ = reader.readOBJ(model_path)
        model = torch.from_numpy(model).float().to(device)
        
        # mean centering & scaling to fit 2x2x2 bounding box
        model = model - torch.mean(model, dim = 0)
        model = model / torch.max(torch.abs(model))

        knn = utils.knn(model, k=6)
        
        for j in range(args.val_size):

            # get list of hole growing steps (first element is the complete tensor)
            samples = getSampleList(model, 1, hidden_range = hidden_range, iteration_range = iteration_range)

            for sample in samples:

            # sample = samples[-1]

                edge_ids = sample[sample[:,1] == 1, 0]

                defs_anchor_ids = edge_ids[torch.randperm(edge_ids.size(0))[:int(edge_ids.size(0) / 10)]]

                def_sample = model.clone()

                mask = torch.zeros(model.size(0), device = device, dtype=torch.bool)
                mask[defs_anchor_ids] = True

                m = 0.05
                dir = np.random.choice([-1,1])

                for k in range(10):

                    norm_fac = torch.sqrt(torch.square(def_sample[mask]).sum(dim=1)) # (n)
                    def_sample[mask] = def_sample[mask] + dir * ((m - m * k/5) / norm_fac)[:,None] * def_sample[mask]

                    next_ids = knn[mask].flatten()

                    mask[mask] = False
                    mask[next_ids] = True

                # the complete sample contains inputs and corresponding targets 
                val_sample = torch.cat([def_sample[sample[:,0]][:,:,None], model[sample[:,0]][:,:,None]], dim=2).cpu()  # (N x d x 2)
                
                val_data.append(val_sample)

                del def_sample

                if len(val_data) % 100 == 0:
                    print (" > generated %3d samples (%.1fs).." % (len(val_data), time.time() - start))

        del model
        
        print ("   ..done! (%.4fs)" % (time.time() - start))
        
    data["val_samples"] = val_data

    print ("\n#Validation Samples: %d" % len(val_data))

    return data

def getCorrectorTrainData(args):

    assert args.predictor_checkpoint != None
    assert args.detector_checkpoint != None

    train_model_paths, validation_model_paths = pathDict[args.dataset]['train'], pathDict[args.dataset]['val']

    device = utils.getDevice()

    assert args.n_min >= 3

    os.makedirs(pathDict[args.dataset]['base'], exist_ok=True)

    hidden_range = (args.h_min, args.h_max)
    iteration_range = (args.n_min, args.n_max)

    predictor = loadModel(args.predictor_checkpoint, device = device)
    detector = loadModel(args.detector_checkpoint, device = device)

    corrector = None if args.corrector_checkpoint == None else loadModel(args.corrector_checkpoint, device = device)

    #####################################################################################
    # Create Train Data #################################################################

    print ("\nGenerating TRAIN Data ------------------\n")

    print ("\nReading train models.. \n")
    train_models, train_knns = readModels(train_model_paths, device=device)

    print ("\nGenerating train samples for %d models.." % (len(train_models)))

    train_data = []

    for (i, (model, model_knn)) in enumerate(zip(train_models, train_knns)):
        
        print ("Processing model %d of %d.." % (i+1, len(train_models)))
        
        start = time.time()
        
        for j in range(args.train_size):

            # get list of hole growing steps (first element is the complete tensor)
            samples = getSampleList(model_knn, 1, hidden_range = hidden_range, iteration_range = iteration_range)

            # set the input (the most 'eroded' step)
            input = model[samples[-1][:,0]]

            # # choose a random point to start from
            # d = args.max_pipeline_iterations
            # d = np.random.randint(args.max_pipeline_iterations, max(args.max_pipeline_iterations, len(samples)))
            # # d = len(samples)

            # base = model[samples[d][:,0]]

            max_iters = min(args.max_pipeline_iterations, len(samples))

            _, new_points_list = reconstructor.reconstruct(input, predictor, detector, corrector = corrector, max_iters = max_iters, t = 0, device = device)

            sample_input = input.clone()
            sample_target = torch.zeros(sample_input.size(), device=device)

            for (m, new_points) in enumerate(new_points_list):

                # calculate for each predicted point the closest point from the complete model
                Dt_min = torch.min(utils.cdist(new_points, model), dim = 1)[1]

                # compute targets as directions from each predicted point to the closest model point
                sample_input = torch.cat([sample_input, new_points], 0)
                sample_target = torch.cat([sample_target, model[Dt_min] - new_points])

                # the complete sample contains inputs and corresponding targets 
                sample = torch.cat([sample_input[:,:,None], sample_target[:,:,None]], dim=2)     # (N, d, 2)

                train_data.append(sample.detach().cpu().clone())

            # start_idx = base.size(0)
            # for (m, p) in enumerate(new_points_list):

            #     # select only points that were on the edge previously
            #     t = model[samples[d-m-1][samples[d-m-1][:,1] == 1, 0]]

            #     # calculate for each predicted point the closest 'true' edge point
            #     Dt_min = torch.min(utils.cdist(p, t), dim = 1)[1]

            #     # compute targets as directions from each predicted point to the closest 'true' edge point
            #     sample_target[start_idx:start_idx+p.size(0)] = t[Dt_min] - p

            #     start_idx += p.size(0)

            # the complete sample contains inputs and corresponding targets 
            # sample = torch.cat([sample_input[:,:,None], sample_target[:,:,None]], dim=2)     # (N, d, 2)

            # train_data.append(sample.detach().cpu())

            if len(train_data) % 100 == 0:
                print (" > generated %3d samples (%.1fs).." % (len(train_data), time.time() - start))
        
        print ("   ..done! (%.1fs)" % (time.time() - start))
        
    print ("\n#Train Samples: %d" % len(train_data))

    data = {
        "train_bins": None,
        "test_samples": None,
        "val_samples": None,
        'sample_stats': {
            'sample_sizes': [sample.size(0) for sample in train_data]
        }
    }

    #####################################################################################
    # Binning ###########################################################################
        
    start = time.time()

    train_bins = [bin.to(device) for bin in utils.getBins([sample.size(0) for sample in train_data], b=50)]
    train_bins_sub = [[] for bin in train_bins]
    bin_stats = []

    print("\nBinning Results:")

    for (i, bin) in enumerate(train_bins):

        size_list = [train_data[sample_id].size(0) for sample_id in bin]

        if len(size_list) == 0:
            continue

        min_lim, max_lim = min(size_list), max(size_list)

        bin_stats.append({
            'count': bin.size(0),
            'min_size': min_lim,
            'max_size': max_lim
        })

        print("Bin %03d: %4d samples (sized %4d to %4d) [%.4f min. sample rate]" % ((i+1), bin.size(0), min_lim, max_lim, min_lim / max_lim))
        
        train_bins_sub[i] = torch.zeros((bin.size(0), min_lim, train_data[0].size(1), 2), dtype=torch.float, device=device)
        
        for (j, sample_id) in enumerate(bin):

            sample = train_data[sample_id.item()]

            # select a min_lim random points
            idx = torch.randperm(sample.size(0))[:min_lim]  
            
            train_bins_sub[i][j, :, :, :] = sample[idx]
            
    data["train_bins"] = train_bins_sub
    data["train_bin_stats"] = bin_stats
            
    print ("\nTotal Binning Time: %.1fs\n" % (time.time() - start))

    #####################################################################################
    # Test Data Generation ##############################################################

    print ("Generating TEST Data -------------------\n")

    test_data = []

    for (i, model_path) in enumerate(train_models):
        
        print ("Processing model %d of %d.." % (i+1, len(train_models)))
        
        start = time.time()
        
        model, _ = reader.readOBJ(model_path)
        model = torch.from_numpy(model).float().to(device)
        
        # mean centering & scaling to fit 2x2x2 bounding box
        model = model - torch.mean(model, dim = 0)
        model = model / torch.max(torch.abs(model))
        
        for j in range(args.test_size):

            # get list of hole growing steps (first element is the complete tensor)
            samples = getSampleList(model, 1, hidden_range = hidden_range, iteration_range = iteration_range)

            # choose a random point to start from
            d = np.random.randint(args.max_pipeline_iterations, len(samples))

            base = model[samples[d][:,0]]

            _, new_points_list = reconstructor.reconstruct(base, predictor, detector, max_iters = args.max_pipeline_iterations, t = 0)

            new_points_list = [newPts.to(device) for newPts in new_points_list]

            sample_input = torch.cat([base] + new_points_list, 0)
            sample_target = torch.zeros(sample_input.size(), device=device)
        
            start_idx = base.size(0)
            for (m, p) in enumerate(new_points_list):

                # select only points that were on the edge previously
                t = model[samples[d-m-1][samples[d-m-1][:,1] == 1, 0]]

                # calculate for each predicted point the closest 'true' edge point
                Dt_min = torch.min(utils.cdist(p, t), dim = 1)[1]

                # compute targets as directions from each predicted point to the closest 'true' edge point
                sample_target[start_idx:start_idx+p.size(0)] = t[Dt_min] - p

                start_idx += p.size(0)

            # the complete sample contains inputs and corresponding targets 
            sample = torch.cat([sample_input[:,:,None], sample_target[:,:,None]], dim=2)     # (N, d, 2)

            test_data.append(sample.detach().cpu())

            if len(test_data) % 100 == 0:
                print (" > generated %3d samples (%.1fs).." % (len(test_data), time.time() - start))
        
        print ("   ..done! (%.4fs)" % (time.time() - start))
        
    data["test_samples"] = test_data
        
    print ("\n#Test Samples: %d\n" % len(test_data))

    #####################################################################################
    # Validation Data Generation ########################################################

    print ("Generating VALIDATION Data -------------\n")
        
    val_data = []

    for (i, model_path) in enumerate(validation_models):
        
        print ("Processing model %d of %d.." % (i+1, len(validation_models)))
        
        start = time.time()
        
        model, _ = reader.readOBJ(model_path)
        model = torch.from_numpy(model).float().to(device)
        
        # mean centering & scaling to fit 2x2x2 bounding box
        model = model - torch.mean(model, dim = 0)
        model = model / torch.max(torch.abs(model))
        
        for j in range(args.val_size):

            # get list of hole growing steps (first element is the complete tensor)
            samples = getSampleList(model, 1, hidden_range = hidden_range, iteration_range = iteration_range)

            # choose a random point to start from
            d = np.random.randint(args.max_pipeline_iterations, len(samples))

            base = model[samples[d][:,0]]

            _, new_points_list = reconstructor.reconstruct(base, predictor, detector, max_iters = args.max_pipeline_iterations, t = 0)

            new_points_list = [newPts.to(device) for newPts in new_points_list]

            sample_input = torch.cat([base] + new_points_list, 0)
            sample_target = torch.zeros(sample_input.size(), device=device)
        
            start_idx = base.size(0)
            for (m, p) in enumerate(new_points_list):

                # select only points that were on the edge previously
                t = model[samples[d-m-1][samples[d-m-1][:,1] == 1, 0]]

                # calculate for each predicted point the closest 'true' edge point
                Dt_min = torch.min(utils.cdist(p, t), dim = 1)[1]

                # compute targets as directions from each predicted point to the closest 'true' edge point
                sample_target[start_idx:start_idx+p.size(0)] = t[Dt_min] - p

                start_idx += p.size(0)

            # the complete sample contains inputs and corresponding targets 
            sample = torch.cat([sample_input[:,:,None], sample_target[:,:,None]], dim=2)     # (N, d, 2)

            val_data.append(sample.detach().cpu())

            if len(val_data) % 100 == 0:
                print (" > generated %3d samples (%.1fs).." % (len(val_data), time.time() - start))
        
        print ("   ..done! (%.4fs)" % (time.time() - start))
        
    data["val_samples"] = val_data

    print ("\n#Validation Samples: %d" % len(val_data))

    return data


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Point Cloud Part Segmentation')

    seed_file = open('utils/seed.txt', "r")
    seed = int(seed_file.read())
    seed_file.close()

    print ("Using Seed:", seed)

    torch.manual_seed(seed)
    np.random.seed(seed)

    parser.add_argument('--type', type=str, default='normal', metavar='TYPE', choices=['normal','corrector'], help='which type of training data to generate: [normal, corrector]')
    parser.add_argument('--train_size', required=True, type=int, default=1, metavar='TRN', help='number of train samples to generate')
    parser.add_argument('--test_size', required=True, type=int, default=1, metavar='TST', help='number of test samples to generate')
    parser.add_argument('--val_size', required=True, type=int, default=1, metavar='VAL', help='number of validation samples to generate')
    parser.add_argument('--dataset', required=True, type=str, default='multi_faces', metavar='D', choices=list(pathDict.keys()), help='Training dataset to use')
    parser.add_argument('--h_min', type=int, default=3, metavar='hmin', help='minimum number of initially hidden points')
    parser.add_argument('--h_max', type=int, default=16, metavar='hmax', help='maximum number of initially hidden points')
    parser.add_argument('--n_min', type=int, default=3, metavar='hmin', help='minimum number of iterations per sample')
    parser.add_argument('--n_max', type=int, default=11, metavar='hmax', help='maximum number of iterations per sample')
    parser.add_argument('--predictor_checkpoint', type=str, metavar='p_cp', help='path to a trained predictor model checkpoint')
    parser.add_argument('--detector_checkpoint', type=str, metavar='d_cp', help='path to a trained detector model checkpoint')
    parser.add_argument('--corrector_checkpoint', type=str, metavar='c_cp', help='path to a trained corrector model checkpoint')
    parser.add_argument('--max_pipeline_iterations', type=int, default=1, metavar='maxP', help='maximum number of times the pipeline is applied to generate corrector train data')

    args = parser.parse_args()

    if args.type == 'corrector':

        train_data = getCorrectorTrainData(args)

        with open(pathDict[args.dataset]['base'] + '/' + args.dataset + ('_train_data_corrector_%02d' % (args.max_pipeline_iterations)), 'wb') as file:
            torch.save(train_data, file)

    elif args.type == 'corrector_synthetic':

        train_data = getSyntheticDeformTrainData(args)

        with open(pathDict[args.dataset]['base'] + '/' + args.dataset + ('_train_data_corrector_%02d' % (args.max_pipeline_iterations)), 'wb') as file:
            torch.save(train_data, file)

    else:

        hidden_range = (args.h_min, args.h_max)
        iteration_range = (args.n_min, args.n_max)

        train_models, validation_models = pathDict[args.dataset]['train'], pathDict[args.dataset]['val']

        if 'train_dir' in  pathDict[args.dataset]:
            train_dir = pathDict[args.dataset]['train_dir']
            for file in os.listdir(train_dir):
                if file.endswith(".obj"):
                    train_models.append(os.path.join(train_dir, file))

        if 'val_dir' in pathDict[args.dataset]:
            val_dir = pathDict[args.dataset]['val_dir']
            for file in os.listdir(val_dir):
                if file.endswith(".obj"):
                    validation_models.append(os.path.join(val_dir, file))

        print ("# train models: ", len(train_models))
        print ("# val models: ", len(validation_models))

        device = torch.device("cpu")

        if torch.device("cuda"):
            device = torch.device("cuda")
            print("Using", torch.cuda.device_count(), "CUDA devices")

        with torch.no_grad():
            
            train_data = getTrainingSet(
                train_models,
                validation_models,
                args.train_size,
                args.test_size,
                args.val_size,
                seed = seed,
                device = device,
                hidden_range = hidden_range,
                iteration_range = iteration_range)

        os.makedirs(pathDict[args.dataset]['base'], exist_ok=True)

        with open(pathDict[args.dataset]['base'] + '/' + args.dataset + '_train_data', 'wb') as file:
            torch.save(train_data, file)