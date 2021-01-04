import sys
import os

sys.path.insert(1, os.getcwd())  # to load from any submodule in the repo

import torch
import numpy as np
import time
from utils import reader
import utils.dpcrUtils as utils
from utils import generator
from utils import writer
import argparse

a = 10 * [torch.tensor([1,2,3])]#

print (a)

a = [t + sum([s.size(0) for s in a[:i]]) for (i, t) in enumerate(a)]

print (a)

# def cdist(a, b = None):

#     """
#         a: (C1 x ... x Cn x N x d) tensor
#         b: (C1 x ... x Cn x M x d) tensor [optional]
#     """

#     aa = torch.sum(torch.square(a), dim=-1, keepdim=True)   # size: (C1 x ... x Cn x N x 1)

#     if b != None:
#         bb = torch.sum(torch.square(b), dim=-1, keepdim=True)                               # size: (C1 x ... x Cn x M x 1)
#         inner = torch.matmul(-2.0 * a, b.transpose(-1, -2))                                  # size: (C1 x ... x Cn x N x M)
#         # return (aa - inner + bb.transpose(-1, -2)).clamp(min=torch.finfo(a.dtype).eps)      # size: (C1 x ... x Cn x N x M)
#         inner.add_(aa)
#         inner.add_(bb.transpose(-1, -2))
#         return inner.clamp(min=torch.finfo(a.dtype).eps)      # size: (C1 x ... x Cn x N x M)

#     else:
#         inner = torch.matmul(-2.0 * a, a.transpose(-1, -2))                                  # size: (C1 x ... x Cn x N x N)
#         # return (aa - inner + aa.transpose(-1, -2)).clamp(min=torch.finfo(a.dtype).eps)      # size: (C1 x ... x Cn x N x N)
#         inner.add_(aa)
#         inner.add_(aa.transpose(-1, -2))
#         return inner.clamp(min=torch.finfo(a.dtype).eps)      # size: (C1 x ... x Cn x N x M)

# a = torch.randn(100,3)
# b = torch.randn(100,3)

# print(torch.sum(cdist(a)-utils.cdist(a)))
# print(torch.sum(cdist(a,b)-utils.cdist(a,b)))

# N = 1000

# start = time.time()
# for i in range(N):
#     d = cdist(a,b)
# end = time.time()
# print ("Average time: %.3fms" % ((end-start) * 1000/N))

# start = time.time()
# for i in range(N):
#     d = utils.cdist(a,b)
# end = time.time()
# print ("Average time: %.3fms" % ((end-start) * 1000/N))

# start = time.time()
# for i in range(N):
#     d = cdist(a,b)
# end = time.time()
# print ("Average time: %.3fms" % ((end-start) * 1000/N))

# start = time.time()
# for i in range(N):
#     d = utils.cdist(a,b)
# end = time.time()
# print ("Average time: %.3fms" % ((end-start) * 1000/N))

# device = utils.getDevice()

# noise = torch.randn((10000, 3), device=device)
# noise_unit = noise / torch.square(noise).sum(dim=1)[:,None].sqrt()

# N = 1000

# start = time.time()
# for i in range(N):
#     noise = torch.randn((10000, 3), device=device)
#     noise_unit = noise / torch.square(noise).sum(dim=1).sqrt()[:,None]
# end = time.time()
# print ("Average time: %.3fms" % ((end-start) * 1000/N))

# noise_unit = noise_unit.transpose(0,1)

# start = time.time()
# for i in range(N):
#     rot = utils.getRandomRotation(3, device=device)
#     noise = rot.matmul(noise_unit)
# end = time.time()
# print ("Average time: %.3fms" % ((end-start) * 1000/N))


# def knn(x, k, q=None, device = torch.device('cpu')):

#     """
#         x: (b,n,d) tensor 
#     """

#     n = x.size(1)

#     # t = torch.cuda.get_device_properties(0).total_memory
#     # c = torch.cuda.memory_cached(0)
#     # a = torch.cuda.memory_allocated(0)
#     # f = c-a  # free inside cache

#     # print ("total:", t)
#     # print ("cached:", c)
#     # print ("allocated:", a)
#     # print ("free:", f)

#     if (q == None or n <= q):
#         return (utils.cdist(x.to(device)).topk(k=k+1, dim=-1, largest=False)[1][:,:,1:]).to(x.device)

#     else:

#         topk = torch.zeros((x.size(0), n, k), dtype=torch.long, device=x.device)

#         x_device = x.to(device)

#         for i in range(0, n, q):
#             topk[:, i:i+q, :] = utils.cdist(x_device[:,i:i+q], x_device).topk(k=k+1, dim=-1, largest=False)[1][:,:,1:].to(x.device)

#         return topk

# N = 100

# x = torch.randn((2,20000, 3))
# device = utils.getDevice()

# knns = torch.zeros(N, 6, dtype=torch.long, device=x.device)

# start = time.time()
# for i in range(N):
#     knns = knn(x, 6, device=device)
# end = time.time()
# print ("Average time: %.3fms" % ((end-start) * 1000/N))

# for q in range(0, 16000, 3000):
#     start = time.time()
#     for i in range(N):
#         knns = knn(x, 6, q=1000, device=device)
#     end = time.time()
#     print ("Average time for q=%d: %.3fms" % (q, (end-start) * 1000/N))

# GPU_MAX_MEM = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()

# print ("GPU_MAX_MEM =", GPU_MAX_MEM)

# start = time.time()
# b = x.size(0)
# n = x.size(1)
# d = x.size(2)
# # q = int(np.sqrt(GPU_MAX_MEM / (36 * b)))
# # q = int((1/6) * (np.sqrt(1 + GPU_MAX_MEM / b) - 1))
# # q = int(-0.25 + (np.sqrt(1/16 - n/2 + GPU_MAX_MEM / (2*4*b*d))))
# q = int(((0.7 * GPU_MAX_MEM) / (4*b) - n*d)/(n*d + d))

# print ("using q =", q)
# for i in range(N):
#     knns = knn(x, 6, q=q, device=device)
# end = time.time()
# print ("Average time for q=%d: %.3fms" % (q, (end-start) * 1000/N))


# sizes = [8000 + np.random.randint(5000) for i in range(6500)]

# start = time.time()

# train_data = []
# for s in sizes:
#     for i in range(1,7):
#         ssize = int(15*(3**i) * (0.6 + 0.4 * np.random.rand()))
#         ssize = max(1, s-ssize)
#         edge_mask = torch.randint(1, (ssize,), dtype = torch.long)
#         v_idx = torch.randint(s, (ssize,), dtype = torch.long)
#         train_data.append(torch.cat([v_idx[:,None], edge_mask[:,None]], dim=1))

# print ("# train samples:", len(train_data))

# train_bins = utils.getBins([sample.size(0) for sample in train_data], b=50)
# train_bins_sub = [[] for bin in train_bins]
# bin_stats = []

# print("\nBinning Results:")

# for (i, bin) in enumerate(train_bins):

#     size_list = [train_data[sample_id].size(0) for sample_id in bin]

#     if len(size_list) == 0:
#         continue

#     min_lim, max_lim = min(size_list), max(size_list)

#     bin_stats.append({
#         'count': train_bins[i].size(0),
#         'min_size': min_lim,
#         'max_size': max_lim
#     })

#     print("Bin %03d: %4d samples (sized %4d to %4d) [%.4f min. sample rate]" % ((i+1), train_bins[i].size(0), min_lim, max_lim, min_lim / max_lim))
    
#     train_bins_sub[i] = torch.zeros((bin.size(0), min_lim, 2), dtype=torch.long)
    
#     for (j, sid) in enumerate(bin):

#         sid = sid.item()

#         sample = train_data[sid]
        
#         Vidx = sample[:, 0]
#         Emask = sample[:, 1]
        
#         # compute sample ratio (percentage of how many points from sample are kept)
#         subsample_ratio = min_lim / size_list[j]

#         # compute number of points in sample that are from either class 0 or class 1
#         c1_samples = torch.sum(Emask).item()
#         c0_samples = size_list[j] - c1_samples

#         # compute exact number of points per class in subsample (note: c1 is preferred, since it usually is the smaller class)
#         c1_subsamples = int(np.ceil(c1_samples * subsample_ratio))
#         c0_subsamples = min_lim - c1_subsamples

#         # get random id's of c1 and c0 subsamples
#         idx = torch.arange(0, size_list[j], dtype=torch.long)
#         rand_c1_idx = idx[Emask == 1][torch.randperm(c1_samples)[:c1_subsamples]]
#         rand_c0_idx = idx[Emask == 0][torch.randperm(c0_samples)[:c0_subsamples]]
        
#         # create sub-sample mask
#         sub_sampled_idx = torch.zeros(size_list[j], dtype=torch.long)
#         sub_sampled_idx[rand_c1_idx] = 1
#         sub_sampled_idx[rand_c0_idx] = 1

#         train_bins_sub[i][j, :, :] = sample[sub_sampled_idx == 1, :]

#         # reference to the old sample can be removed to free up memory
#         train_data[sid] = None

# print ("\nTotal Binning Time: %.1fs\n" % (time.time() - start))




# sizes = []
# start = time.time()
# # for (i, model_path) in enumerate(model_paths):
# #     sizes.push(reader.readOBJsize(model_path))

# print ("reading took %.2fs .." % (time.time() - start))
# print ("total vertices:", sum(sizes))

# # knn_tensor = torch.empty((sum(sizes), 6), dtype=torch.long)

# knns = []
# start_idx = 0
# for s in sizes:
#     knns.append(torch.randint(s, (s, 6), dtype = torch.int64))
    # knn_tensor[start_idx:start_idx+sizes[i]] = torch.randint(13000, (sizes[i], 6))

# print (knns[0].dtype)

# cum_lengths = [0]
# for knn in knns:
#     cum_lengths.append(cum_lengths[-1] + knn.size(0))

# knn_tensor = torch.zeros((cum_lengths[-1], 6))
# knn_tensor = torch.cat(knns)


# cum_lengths = [0]
# for knn in knns:
#     cum_lengths.append(cum_lengths[-1] + knn.size(0))

# knn_tensor = torch.empty((sum(sizes), 6), dtype=torch.long)
# start_idx = 0
# for (s, knn) in zip(sizes, knns):
#     knn_tensor[start_idx:start_idx+s] = knn
#     start_idx += s


# knn_tensor = torch.cat(knns[:100])
# knns = knns[100:]
# while len(knns) > 0:
#     knn_tensor = torch.cat([knn_tensor] + knns[:100])
#     knns = knns[100:]

# device = utils.getDevice()

# pts = []

# n = 10
# base_angles = 2 * np.pi * np.arange(n) / n

# print ("base_angles:", base_angles)


# for i in range(2):
    
#     model, _ = reader.readOBJ('D:/Github/Repos/dpcr/data/multi_model_training/scans/amb_%02d.obj' % (i))
#     model = torch.from_numpy(model).float().to(device)
#     model = model - torch.mean(model, dim = 0)
#     # model = model / torch.max(torch.abs(model))
#     pts.append(model.to(device))

#     if (i+1)%5==0:
#         print ('Loaded %02d models..' % (i))

# model1 = pts[0]
# model2 = pts[1]

# diffs = []
# angles = []

# min_angles = []
# min_diff = -1

# for i in range(n):
#     for j in range(n):
#         for k in range(n):

#             rot = utils.getRotation(3, base_angles[[i,j,k]], device=device, dtype=torch.float)

#             model2_rot  = rot.matmul(model2.transpose(0,1)).transpose(0,1)

#             d_sum = torch.sum(torch.sqrt(utils.cdist(model1, model2_rot).double()))

#             diffs.append(d_sum)
#             angles.append(base_angles[[i,j,k]])

#             if min_diff < 0 or d_sum < min_diff:
#                 min_diff = d_sum
#                 min_angles = base_angles[[i,j,k]]

# print (min_angles)
# print ("min_diff:", min_diff)
# # print (diffs)

# rot = utils.getRotation(3, min_angles, device=device, dtype=torch.float)
# model2_rot  = rot.matmul(model2.transpose(0,1)).transpose(0,1)

# rot2 = utils.getRotation(3, [min_angles[0] + np.pi, min_angles[1] + np.pi, min_angles[2] + np.pi] , device=device, dtype=torch.float)
# model2_rot2  = rot2.matmul(model2.transpose(0,1)).transpose(0,1)
# d2_sum = torch.sum(torch.sqrt(utils.cdist(model1, model2_rot2).double()))
# print ("d2_sum:", d2_sum)

# data = model2_rot.cpu().numpy()
# colors = 200 * np.ones((data.shape[0], 3), dtype=np.int)

# utils.exportPLY(data, colors, "D:\\Github\\Repos\\dpcr\\utils\\", name = 'm2_rot_aligned')
# utils.exportPLY(model2_rot2.cpu().numpy(), colors, "D:\\Github\\Repos\\dpcr\\utils\\", name = 'm2_rot_aligned2')
# utils.exportPLY(model1.cpu().numpy(), 200 * np.ones((model1.size(0), 3), dtype=np.int), "D:\\Github\\Repos\\dpcr\\utils\\", name = 'm1')
# utils.exportPLY(model2.cpu().numpy(), 200 * np.ones((model2.size(0), 3), dtype=np.int), "D:\\Github\\Repos\\dpcr\\utils\\", name = 'm2')

# print ('finished')

# a = torch.randn((5,3,4))
# print (a)

# b = torch.tensor([[0,1,2], [0,2,1]], dtype=torch.int16)
# print (b)
# print (b.dtype)

# bsi = torch.tensor([[0],[2]])

# batch_pts = a[bsi]
# print ("batch_pts: ", batch_pts)
# print ("batch_pts.size(): ", batch_pts.size())

# print (batch_pts[:,:,b,:])
# print (batch_pts[:,:,b,:].size())

# batch_pts = batch_pts * 2

# print (batch_pts[:,:,b,:])

# for i in range(40):
#     print ('\'data/multi_model_training/scans/amb_%02d.obj\',' % (i))

# parser = argparse.ArgumentParser(description='Point Cloud Part Segmentation')

# parser.add_argument('--train_size', type=int, default=10, metavar='TRN', help='number of train samples to generate')
# parser.add_argument('--test_size', type=int, default=1, metavar='TST', help='number of test samples to generate')
# parser.add_argument('--val_size', type=int, default=1, metavar='VAL', help='number of validation samples to generate')
# parser.add_argument('--dataset', type=str, default='single_bunny', metavar='D', choices=list(generator.getPaths().keys()), help='Training dataset to use')
# parser.add_argument('--h_min', type=int, default=3, metavar='hmin', help='minimum number of initially hidden points')
# parser.add_argument('--h_max', type=int, default=16, metavar='hmax', help='maximum number of initially hidden points')
# parser.add_argument('--n_min', type=int, default=3, metavar='hmin', help='minimum number of iterations per sample')
# parser.add_argument('--n_max', type=int, default=11, metavar='hmax', help='maximum number of iterations per sample')

# parser.add_argument('--predictor_checkpoint', type=str, default='hpc/results/predictor_single_cube_cnet_mish_radam_sd/2020-12-06_164033/predictor_checkpoints.t7', metavar='D', choices=list(generator.getPaths().keys()), help='Training dataset to use')
# parser.add_argument('--detector_checkpoint', type=str, default='hpc/results/detector_single_cube_cnet_mish_radam_sd/2020-12-06_103922/detector_checkpoints.t7', metavar='D', choices=list(generator.getPaths().keys()), help='Training dataset to use')
# parser.add_argument('--corrector_checkpoint', type=str, default=None, metavar='D', choices=list(generator.getPaths().keys()), help='Training dataset to use')

# parser.add_argument('--max_pipeline_iterations', type=int, default=3, metavar='hmax', help='maximum number of iterations per sample')

# args = parser.parse_args()

# # data = generator.getSyntheticDeformTrainData(args)
# data = generator.getCorrectorTrainData(args)

# print (sum([bin.size(0) for bin in data["train_bins"]]))

# for (i, sample) in enumerate(train_data):
#     writer.writeOBJ('./tests/deformations/example_%02d.obj' % (i),sample[0].cpu().numpy(),None,writeFaces=False)