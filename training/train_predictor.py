import sys
import os
sys.path.insert(1, os.getcwd())  # to load from any submodule in the repo

from models.models import getModel

import utils.dpcrUtils as utils
from utils.radam import RAdam
from utils.ranger2020 import Ranger
from utils.schedules import MultiplicativeAnnealing
from utils.generator import getPaths as getDataPaths

from external.dgcnn.util import IOStream

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiplicativeLR

import argparse
import time
from datetime import datetime
import copy 
import shutil
import itertools


def getEmptyCheckpoint():

    return  {
                'model_state_dict': [],
                'optimizer_state_dict': [],
                'train_time': [],
                'test_time': [],
                'val_time': [],
                'train_loss': [],
                'test_loss': [],
                'val_loss': [],
                'train_batch_loss': [],
                'train_batch_N': [],
                'train_batch_lr_adjust': [],
                'train_batch_loss_reduction': [],
                'train_settings': [],
                'lr': []
            }

# def getBatchLoss(model, pts, batch_sample, neighbors_dirs, perms, loss_function, edge_loss=False, rotations=None):

def getBatchLoss(model, batch_input, batch_target, perms, loss_function, edge_loss=False, rotations=None):

    """
        Computes the (minibatch) loss for a given model

        model       - the (predictor) model to be evaluated
        pts         - all available input points: (N x 3) tensor
        batch_sample   - a complete (batched) sample: (b x N_batch x 2) tensor (b = batch size, N_batch = size of samples in batch) containing ids [:,:,0] and edge mask [:,:,1]

        neighbors_dirs          - for each point in 'pts', the directions of its k nearest neighbors: (N x k x 3) tensor
        perms                   - a (k! x k) tensor of permutations to test (that was used to calculate 'neighbors_dirs_perms')
        edge_loss               - deprecated
        rotations               - rotation matrices for each sample in the batch (B x d x d) tensor

        batch_pts_mask - (optional) the edge maskfor the input batch: (b x N_batch) tensor of {0,1}

        Returns:    A pytorch loss that can be backpropagated

    """

    device = model.base.device

    batch_target_permutations = batch_target[:,:,perms,:].to(device)    # size: (B x N x perm_count x k x d)

    # k = neighbors_dirs.size(1)
    # d = pts.size(1)
    # B = batch_sample.size(0)
    # N = batch_sample.size(1)
    # device = model.base.device

    # batch_sample_ids = batch_sample[:,:,0]          # size: (B x N)

    # batch_input = pts[batch_sample_ids].to(device)                                   # size: (B x N x d)
    # batch_target_permutations = neighbors_dirs[batch_sample_ids][:,:,perms,:].to(device)    # size: (B x N x perm_count x k x d)
    # batch_target = neighbors_dirs[batch_sample_ids].to(device)                             # size: (B x N x k x d)

    # if rotations != None:
    #     batch_input = rotations.to(device).matmul(batch_input.permute(0,2,1)).permute(0,2,1)
    #     batch_target_permutations = rotations.to(device).matmul(batch_target_permutations.permute(2,3,0,4,1)).permute(2,4,0,1,3)
    #     batch_target = rotations.to(device).matmul(batch_target.permute(2,0,3,1)).permute(1,3,0,2)

    # batch_predictions = model(batch_input_tensor.transpose(1,2).contiguous()).transpose(1,2)    # size: (B x N x k * d)
    batch_predictions = model(batch_input.transpose(1,2)).transpose(1,2)     # size: (B x N x k * d)
    batch_predictions = batch_predictions.view(batch_target.size())          # size: (B x N x k x d)

    # for each vertex compute a permutation of neighbors s.t. the sum of differences to the prediction is minimized & expand it to each dimension
    with torch.no_grad():
        gather_idx = utils.matchPointsBatched(batch_predictions, y=None, perms=perms.to(device), y_perms=batch_target_permutations)     # size: (B x N x k)
        gather_idx = gather_idx.unsqueeze(-1)                           # size: (B x N x k x 1)
        gather_idx = gather_idx.expand(-1,-1,-1, batch_predictions.size(-1))  # size: (B x N x k x d)

        batch_target = batch_target.gather(2, gather_idx)    # size: (B x N x k x d)

    batch_loss = loss_function(batch_predictions, batch_target)

    return batch_loss

def getTestLoss(pts, samples, model, neighbors_dirs, perms, loss_function, edge_loss=False):

    cum_test_loss = 0.0
        
    device = model.base.device

    for sample in samples:
        # get loss (batchsize = 1 for test set since there is no binning)

        batch_sample_ids = sample[:,0].unsqueeze(0)          # size: (1 x N)

        batch_input = pts[batch_sample_ids].to(device)                  # size: (1 x N x d)
        batch_target = neighbors_dirs[batch_sample_ids].to(device)      # size: (1 x N x k x d)            

        cum_test_loss += getBatchLoss(model, batch_input, batch_target, perms, loss_function).item()

    return cum_test_loss / len(samples)


# def getNewBatchLoss(model, pts, batch_ids, neighbors_dirs, loss_function):

#     input_tensor = pts[batch_ids].transpose(1,2).contiguous()   # size: (batchsize, d, N_batch)
    
#     batchsize = input_tensor.size(0)
#     d = input_tensor.size(1)
#     N_batch = input_tensor.size(2)

#     y = model(input_tensor)                              # size: (batchsize, d * k, N_batch)
#     y = y.transpose(1,2).contiguous()                    # size: (batchsize, N_batch, d * k)
#     y = y.view((batchsize, N_batch, -1, d))              # size: (batchsize, N_batch, k, d)
    
#     y_norm = torch.sqrt(torch.sum(torch.square(y), dim=-1)) # size: (batchsize, N_batch, k)
#     y_n = y / y_norm.unsqueeze(-1)   # size: (batchsize, N_batch, k, d)

#     yy = 2 * torch.matmul(y_n, y_n.transpose(-2, -1))     # (batchsize, N_batch, k, k)
#     y2 = torch.sum(y_n**2, dim=-1, keepdim=True)       # (batchsize, N_batch, k, 1)

#     # internal directions
#     dir_yy = y_n.unsqueeze(-2) - y_n.unsqueeze(-3) # (batchsize, N_batch, k, k, d)

#     # calculate spring force by using inner distances (of normed input vectors)
#     Sf_yy = torch.exp(-(y2 - yy + y2.transpose(-2, -1))) # (batchsize, N_batch, k, k)

#     # calculate displacement
#     Dp_yy = torch.sum(dir_yy * Sf_yy.unsqueeze(-1), dim=-2).squeeze(-2) # (batchsize, N_batch, k, d)

#     # calculate target and norm to same length as y
#     T_y = y_n + Dp_yy    # (batchsize, N_batch, k, d)
#     T_y = T_y / torch.sqrt(torch.sum(torch.square(T_y), dim=-1)).unsqueeze(-1)   # (batchsize, N_batch, k, d)
#     T_y = T_y * y_norm.unsqueeze(-1)     # (batchsize, N_batch, k, d)
    
#     # calculate internal loss
#     L_yy = loss_function(y, T_y)
    
#     # ---------------------------------------------------------------------------
    
#     t = neighbors_dirs[batch_ids] # (batch_size, N_batch, k, d)
    
#     yt = 2 * torch.matmul(y, t.transpose(3, 2))     # (batch_size, N_batch, k, k)
#     y2 = torch.sum(y**2, dim=3, keepdim=True)       # (batch_size, N_batch, 1, k)
#     t2 = torch.sum(t**2, dim=3, keepdim=True)       # (batch_size, N_batch, 1, k)
    
#     # distances to solutions
#     D_yt = y2 - yt + t2.transpose(3, 2) # (batch_size, N_batch, k, k)
#     D_yt[D_yt == 0] = torch.max(D_yt) + 1.0
    
#     # minimum distances to each solution
#     Dt_min = torch.min(D_yt, dim = 2)[1]      # (batch_size, N_batch, k)

#     # nearest true neighbors
#     t_min = t.gather(dim=2, index=Dt_min.unsqueeze(-1).expand((-1,-1,-1,d))) # (batch_size, N_batch, k, d)

#     L_yt = loss_function(y, t_min)

#     return L_yy, L_yt



def train(
        train_data,
        exp_dir = datetime.now().strftime("predictor_model/%Y-%m-%d_%H%M"),
        learning_rate = 0.00005,
        rsize = 10,
        epochs = 1,
        checkpoint_path = '',
        seed = 6548,
        batch_size = 4,
        neighbors_count = 6,
        edge_loss = False,
        model_type = 'cnet',
        model_cap = 'normal',
        optimizer = 'radam',
        safe_descent = True,
        activation_type = 'mish',
        activation_args = {},
        io = None,
        dynamic_lr = True,
        dropout = 0,
        rotations = False,
        use_batch_norm = True,
        batch_norm_momentum = None,
        batch_norm_affine = True,
        use_gc = True,
        no_lr_schedule = False,
        diff_features_only = False,
        gpu_preload = True,
        scale_min = 1,
        scale_max = 1,
        noise = 0
    ):

    start_time = time.time()

    scale_min = scale_min if scale_min < 1 else 1
    scale_max = scale_max if scale_max > 1 else 1

    io.cprint(
        "-------------------------------------------------------" +
        "\nexport dir = " + '/checkpoints/' + exp_dir + 
        "\nbase_learning_rate = " + str(learning_rate) +
        "\nuse_batch_norm = " + str(use_batch_norm) +
        "\nbatch_norm_momentum = " + str(batch_norm_momentum) +
        "\nbatch_norm_affine = " + str(batch_norm_affine) +
        "\nno_lr_schedule = " + str(no_lr_schedule) +
        "\nuse_gc = " + str(use_gc) +
        "\nrsize = " + str(rsize) +
        "\npython_version: " + sys.version +
        "\ntorch_version: " + torch.__version__ +
        "\nnumpy_version: " + np.version.version +
        "\nmodel_type: " + model_type +
       "\nmodel_cap: " + model_cap +
        "\noptimizer: " + optimizer +
        "\nactivation_type: " + activation_type +
        "\nsafe_descent: " + str(safe_descent) +
        "\ndynamic_lr: " + str(dynamic_lr) +
        "\nrotations: " + str(rotations) +
        "\nscaling: " + str(scale_min) + " to " + str(scale_max) +
        "\nnoise: " + str(noise) +
        "\nepochs = " + str(epochs) + 
        (("\ncheckpoint = " + checkpoint_path) if checkpoint_path != '' else '') + 
        "\nseed = " + str(seed) + 
        "\nbatch_size = " + str(batch_size) +
        "\n#train_data = " + str(sum([bin.size(0) for bin in train_data["train_bins"]])) +
        "\n#test_data = " + str(len(train_data["test_samples"])) +
        "\n#validation_data = " + str(len(train_data["val_samples"])) +
        "\nneighbors_count = " + str(neighbors_count) + 
        "\nedge_loss = " + str(edge_loss) + 
        "\n-------------------------------------------------------" + 
        "\nstart_time: " + datetime.now().strftime("%Y-%m-%d_%H%M%S") +
        "\n-------------------------------------------------------"
        )

    # initialize torch & cuda ---------------------------------------------------------------------

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = utils.getDevice(io)

    # extract train- & test data (and move to device) --------------------------------------------

    pts = train_data["pts"].to(device if gpu_preload else torch.device('cpu'))
    knn = train_data["knn"]

    val_pts = train_data["val_pts"].to(device if gpu_preload else torch.device('cpu'))
    val_knn = train_data["val_knn"]

    print ("pts.device:", pts.device)
    print ("val_pts.device:", val_pts.device)

    train_bins = train_data["train_bins"]
    test_samples = train_data["test_samples"]
    val_samples = train_data["val_samples"]

    # the maximum noise offset for each point is equal to the distance to its nearest neighbor
    max_noise = torch.square(pts[knn[:, 0]] - pts).sum(dim=1).sqrt()

    # Initialize Model ------------------------------------------------------------------------------

    model_args = {
        'model_type': model_type,
        'model_cap': model_cap,
        'input_channels': pts.size(1),
        'output_channels': pts.size(1) * neighbors_count,
        'rsize': rsize,
        'emb_dims': 1024,
        'activation_type': activation_type,
        'activation_args': activation_args,
        'dropout': dropout,
        'batch_norm': use_batch_norm,
        'batch_norm_affine': batch_norm_affine,
        'batch_norm_momentum': batch_norm_momentum,
        'diff_features_only': diff_features_only
    }

    model = getModel(model_args).to(device)

    # init optimizer & scheduler -------------------------------------------------------------------

    lookahead_sync_period = 6

    opt = None
    if optimizer == 'radam':
        opt = RAdam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, use_gc=use_gc)
    elif optimizer == 'lookahead':
        opt = Ranger(model.parameters(), lr=learning_rate, alpha=0.9, k=lookahead_sync_period)

    # make sure that either a LR schedule is given or dynamic LR is enabled
    assert dynamic_lr or not no_lr_schedule

    scheduler = None if no_lr_schedule else MultiplicativeLR(opt, lr_lambda=MultiplicativeAnnealing(epochs))

    # set train settings & load previous model state ------------------------------------------------------------

    checkpoint = getEmptyCheckpoint()
    last_epoch = 0

    if (checkpoint_path != ''):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'][-1])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'][-1])
        last_epoch = len(checkpoint['model_state_dict'])
        print ('> loaded checkpoint! (%d epochs)' % (last_epoch))

    checkpoint['train_settings'].append({
        'learning_rate': learning_rate,
        'scheduler': scheduler,
        'epochs': epochs,
        'seed': seed,
        'batch_size': batch_size,
        'neighbors_count': neighbors_count,
        'edge_loss': edge_loss,
        'optimizer': optimizer,
        'safe_descent:': str(safe_descent),
        'dynamic_lr': str(dynamic_lr),
        'rotations': str(rotations),
        'scale_min': scale_min,
        'scale_max': scale_max,
        'noise': noise,
        'train_data_count': sum([bin.size(0) for bin in train_data["train_bins"]]),
        'test_data_count': len(train_data["test_samples"]),
        'validation_data_count': len(train_data["val_samples"]),
        'model_args': model_args
    })

    # pre-permute the neighbors (labels) ----------------------------------------------------------

    # io.cprint ("\nPre-Computing Neighbor Permutations..")
    io.cprint ("\nPreparing Data..")

    start = time.time()

    neighbors_dirs = (pts[knn] - pts[:, None, :]).to(device if gpu_preload else torch.device('cpu'))
    val_neighbors_dirs = (val_pts[val_knn] - val_pts[:, None, :]).to(device if gpu_preload else torch.device('cpu'))

    perms_np = np.array(list(itertools.permutations(np.arange(neighbors_count))))
    perms = torch.from_numpy(perms_np).long().to(device if gpu_preload else torch.device('cpu'))

    end = time.time()

    io.cprint ("  > done! (%.2fs)" % (end - start))

    # set up report interval (for logging) and batch size -------------------------------------------------------------------

    report_interval = 100
    loss_function = torch.nn.MSELoss(reduction='mean')


    # begin training ###########################################################################################################################

    io.cprint("\nBeginning Training..\n")

    for epoch in range(last_epoch + 1, last_epoch + epochs + 1):

        io.cprint("Epoch: %d ------------------------------------------------------------------------------------------" % (epoch))
        io.cprint("Current LR: %.10f" % (opt.param_groups[0]['lr']))

        model.train()
        opt.zero_grad()

        checkpoint['train_batch_loss'].append([])
        checkpoint['train_batch_N'].append([])
        checkpoint['train_batch_lr_adjust'].append([])
        checkpoint['train_batch_loss_reduction'].append([])
        checkpoint['lr'].append(opt.param_groups[0]['lr'])

        # draw random batches from random bins
        binbatches = utils.drawBinBatches([bin.size(0) for bin in train_bins], batchsize=batch_size)

        checkpoint['train_batch_N'][-1] = [train_bins[bin_id][batch_ids].size(1) for (bin_id, batch_ids) in binbatches]

        failed_loss_optims = 0
        cum_lr_adjust_fac = 0
        cum_loss_reduction = 0

        # pre-compute random rotations if needed
        batch_rotations = [None] * len(binbatches)
        if rotations:
            start_rotations = time.time()
            batch_rotations = torch.zeros((len(binbatches), batch_size, pts.size(1), pts.size(1)), device=(device if gpu_preload else torch.device('cpu')))
            for i in range(len(binbatches)):
                for j in range(batch_size):
                        batch_rotations[i,j] = utils.getRandomRotation(pts.size(1), device=(device if gpu_preload else torch.device('cpu')))
            print ("created batch rotations (%ds)" % (time.time()-start_rotations))

        b = 0   # batch counter

        train_start = time.time()

        for (bin_id, batch_ids) in binbatches:

            b += 1

            # prediction & loss ----------------------------------------

            batch_sample = train_bins[bin_id][batch_ids]    # size: (B x N x 2)
            batch_sample_ids = batch_sample[:,:,0]          # size: (B x N)

            batch_input = pts[batch_sample_ids].to(device)                  # size: (B x N x d)
            batch_target = neighbors_dirs[batch_sample_ids].to(device)      # size: (B x N x k x d)
            
            if noise > 0:
                noise_v = torch.randn(batch_input.size(), device=batch_input.device)        # size: (B x N x d)
                noise_v.div_(torch.square(noise_v).sum(dim=2).sqrt()[:,:,None])             # norm to unit vectors
                batch_input.addcmul(noise_v, max_noise[batch_sample_ids][:,:,None], value=noise)      # size: (B x N x d)
                # noise is not applied to targets

            if scale_min < 1 or scale_max > 1:
                # batch_scales = scale_min + torch.rand(batch_input.size(0), device=batch_input.device) * (scale_max - scale_min) # size: (B)
                batch_scales = torch.rand(batch_input.size(0), device=batch_input.device)
                batch_scales.mul_(scale_max - scale_min)
                batch_scales.add_(scale_min)
                batch_input.mul(batch_scales[:, None, None])               # size: (B x N x k x d)
                batch_target.mul(batch_scales[:, None, None, None])        # size: (B x N x k x d)

            if batch_rotations[b-1] != None:
                batch_input = batch_rotations[b-1].to(device).matmul(batch_input.permute(0,2,1)).permute(0,2,1)         # size: (B x N x d)
                batch_target = batch_rotations[b-1].to(device).matmul(batch_target.permute(2,0,3,1)).permute(1,3,0,2)   # size: (B x N x k x d)

            # batch_loss = getBatchLoss(model, pts, batch_sample, neighbors_dirs, perms, loss_function, edge_loss = edge_loss, rotations=batch_rotations[b-1])
            batch_loss = getBatchLoss(model, batch_input, batch_target, perms, loss_function)

            # make_dot(batch_loss, params=dict(model.named_parameters())).render("graph", format="pdf")
            # exit()

            batch_loss.backward()

            checkpoint['train_batch_loss'][-1].append(batch_loss.item())

            new_loss = 0.0
            lr_adjust = 1.0
            loss_reduction = 0.0

            # if safe descent is enabled, try to optimize the descent step so that a reduction in loss is guaranteed
            if safe_descent:

                # create backups to restore states before the optimizer step
                model_state_backup = copy.deepcopy(model.state_dict())
                opt_state_backup = copy.deepcopy(opt.state_dict())
                
                # make an optimizer step
                opt.step()

                # in each itearation, check if the optimzer gave an improvement
                # if not, restore the original states, reduce the learning rate and try again
                # no gradient needed for the plain loss calculation
                with torch.no_grad():
                    for i in range(10):
                        
                        new_loss = getBatchLoss(model, batch_input, batch_target, perms, loss_function).item()
                        # new_loss = getBatchLoss(model, pts, batch_sample, neighbors_dirs, perms, loss_function, edge_loss=edge_loss, rotations=batch_rotations[b-1]).item()
                        
                        # if the model performs better now we continue, if not we try a smaller learning step
                        if (new_loss < batch_loss.item()):
                            # print("lucky! (%f -> %f) reduction: %.4f%%" % (batch_loss.item(), new_loss, 100 * (batch_loss.item()-new_loss) / batch_loss.item()))
                            break
                        else:
                            # print("try again.. (%f -> %f)" % (batch_loss.item(), new_loss))
                            model.load_state_dict(model_state_backup)
                            opt.load_state_dict(opt_state_backup)
                            lr_adjust *= 0.7
                            opt.step(lr_adjust = lr_adjust)

                loss_reduction = 100 * (batch_loss.item()-new_loss) / batch_loss.item()

                if new_loss >= batch_loss.item():
                    failed_loss_optims += 1
                else:
                    cum_lr_adjust_fac += lr_adjust
                    cum_loss_reduction += loss_reduction
 
            else:

                cum_lr_adjust_fac += lr_adjust
                opt.step()

            checkpoint['train_batch_lr_adjust'][-1].append(lr_adjust)
            checkpoint['train_batch_loss_reduction'][-1].append(loss_reduction)   
              
            # reset gradients
            opt.zero_grad()    

            # statistic caluclation and output -------------------------

            if b % report_interval == 0:

                # SPECIAL: only do this in LONG running training processes.. Save model and optimizer state ..
                if len(checkpoint['model_state_dict']) == 0:
                    checkpoint['model_state_dict'] = [None]
                if len(checkpoint['optimizer_state_dict']) == 0:
                    checkpoint['optimizer_state_dict'] = [None]
                checkpoint['model_state_dict'][-1] = copy.deepcopy(model.state_dict())
                checkpoint['optimizer_state_dict'][-1] = copy.deepcopy(opt.state_dict())

                torch.save(checkpoint, exp_dir + '/predictor_checkpoints.t7')

                last_100_loss = sum(checkpoint['train_batch_loss'][-1][b-report_interval:b]) / report_interval
                improvement_indicator = '+' if epoch > 1 and last_100_loss < checkpoint['train_loss'][-1] else ''

                io.cprint('  Batch %4d to %4d | loss: %.10f%1s | av. dist. per neighbor: %.10f | E%3d | T:%5ds | Failed Optims: %3d (%05.2f%%) | Av. Adjust LR: %.6f | Av. Loss Reduction: %07.4f%%' %
                    (
                                    b - (report_interval - 1),
                                            b,
                                                    last_100_loss,
                                                        improvement_indicator,
                                                                                        np.sqrt(last_100_loss),
                                                                                                epoch,
                                                                                                        time.time() - train_start,
                                                                                                                                failed_loss_optims,
                                                                                                                                    100*(failed_loss_optims / report_interval),
                                                                                                                                                                (cum_lr_adjust_fac / (report_interval-failed_loss_optims) if failed_loss_optims < report_interval else -1),
                                                                                                                                                                                        (cum_loss_reduction / (report_interval-failed_loss_optims) if failed_loss_optims < report_interval else -1)
                    )
                )

                failed_loss_optims = 0
                cum_lr_adjust_fac = 0
                cum_loss_reduction = 0

                # break

        checkpoint['train_loss'].append(sum(checkpoint['train_batch_loss'][-1]) / b)
        checkpoint['train_time'].append(time.time() - train_start)

        io.cprint('----\n  TRN | time: %5ds | loss: %.10f| av. dist. per neighbor: %.10f' % 
            (                           checkpoint['train_time'][-1],
                                                    checkpoint['train_loss'][-1],
                                                        np.sqrt(checkpoint['train_loss'][-1])
            )
        )

        torch.cuda.empty_cache()

        ####################
        # Test & Validation
        ####################

        with torch.no_grad():

            if use_batch_norm:

                model.eval_bn()

                eval_bn_start = time.time()

                # run through all train samples again to accumulate (current!) layer-wise input distribution statistics (mean and variance) with fixed weights
                # these statistics are later used for the BatchNorm layers during inference
                for (bin_id, batch_ids) in binbatches:

                    batch_sample_ids = train_bins[bin_id][batch_ids][:,:,0].to(device)     # size: (B x N)
                    batch_input = pts[batch_sample_ids].to(device)                         # size: (B x N x d)

                    model(batch_input.transpose(1,2))

                io.cprint('Accumulated BN Layer statistics (%ds)' % (time.time() - eval_bn_start))

            model.eval()

            if len(test_samples) > 0:

                test_start = time.time()

                test_loss = getTestLoss(pts, test_samples, model, neighbors_dirs, perms, loss_function, edge_loss=edge_loss)
                
                checkpoint['test_loss'].append(test_loss)
                checkpoint['test_time'].append(time.time() - test_start)

                io.cprint('  TST | time: %5ds | loss: %.10f| av. dist. per neighbor: %.10f' %
                    ( checkpoint['test_time'][-1], checkpoint['test_loss'][-1], np.sqrt(checkpoint['test_loss'][-1])))
            else:
                io.cprint('  TST | n/a (no samples)')


            if len(val_samples) > 0:

                val_start = time.time()

                val_loss = getTestLoss(pts, val_samples, model, val_neighbors_dirs, perms, loss_function, edge_loss=edge_loss)
                
                checkpoint['val_loss'].append(val_loss)
                checkpoint['val_time'].append(time.time() - val_start)

                io.cprint('  VAL | time: %5ds | loss: %.10f| av. dist. per neighbor: %.10f' %
                    ( checkpoint['val_time'][-1], checkpoint['val_loss'][-1], np.sqrt(checkpoint['val_loss'][-1])))

            else:
                io.cprint('  VAL | n/a (no samples)')

        ####################
        # Scheduler Step
        ####################

        if not no_lr_schedule:
            scheduler.step()

        if epoch > 1 and dynamic_lr and sum(checkpoint['train_batch_lr_adjust'][-1]) > 0:
            io.cprint("----\n  dynamic lr adjust: %.10f" % (0.5 * (1 + sum(checkpoint['train_batch_lr_adjust'][-1]) / len(checkpoint['train_batch_lr_adjust'][-1]))))
            for param_group in opt.param_groups:
                param_group['lr'] *= 0.5 * (1 + sum(checkpoint['train_batch_lr_adjust'][-1]) / len(checkpoint['train_batch_lr_adjust'][-1]))


        # Save model and optimizer state ..
        checkpoint['model_state_dict'].append(copy.deepcopy(model.state_dict()))
        checkpoint['optimizer_state_dict'].append(copy.deepcopy(opt.state_dict()))

        torch.save(checkpoint, exp_dir + '/predictor_checkpoints.t7')


    io.cprint(
        "\n-------------------------------------------------------" + 
        ("\ntotal_time: %.2fh" % ((time.time() - start_time) / 3600)) +
        ("\ntrain_time: %.2fh" % (sum(checkpoint['train_time']) / 3600)) +
        ("\ntest_time: %.2fh" % (sum(checkpoint['test_time']) / 3600)) +
        ("\nval_time: %.2fh" % (sum(checkpoint['val_time']) / 3600)) +
        "\n-------------------------------------------------------" + 
        "\nend_time: " + datetime.now().strftime("%Y-%m-%d_%H%M%S") +
        "\n-------------------------------------------------------"
    )


if __name__ == "__main__":

    print ("Python Version:", sys.version)
    
    parser = argparse.ArgumentParser(description='Point Cloud Part Segmentation')

    seed_file = open('utils/seed.txt', "r")
    seed = int(seed_file.read())
    seed_file.close()

    train_data_dict = getDataPaths()

    # parser.add_argument('--train_data', type=str, required=True, choices=list(train_data_dict.keys()), metavar='T', help='choose type of training data')
    parser.add_argument('--dataset', required=True, type=str, default='multi_faces', metavar='D', choices=list(train_data_dict.keys()), help='Training dataset to use')
    parser.add_argument('--batch_size', type=int, default=4, metavar='B', help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=100, metavar='E', help='number of epochs to train ')
    parser.add_argument('--lr', type=float, default=0.0005, metavar='LR', help='learning rate')
    parser.add_argument('--seed', type=int, default=seed, metavar='S', help='random seed')
    parser.add_argument('--F', type=int, default=20, metavar='K', help='Num of nearest neighbors to use in the DGCNN model')
    parser.add_argument('--neighbors', type=int, default=6, metavar='N', help='Num of nearest neighbors to predict')
    parser.add_argument('--checkpoint', type=str, default='', metavar='C',help='Path to previous model checkpoint')
    parser.add_argument('--model', type=str, default='mish', metavar='M', choices=['semseg', 'unet','unet_plus','cnet'], help='Model definition to use, [classic, unet, unet_plus, cnet, cnet_plus]')
    parser.add_argument('--model_cap', type=str, default='normal', metavar='M', choices=['normal', 'small', 'smaller'], help='Model capacity to use, [normal, small, smaller]')
    parser.add_argument('--optimizer', type=str, default='radam', metavar='O', choices=['radam', 'lookahead'], help='Optimizer to use, [radam, lookahead]')
    parser.add_argument('--activation', type=str, default='mish', metavar='A', choices=['mish', 'relu', 'swish', 'splash', 'squish'], help='Activation function to use, [mish, relu, swish, splash, squish]')
    parser.add_argument('--safe_descent', default=False, action='store_true')
    parser.add_argument('--dynamic_lr', default=False, action='store_true')
    parser.add_argument('--dropout', type=float, default=0, metavar='D', help='dropout probability')
    parser.add_argument('--scale_min', type=float, default=1, metavar='SMIN', help='minimum random scaling')
    parser.add_argument('--scale_max', type=float, default=1, metavar='SMAX', help='maximum random scaling')
    parser.add_argument('--noise', type=float, default=0, metavar='NOI', help='random noise per batch')
    parser.add_argument('--edge_loss', default=False, action='store_true')
    parser.add_argument('--rotations', default=False, action='store_true')
    parser.add_argument('--use_gc', default=False, action='store_true')
    parser.add_argument('--diff_features_only', default=False, action='store_true')
    parser.add_argument('--no_batchnorm', default=False, action='store_true')
    parser.add_argument('--no_lr_schedule', default=False, action='store_true')
    parser.add_argument('--static_batchnorm', default=False, action='store_true')
    
    args = parser.parse_args()

    print("Reading Train- and Test Data..")

    start = time.time()

    train_data = None
    with open(train_data_dict[args.dataset]['base'] + '/' + args.dataset + '_train_data', 'rb') as file:
        train_data = torch.load(file, map_location=torch.device('cpu'))

    print ("  > done! (%.2fs)" % (time.time() - start))

    # prepare directories and IOStream -------------------------------------------------------------

    now = datetime.now()
    exp_dir = str("training/checkpoints/predictor_"
        + args.dataset + "_"
        + args.model + "_"
        + args.activation +  "_"
        + args.optimizer
        + ("_sd" if args.safe_descent else "")
        + ("_noise" if args.noise > 0 else "")
        + ("_scale" if args.scale_min < 1 or args.scale_max > 1 else "")
        + "/" + now.strftime("%Y-%m-%d_%H%M%S")
    )
    
    os.makedirs(exp_dir, exist_ok=True)
    
    io = IOStream(exp_dir + '/run.log')
    io.cprint('Command: ' + str(sys.argv))

    # copy model- and training files
    shutil.copy('training/train_detector.py', exp_dir + '/train_detector.py')
    shutil.copy('models/models.py', exp_dir + '/models.py')

    train(
        train_data = train_data,
        exp_dir = exp_dir,
        io = io,
        learning_rate = args.lr,
        rsize = args.F,
        epochs = args.epochs,
        checkpoint_path = args.checkpoint,
        seed = args.seed,
        batch_size = args.batch_size,
        neighbors_count = args.neighbors,
        edge_loss=args.edge_loss,
        model_type = args.model,
        model_cap=args.model_cap,
        optimizer = args.optimizer,
        safe_descent=args.safe_descent,
        dynamic_lr=args.dynamic_lr,
        dropout=args.dropout,
        rotations=args.rotations,
        use_gc = args.use_gc,
        use_batch_norm=not args.no_batchnorm,
        batch_norm_affine = not args.static_batchnorm,
        no_lr_schedule = args.no_lr_schedule,
        diff_features_only = args.diff_features_only,
        scale_min = args.scale_min,
        scale_max = args.scale_max,
        noise = args.noise,
        gpu_preload = False,
    )
