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
from torch.nn.functional import cross_entropy

import argparse
import time
from datetime import datetime
import copy 
import shutil


def getEmptyCheckpoint():

    return  {
                'model_state_dict': [],
                'optimizer_state_dict': [],
                'train_time': [],
                'test_time': [],
                'val_time': [],
                'train_loss': [],
                'train_acc': [],
                'train_C0_acc': [],
                'train_C1_acc': [],
                'test_loss': [],
                'test_acc': [],
                'test_C0_acc': [],
                'test_C1_acc': [],
                'val_loss': [],
                'val_acc': [],
                'val_C0_acc': [],
                'val_C1_acc': [],
                'train_batch_loss': [],
                'train_batch_N': [],
                'train_batch_acc': [],
                'train_batch_C0_acc': [],
                'train_batch_C1_acc': [],
                'train_batch_lr_adjust': [],
                'train_batch_loss_reduction': [],
                'train_settings': [],
                'lr': []
            }

def getTestLoss(pts, samples, model, class_weights):

    cum_loss = 0.0
    cum_acc = 0.0
    cum_acc_c0 = 0.0
    cum_acc_c1 = 0.0

    for sample in samples:

        sample_ids = sample[:,0].flatten().unsqueeze(0)     # size: (1, N_sample)

        # inp = pts[sample_ids].transpose(1,2).contiguous()     # size: (1, 3, N_sample)
        inp = pts[sample_ids].transpose(1,2)                    # size: (1, 3, N_sample)

        # p = model(inp).transpose(1,2).squeeze(0).contiguous()     # size: (N_sample, 2)
        p = model(inp).transpose(1,2).squeeze(0)                    # size: (N_sample, 2)
        t = sample[:,1].flatten().to(p.device)                      # size: (N_sample)

        cum_loss = cum_loss + cross_entropy(p, t, class_weights, reduction='mean').item()

        success_vector = torch.argmax(p, dim=1) == t

        cum_acc += torch.sum(success_vector).item() / success_vector.numel()
        cum_acc_c0 += torch.sum(success_vector[t == 0]).item() / torch.sum(t == 0).item()
        cum_acc_c1 += torch.sum(success_vector[t == 1]).item() / torch.sum(t == 1).item()

    return cum_loss/len(samples), cum_acc/len(samples), cum_acc_c0/len(samples), cum_acc_c1/len(samples)



def train(
        train_data,
        exp_dir = datetime.now().strftime("detector_model/%Y-%m-%d_%H%M"),
        learning_rate = 0.00005,
        rsize = 10,
        epochs = 1,
        checkpoint_path = '',
        seed = 6548,
        batch_size = 4,
        model_type = 'cnet',
        model_cap = 'normal',
        optimizer = 'radam',
        safe_descent = True,
        activation_type = 'mish',
        activation_args = {},
        io = IOStream('run.log'),
        dynamic_lr = True,
        dropout = 0,
        rotations = False,
        use_batch_norm = True,
        batch_norm_momentum = None,
        batch_norm_affine = True,
        use_gc = True,
        no_lr_schedule = False,
        diff_features_only = False,
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
        "\n-------------------------------------------------------" + 
        "\nstart_time: " + datetime.now().strftime("%Y-%m-%d_%H%M%S") +
        "\n-------------------------------------------------------"
    )

    # initialize torch & cuda ---------------------------------------------------------------------

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = utils.getDevice(io)

    # extract train- & test data (and move to device) --------------------------------------------

    pts = train_data["pts"].to(device)
    val_pts = train_data["val_pts"].to(device)

    train_bins = train_data["train_bins"]
    test_samples = train_data["test_samples"]
    val_samples = train_data["val_samples"]

    # the maximum noise offset for each point is equal to the distance to its nearest neighbor
    max_noise = torch.square(pts[train_data["knn"][:, 0]] - pts).sum(dim=1).sqrt()

    # Initialize Model ------------------------------------------------------------------------------

    model_args = {
        'model_type': model_type,
        'model_cap': model_cap,
        'input_channels': pts.size(1),
        'output_channels': 2,
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

    # calculate class weights ---------------------------------------------------------------------

    av_c1_freq = sum([torch.sum(bin[:,:,1]).item() for bin in train_data["train_bins"]]) / sum([bin[:,:,1].numel() for bin in train_data["train_bins"]])
    class_weights = torch.tensor([av_c1_freq, 1-av_c1_freq]).float().to(device)

    io.cprint("\nC0 Weight: %.4f" % (class_weights[0].item()))
    io.cprint("C1 Weight: %.4f" % (class_weights[1].item()))

    # Adjust Weights in favor of C1 (edge:true class)
    # class_weights[0] = class_weights[0] / 2
    # class_weights[1] = 1 - class_weights[0]
    # io.cprint("\nAdjusted C0 Weight: %.4f" % (class_weights[0].item()))
    # io.cprint("Adjusted C1 Weight: %.4f" % (class_weights[1].item()))

    # set up report interval (for logging) and batch size -------------------------------------------------------------------

    report_interval = 100


    # begin training ###########################################################################################################################

    io.cprint("\nBeginning Training..\n")

    for epoch in range(last_epoch + 1, last_epoch + epochs + 1):

        io.cprint("Epoch: %d ------------------------------------------------------------------------------------------" % (epoch))
        io.cprint("Current LR: %.10f" % (opt.param_groups[0]['lr']))

        model.train()
        opt.zero_grad()

        checkpoint['train_batch_loss'].append([])
        checkpoint['train_batch_N'].append([])
        checkpoint['train_batch_acc'].append([])
        checkpoint['train_batch_C0_acc'].append([])
        checkpoint['train_batch_C1_acc'].append([])
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
            batch_rotations = torch.zeros((len(binbatches), batch_size, pts.size(1), pts.size(1)), device=device)
            for i in range(len(binbatches)):
                for j in range(batch_size):
                        batch_rotations[i,j] = utils.getRandomRotation(pts.size(1), device=device)
            print ("created batch rotations (%ds)" % (time.time()-start_rotations))

        b = 0   # batch counter
        
        train_start = time.time()

        for (bin_id, batch_ids) in binbatches:

            b += 1

            batch_pts_ids = train_bins[bin_id][batch_ids][:,:,0]                # size: (B x N)
            batch_input = pts[batch_pts_ids]                                    # size: (B x N x d)
            batch_target = train_bins[bin_id][batch_ids][:,:,1].to(device)      # size: (B x N)

            if batch_rotations[b-1] != None:
                batch_input = batch_input.matmul(batch_rotations[b-1])

            if noise > 0:
                noise_v = torch.randn(batch_input.size(), device=batch_input.device)    # size: (B x N x d)
                noise_v.div_(torch.square(noise_v).sum(dim=2).sqrt()[:,:,None])         # norm to unit vectors
                batch_input.addcmul(noise_v, max_noise[batch_pts_ids][:,:,None], value=noise)

            if scale_min < 1 or scale_max > 1:
                # batch_scales = scale_min + torch.rand(batch_input.size(0), device=batch_input.device) * (scale_max - scale_min)
                batch_scales = torch.rand(batch_input.size(0), device=batch_input.device)
                batch_scales.mul_(scale_max - scale_min)
                batch_scales.add_(scale_min)
                batch_input.mul(batch_scales[:, None, None])

            batch_input = batch_input.transpose(1,2)                            # size: (B x d x N)

            # prediction & loss ----------------------------------------            

            batch_prediction = model(batch_input).transpose(1,2)                # size: (B x N x 2)
            batch_loss = cross_entropy(batch_prediction.reshape(-1,2), batch_target.view(-1), class_weights, reduction='mean')
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
                        
                        # new_batch_prediction = model(batch_input).transpose(1,2).contiguous()
                        new_batch_prediction = model(batch_input).transpose(1,2)
                        new_loss = cross_entropy(new_batch_prediction.reshape(-1,2), batch_target.view(-1), class_weights, reduction='mean').item()
                        
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

            # make class prediction and save stats -----------------------

            success_vector = torch.argmax(batch_prediction, dim=2) == batch_target

            c0_idx = batch_target == 0
            c1_idx = batch_target == 1

            checkpoint['train_batch_acc'][-1].append(torch.sum(success_vector).item() / success_vector.numel())
            checkpoint['train_batch_C0_acc'][-1].append(torch.sum(success_vector[c0_idx]).item() / torch.sum(c0_idx).item())    # TODO handle divsion by zero
            checkpoint['train_batch_C1_acc'][-1].append(torch.sum(success_vector[c1_idx]).item() / torch.sum(c1_idx).item())    # TODO

            # statistic caluclation and output -------------------------

            if b % report_interval == 0:

                last_100_loss = sum(checkpoint['train_batch_loss'][-1][b-report_interval:b]) / report_interval
                last_100_acc = sum(checkpoint['train_batch_acc'][-1][b-report_interval:b]) / report_interval
                last_100_acc_c0 = sum(checkpoint['train_batch_C0_acc'][-1][b-report_interval:b]) / report_interval
                last_100_acc_c1 = sum(checkpoint['train_batch_C1_acc'][-1][b-report_interval:b]) / report_interval

                io.cprint('  Batch %4d to %4d | loss: %.5f%1s| acc: %.4f%1s| C0 acc: %.4f%1s| C1 acc: %.4f%1s| E%3d | T:%5ds | Failed Optims: %3d (%05.2f%%) | Av. Adjust LR: %.6f | Av. Loss Reduction: %07.4f%%' % (  
                                    b - (report_interval - 1),
                                        b,
                                                    last_100_loss,
                                                        '+' if epoch > 1 and last_100_loss < checkpoint['train_loss'][-1] else '',
                                                                    last_100_acc,
                                                                        '+' if epoch > 1 and last_100_acc > checkpoint['train_acc'][-1] else '',
                                                                                    last_100_acc_c0,
                                                                                        '+' if epoch > 1 and last_100_acc_c0 > checkpoint['train_C0_acc'][-1] else '',
                                                                                                    last_100_acc_c1,
                                                                                                        '+' if epoch > 1 and last_100_acc_c1 > checkpoint['train_C1_acc'][-1] else '',
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

        checkpoint['train_loss'].append(sum(checkpoint['train_batch_loss'][-1]) / b)
        checkpoint['train_acc'].append(sum(checkpoint['train_batch_acc'][-1]) / b)
        checkpoint['train_C0_acc'].append(sum(checkpoint['train_batch_C0_acc'][-1]) / b)
        checkpoint['train_C1_acc'].append(sum(checkpoint['train_batch_C1_acc'][-1]) / b)
        checkpoint['train_time'].append(time.time() - train_start)

        io.cprint('----\n  TRN | time: %5ds | loss: %.10f | acc: %.4f | C0 acc: %.4f | C1 acc: %.4f' % 
            (                           checkpoint['train_time'][-1],
                                                    checkpoint['train_loss'][-1],
                                                                checkpoint['train_acc'][-1],
                                                                                checkpoint['train_C0_acc'][-1],
                                                                                            checkpoint['train_C1_acc'][-1]
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

                # run through all train samples again to accumulate layer-wise input distribution statistics (mean and variance) with fixed weights
                # these statistics are later used for the BatchNorm layers during inference
                for (bin_id, batch_ids) in binbatches:

                    batch_pts_ids = train_bins[bin_id][batch_ids][:,:,0]                # size: (B xN)
                    batch_input = pts[batch_pts_ids]                                    # size: (B x N x d)

                    # batch_input = batch_input.transpose(1,2).contiguous()             # size: (B x d x N)
                    batch_input = batch_input.transpose(1,2)                            # size: (B x d x N) 
                    model(batch_input)

                io.cprint('Accumulated BN Layer statistics (%ds)' % (time.time() - eval_bn_start))

            model.eval()

            if len(test_samples) > 0:

                test_start = time.time()

                test_loss, test_acc, test_acc_c0, test_acc_c1 = getTestLoss(pts, test_samples, model, class_weights)

                checkpoint['test_loss'].append(test_loss)
                checkpoint['test_acc'].append(test_acc)
                checkpoint['test_C0_acc'].append(test_acc_c0)
                checkpoint['test_C1_acc'].append(test_acc_c1)

                checkpoint['test_time'].append(time.time() - test_start)

                io.cprint('  TST | time: %5ds | loss: %.10f | acc: %.4f | C0 acc: %.4f | C1 acc: %.4f' % ( checkpoint['test_time'][-1],
                                                                                                    checkpoint['test_loss'][-1],
                                                                                                    checkpoint['test_acc'][-1],
                                                                                                    checkpoint['test_C0_acc'][-1],
                                                                                                    checkpoint['test_C1_acc'][-1])
                )

            else:
                io.cprint('  TST | n/a (no samples)')


            if len(val_samples) > 0:

                val_start = time.time()

                val_loss, val_acc, val_acc_c0, val_acc_c1 = getTestLoss(val_pts, val_samples, model, class_weights)

                checkpoint['val_loss'].append(val_loss)
                checkpoint['val_acc'].append(val_acc)
                checkpoint['val_C0_acc'].append(val_acc_c0)
                checkpoint['val_C1_acc'].append(val_acc_c1)

                checkpoint['val_time'].append(time.time() - val_start)

                io.cprint('  VAL | time: %5ds | loss: %.10f | acc: %.4f | C0 acc: %.4f | C1 acc: %.4f' % ( checkpoint['val_time'][-1],
                                                                                                    checkpoint['val_loss'][-1],
                                                                                                    checkpoint['val_acc'][-1],
                                                                                                    checkpoint['val_C0_acc'][-1],
                                                                                                    checkpoint['val_C1_acc'][-1])
                )
            
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

        torch.save(checkpoint, exp_dir + '/detector_checkpoints.t7')


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

    parser.add_argument('--train_data', type=str, required=True, choices=list(train_data_dict.keys()), metavar='T', help='choose type of training data')
    parser.add_argument('--batch_size', type=int, default=4, metavar='B', help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=100, metavar='E', help='number of epochs to train ')
    parser.add_argument('--lr', type=float, default=0.0005, metavar='LR', help='learning rate')
    parser.add_argument('--seed', type=int, default=seed, metavar='S', help='random seed')
    parser.add_argument('--F', type=int, default=20, metavar='K', help='Num of nearest neighbors to use in the DGCNN model')
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
    with open(train_data_dict[args.train_data]['base'] + '/' + args.train_data + '_train_data', 'rb') as file:
        train_data = torch.load(file, map_location=torch.device('cpu'))

    print ("  > done! (%.2fs)" % (time.time() - start))

    # prepare directories and IOStream -------------------------------------------------------------

    now = datetime.now()
    exp_dir = str("training/checkpoints/detector_"
        + args.train_data + "_"
        + args.model + "_"
        + args.activation + "_"
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
        noise = args.noise
    )
