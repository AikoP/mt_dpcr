import sys
import os
sys.path.insert(1, os.getcwd())  # to load from any submodule in the repo

from models.models import getModel

import utils.dpcrUtils as utils
import utils.generator as generator
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

def getBatchLoss(model, batch_sample, loss_function, edge_loss = False, rotations=None):

    """
        Computes the (minibatch) loss for a given model

        model           - the (predictor) model to be evaluated
        batch_sample    - batch_sample containing input and target: (B x N x d x 2) tensor
        loss_function   - the loss function to be used

        edge_loss - (optional) wether the loss should only be computed from points where the target is nonzero
        rotations - rotation matrices for each sample in the batch (B x d x d) tensor

        Returns:    A pytorch loss that can be backpropagated

    """

    batch_sample = batch_sample.to(model.base.device)

    if rotations != None:
        batch_sample = batch_sample.permute(3,0,1,2).matmul(rotations).permute(1,2,3,0)

    input = batch_sample[:,:,:,0].squeeze(-1)    # size: (B x N x d)
    target = batch_sample[:,:,:,1].squeeze(-1)   # size: (B x N x d)
            
    prediction = model(input.transpose(1,2)).transpose(1,2)   # size: (B x N x d)

    if edge_loss:
        prediction[(target == 0).min(dim=-1)[0]] = 0

    return loss_function(prediction, target)

def getTestLoss(model, samples, loss_function, edge_loss=False):

    cum_test_loss = 0.0

    for sample in samples:
        batch_sample = sample.unsqueeze(0)  # size: (1 x N x d, 2)
        cum_test_loss += getBatchLoss(model, batch_sample, loss_function, edge_loss = edge_loss).item()

    return cum_test_loss / len(samples)


def train(
        train_data,
        exp_dir = datetime.now().strftime("corrector_model/%Y-%m-%d_%H%M"),
        learning_rate = 0.00005,
        rsize = 10,
        epochs = 1,
        checkpoint_path = '',
        seed = 6548,
        batch_size = 4,
        edge_loss = False,
        model_type = 'cnet',
        model_cap = 'normal',
        optimizer_type = 'radam',
        reset_optimizer = True,         # if true, does not load optimizer chekcpoints
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
        diff_features_only = False
    ):

    start_time = time.time()

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
        "\noptimizer: " + optimizer_type +
        "\nactivation_type: " + activation_type +
        "\nsafe_descent: " + str(safe_descent) +
        "\ndynamic_lr: " + str(dynamic_lr) +
        "\nrotations: " + str(rotations) +
        "\nepochs = " + str(epochs) + 
        (("\ncheckpoint = " + checkpoint_path) if (checkpoint_path != None and checkpoint_path != '') else '') + 
        "\nseed = " + str(seed) + 
        "\nbatch_size = " + str(batch_size) +
        "\n#train_data = " + str(sum([bin.size(0) for bin in train_data["train_bins"]])) +
        "\n#test_data = " + str(len(train_data["test_samples"])) +
        "\n#validation_data = " + str(len(train_data["val_samples"])) + 
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

    # train_bins = [bin.float().to(device) for bin in train_data["train_bins"]]
    # test_samples = [sample.float().to(device) for sample in train_data["test_samples"]]
    # val_samples = [sample.float().to(device) for sample in train_data["val_samples"]]

    train_bins = [bin.float() for bin in train_data["train_bins"]]
    test_samples = [sample.float() for sample in train_data["test_samples"]]
    val_samples = [sample.float() for sample in train_data["val_samples"]]

    # Initialize Model ------------------------------------------------------------------------------

    model_args = {
        'model_type': model_type,
        'model_cap': model_cap,
        'input_channels': test_samples[0].size(1),
        'output_channels': test_samples[0].size(1),
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

    optimizer = None
    if optimizer_type == 'radam':
        optimizer = RAdam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, use_gc=use_gc)
    elif optimizer_type == 'lookahead':
        optimizer = Ranger(model.parameters(), lr=learning_rate, alpha=0.9, k=lookahead_sync_period)

    # make sure that either a LR schedule is given or dynamic LR is enabled
    assert dynamic_lr or not no_lr_schedule

    scheduler = None if no_lr_schedule else MultiplicativeLR(optimizer, lr_lambda=MultiplicativeAnnealing(epochs))

    # set train settings & load previous model state ------------------------------------------------------------

    checkpoint = getEmptyCheckpoint()
    last_epoch = 0

    if (checkpoint_path != None and checkpoint_path != ''):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'][-1])
        if not reset_optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'][-1])
        last_epoch = len(checkpoint['model_state_dict'])
        print ('> loaded checkpoint! (%d epochs)' % (last_epoch))

    checkpoint['train_settings'].append({
        'learning_rate': learning_rate,
        'scheduler': scheduler,
        'epochs': epochs,
        'seed': seed,
        'batch_size': batch_size,
        'edge_loss': edge_loss,
        'optimizer': optimizer_type,
        'safe_descent:': str(safe_descent),
        'dynamic_lr': str(dynamic_lr),
        'rotations': str(rotations),
        'train_data_count': sum([bin.size(0) for bin in train_data["train_bins"]]),
        'test_data_count': len(train_data["test_samples"]),
        'validation_data_count': len(train_data["val_samples"]),
        'model_args': model_args
    })

    # set up report interval (for logging) and batch size -------------------------------------------------------------------

    report_interval = 100
    loss_function = torch.nn.MSELoss(reduction='mean')


    # begin training ###########################################################################################################################

    io.cprint("\nBeginning Training..\n")

    for epoch in range(last_epoch + 1, last_epoch + epochs + 1):

        io.cprint("Epoch: %d ------------------------------------------------------------------------------------------" % (epoch))
        io.cprint("Current LR: %.10f" % (optimizer.param_groups[0]['lr']))

        model.train()
        optimizer.zero_grad()

        checkpoint['train_batch_loss'].append([])
        checkpoint['train_batch_N'].append([])
        checkpoint['train_batch_lr_adjust'].append([])
        checkpoint['train_batch_loss_reduction'].append([])
        checkpoint['lr'].append(optimizer.param_groups[0]['lr'])

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
            batch_rotations = torch.zeros((len(binbatches), batch_size, test_samples[0].size(1), test_samples[0].size(1)), device=device)
            for i in range(len(binbatches)):
                for j in range(batch_size):
                        batch_rotations[i,j] = utils.getRandomRotation(test_samples[0].size(1), device=device)
            print ("created batch rotations (%ds)" % (time.time()-start_rotations))

        b = 0   # batch counter

        train_start = time.time()

        for (bin_id, batch_ids) in binbatches:

            b += 1

            # print ("handling batch %d" % (b))
                
            # prediction & loss ----------------------------------------

            batch_sample = train_bins[bin_id][batch_ids].to(model.base.device)    # size: (B x N x d x 2)

            batch_loss = getBatchLoss(model, batch_sample, loss_function, edge_loss = edge_loss, rotations=batch_rotations[b-1])
            batch_loss.backward()

            checkpoint['train_batch_loss'][-1].append(batch_loss.item())
            
            new_loss = 0.0
            lr_adjust = 1.0
            loss_reduction = 0.0

            # if safe descent is enabled, try to optimize the descent step so that a reduction in loss is guaranteed
            if safe_descent:

                # create backups to restore states before the optimizer step
                model_state_backup = copy.deepcopy(model.state_dict())
                opt_state_backup = copy.deepcopy(optimizer.state_dict())
                
                # make an optimizer step
                optimizer.step()

                # in each itearation, check if the optimzer gave an improvement
                # if not, restore the original states, reduce the learning rate and try again
                # no gradient needed for the plain loss calculation
                with torch.no_grad():
                    for i in range(10):
                        
                        new_loss = getBatchLoss(model, batch_sample, loss_function, edge_loss = edge_loss, rotations=batch_rotations[b-1]).item()
                        
                        # if the model performs better now we continue, if not we try a smaller learning step
                        if (new_loss < batch_loss.item()):
                            # print("lucky! (%f -> %f) reduction: %.4f%%" % (batch_loss.item(), new_loss, 100 * (batch_loss.item()-new_loss) / batch_loss.item()))
                            break
                        else:
                            # print("try again.. (%f -> %f)" % (batch_loss.item(), new_loss))
                            model.load_state_dict(model_state_backup)
                            optimizer.load_state_dict(opt_state_backup)
                            lr_adjust *= 0.7
                            optimizer.step(lr_adjust = lr_adjust)

                loss_reduction = 100 * (batch_loss.item()-new_loss) / batch_loss.item()

                if new_loss >= batch_loss.item():
                    failed_loss_optims += 1
                else:
                    cum_lr_adjust_fac += lr_adjust
                    cum_loss_reduction += loss_reduction
 
            else:

                cum_lr_adjust_fac += lr_adjust
                optimizer.step()

            checkpoint['train_batch_lr_adjust'][-1].append(lr_adjust)
            checkpoint['train_batch_loss_reduction'][-1].append(loss_reduction)   
              
            # reset gradients
            optimizer.zero_grad()    

            # statistic caluclation and output -------------------------

            if b % report_interval == 0:

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

                # run through all train samples again to accumulate layer-wise input distribution statistics (mean and variance) with fixed weights
                # these statistics are later used for the BatchNorm layers during inference
                for (bin_id, batch_ids) in binbatches:
                    input = train_bins[bin_id][batch_ids][:,:,:,0].squeeze(-1)      # size: (B x N x d)
                    model(input.transpose(1,2).to(model.base.device)).transpose(1,2)                      # size: (B x N x d)

                io.cprint('Accumulated BN Layer statistics (%ds)' % (time.time() - eval_bn_start))

            model.eval()

            test_start = time.time()

            test_loss = getTestLoss(model, test_samples, loss_function, edge_loss=edge_loss)
            
            checkpoint['test_loss'].append(test_loss)
            checkpoint['test_time'].append(time.time() - test_start)

            io.cprint('  TST | time: %5ds | loss: %.10f| av. dist. per neighbor: %.10f' %
                ( checkpoint['test_time'][-1], checkpoint['test_loss'][-1], np.sqrt(checkpoint['test_loss'][-1])))

            val_start = time.time()

            val_loss = getTestLoss(model, val_samples, loss_function, edge_loss=edge_loss)
            
            checkpoint['val_loss'].append(val_loss)
            checkpoint['val_time'].append(time.time() - val_start)

            io.cprint('  VAL | time: %5ds | loss: %.10f| av. dist. per neighbor: %.10f' %
                ( checkpoint['val_time'][-1], checkpoint['val_loss'][-1], np.sqrt(checkpoint['val_loss'][-1])))


        ####################
        # Scheduler Step
        ####################

        if not no_lr_schedule:
            scheduler.step()

        if epoch > 1 and dynamic_lr and sum(checkpoint['train_batch_lr_adjust'][-1]) > 0:
            io.cprint("----\n  dynamic lr adjust: %.10f" % (0.5 * (1 + sum(checkpoint['train_batch_lr_adjust'][-1]) / len(checkpoint['train_batch_lr_adjust'][-1]))))
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5 * (1 + sum(checkpoint['train_batch_lr_adjust'][-1]) / len(checkpoint['train_batch_lr_adjust'][-1]))


        # Save model and optimizer state ..
        checkpoint['model_state_dict'].append(copy.deepcopy(model.state_dict()))
        checkpoint['optimizer_state_dict'].append(copy.deepcopy(optimizer.state_dict()))

        torch.save(checkpoint, exp_dir + '/corrector_checkpoints.t7')


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

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_data_dict = getDataPaths()

    # parser.add_argument('--train_data', type=str, required=True, choices=list(train_data_dict.keys()), metavar='T', help='choose type of training data')
    parser.add_argument('--batch_size', type=int, default=4, metavar='B', help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=100, metavar='E', help='number of epochs to train ')
    parser.add_argument('--lr', type=float, default=0.0005, metavar='LR', help='learning rate')
    parser.add_argument('--seed', type=int, default=seed, metavar='S', help='random seed')
    parser.add_argument('--F', type=int, default=20, metavar='K', help='Num of nearest neighbors to use in the DGCNN model')
    # parser.add_argument('--checkpoint', type=str, default='', metavar='C',help='Path to previous model checkpoint')
    parser.add_argument('--model', type=str, default='mish', metavar='M', choices=['semseg', 'unet','unet_plus','cnet'], help='Model definition to use, [classic, unet, unet_plus, cnet, cnet_plus]')
    parser.add_argument('--model_cap', type=str, default='normal', metavar='M', choices=['normal', 'small', 'smaller'], help='Model capacity to use, [normal, small, smaller]')
    parser.add_argument('--optimizer', type=str, default='radam', metavar='O', choices=['radam', 'lookahead'], help='Optimizer to use, [radam, lookahead]')
    parser.add_argument('--activation', type=str, default='mish', metavar='A', choices=['mish', 'relu', 'swish', 'splash', 'squish'], help='Activation function to use, [mish, relu, swish, splash, squish]')
    parser.add_argument('--safe_descent', default=False, action='store_true')
    parser.add_argument('--dynamic_lr', default=False, action='store_true')
    parser.add_argument('--dropout', type=float, default=0, metavar='D', help='dropout probability')
    parser.add_argument('--edge_loss', default=False, action='store_true')
    parser.add_argument('--rotations', default=False, action='store_true')
    parser.add_argument('--stage_start', type=int, default=1, metavar='STS', help='which corrector stage (number of pipeline iterations) to train (start)')
    parser.add_argument('--stage_end', type=int, default=1, metavar='STE', help='which corrector stage (number of pipeline iterations) to train (end)')
    parser.add_argument('--use_gc', default=False, action='store_true')
    parser.add_argument('--diff_features_only', default=False, action='store_true')
    parser.add_argument('--no_batchnorm', default=False, action='store_true')
    parser.add_argument('--no_lr_schedule', default=False, action='store_true')
    parser.add_argument('--static_batchnorm', default=False, action='store_true')

    # arguments for train data generation
    # parser.add_argument('--train_size', required=True, type=int, default=1, metavar='TRN', help='number of train samples to generate')
    # parser.add_argument('--test_size', required=True, type=int, default=1, metavar='TST', help='number of test samples to generate')
    # parser.add_argument('--val_size', required=True, type=int, default=1, metavar='VAL', help='number of validation samples to generate')
    parser.add_argument('--dataset', required=True, type=str, default='multi_faces', metavar='D', choices=list(train_data_dict.keys()), help='Training dataset to use')
    # parser.add_argument('--h_min', type=int, default=3, metavar='hmin', help='minimum number of initially hidden points')
    # parser.add_argument('--h_max', type=int, default=16, metavar='hmax', help='maximum number of initially hidden points')
    # parser.add_argument('--n_min', type=int, default=3, metavar='hmin', help='minimum number of iterations per sample')
    # parser.add_argument('--n_max', type=int, default=11, metavar='hmax', help='maximum number of iterations per sample')
    # parser.add_argument('--predictor_checkpoint', type=str, metavar='p_cp', help='path to a trained predictor model checkpoint')
    # parser.add_argument('--detector_checkpoint', type=str, metavar='d_cp', help='path to a trained detector model checkpoint')
    # parser.add_argument('--corrector_checkpoint', type=str, metavar='c_cp', help='path to a trained corrector model checkpoint')
    # parser.add_argument('--max_pipeline_iterations', type=int, default=1, metavar='maxP', help='maximum number of times the pipeline is applied to generate corrector train data (IS OVERWRITTEN HERE)')

    args = parser.parse_args()

    assert args.stage_start <= args.stage_end

    exp_dir = None

    # prepare directories and IOStream -------------------------------------------------------------

    now = datetime.now()
    exp_dir = ('training/checkpoints/corrector_') + args.dataset + "_" + args.model + "_" + args.activation +  "_" + args.optimizer + ("_sd" if args.safe_descent else "") + "/" + now.strftime("%Y-%m-%d_%H%M%S")
    os.makedirs(exp_dir, exist_ok=True)
    
    io = IOStream(exp_dir + '/run.log')
    io.cprint('Command: ' + str(sys.argv))

    # copy model- and training files
    shutil.copy('training/train_corrector.py', exp_dir + '/train_corrector.py')
    shutil.copy('models/models.py', exp_dir + '/models.py')

    print("Reading Train- and Test Data..")

    start = time.time()

    train_data = None
    with open(train_data_dict[args.dataset]['base'] + '/' + args.dataset + '_train_data_corrector', 'rb') as file:
        train_data = torch.load(file, map_location=torch.device('cpu'))

    print ("  > done! (%.2fs)" % (time.time() - start))

    train(
        train_data = train_data,
        exp_dir = exp_dir,
        io = io,
        learning_rate = args.lr,
        rsize = args.F,
        epochs = args.epochs,
        # checkpoint_path = args.corrector_checkpoint,
        seed = args.seed,
        batch_size = args.batch_size,
        edge_loss=args.edge_loss,
        model_type = args.model,
        model_cap=args.model_cap,
        optimizer_type = args.optimizer,
        reset_optimizer = True,
        safe_descent=args.safe_descent,
        dynamic_lr=args.dynamic_lr,
        dropout=args.dropout,
        rotations=args.rotations,
        use_gc = args.use_gc,
        use_batch_norm=not args.no_batchnorm,
        batch_norm_affine = not args.static_batchnorm,
        no_lr_schedule = args.no_lr_schedule,
        diff_features_only = args.diff_features_only
    )


    # for s in range(args.stage_start, args.stage_end + 1):

    #     print("Generating Train-, Test- and Validation Data for STAGE %d.." % (s))

    #     args.max_pipeline_iterations = s
    #     args.n_min = max(args.n_min, s+1)
    #     args.n_max = max(args.n_max, args.n_min+1)
    #     if exp_dir != None:
    #         args.corrector_checkpoint = exp_dir + '/corrector_checkpoints.t7'

    #     train_data = generator.getCorrectorTrainData(args)
    #     with open(train_data_dict[args.dataset]['base'] + '/' + args.dataset + ('_train_data_corrector_%02d' % (args.max_pipeline_iterations)), 'wb') as file:
    #         torch.save(train_data, file)

    #     # train_data = generator.getSyntheticDeformTrainData(args)
    #     # with open(train_data_dict[args.dataset]['base'] + '/' + args.dataset + ('_train_data_corrector_synth_%02d' % (args.max_pipeline_iterations)), 'wb') as file:
    #     #     torch.save(train_data, file)

    #     # prepare directories and IOStream -------------------------------------------------------------

    #     now = datetime.now()
    #     exp_dir = ('training/checkpoints/corrector_%02d_' % (s)) + args.dataset + "_" + args.model + "_" + args.activation +  "_" + args.optimizer + ("_sd" if args.safe_descent else "") + "/" + now.strftime("%Y-%m-%d_%H%M%S")
    #     os.makedirs(exp_dir, exist_ok=True)
        
    #     io = IOStream(exp_dir + '/run.log')
    #     io.cprint('Command: ' + str(sys.argv))

    #     # copy model- and training files
    #     shutil.copy('training/train_corrector.py', exp_dir + '/train_corrector.py')
    #     shutil.copy('models/models.py', exp_dir + '/models.py')

    #     train(
    #         train_data = train_data,
    #         exp_dir = exp_dir,
    #         io = io,
    #         learning_rate = args.lr,
    #         rsize = args.F,
    #         epochs = args.epochs,
    #         checkpoint_path = args.corrector_checkpoint,
    #         seed = args.seed,
    #         batch_size = args.batch_size,
    #         edge_loss=args.edge_loss,
    #         model_type = args.model,
    #         model_cap=args.model_cap,
    #         optimizer_type = args.optimizer,
    #         reset_optimizer = True,
    #         safe_descent=args.safe_descent,
    #         dynamic_lr=args.dynamic_lr,
    #         dropout=args.dropout,
    #         rotations=args.rotations,
    #         use_gc = args.use_gc,
    #         use_batch_norm=not args.no_batchnorm,
    #         batch_norm_affine = not args.static_batchnorm,
    #         no_lr_schedule = args.no_lr_schedule,
    #         diff_features_only = args.diff_features_only
    #     )
