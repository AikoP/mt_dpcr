import sys
import os
sys.path.insert(1, os.getcwd())  # to load from any submodule in the repo

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from utils.schedules import MultiplicativeAnnealing

from matplotlib import rcParams
rcParams['font.family'] = 'Century Schoolbook'
rcParams['font.cursive'] = 'Century Schoolbook'

rcParams['text.usetex'] = True 
rcParams['text.latex.preamble'] = [r'\usepackage{fouriernc}']

def plotDetector(checkpoint, mode='acc', y_max = 0.01, loc = 'lower right'):

    rcParams['font.size'] = '19'
    
    train_batch_loss = checkpoint['train_batch_loss']
    train_batch_acc = checkpoint['train_batch_acc']
    train_loss = checkpoint['train_loss']
    train_acc = checkpoint['train_acc']
    train_C1_acc = checkpoint['train_C1_acc']
    
    test_loss = checkpoint['test_loss']
    test_acc = checkpoint['test_acc']
    val_loss = checkpoint['val_loss']
    val_acc = checkpoint['val_acc']
    val_C1_acc = checkpoint['val_C1_acc']
    
    print("train time: ", sum(checkpoint['train_time']) / 3600, "h")
    
    start = 1
    end = len(train_batch_acc)
    
    # -------------------------------------------------------------------------------------

    N = end - start + 1

    train_batch_acc_np = np.array(train_batch_acc)[0:end+1].reshape(-1)
    train_batch_loss_np = np.array(train_batch_loss)[0:end+1].reshape(-1)

    plt.figure(figsize=(16,9))
    ax = plt.axes()
    
    plt.setp(ax.spines.values(), color=3 * [0.5])
    ax.set_facecolor(3 * [0.99])
    ax.tick_params(axis='x', colors=3 * [0.3])
    ax.tick_params(axis='y', colors=3 * [0.3])
    
    ax.set_xlim(0, end+0.5)

    x = np.arange(start, end+1, 1)
    x_cont = np.linspace(0, end, train_batch_acc_np.shape[0])

    y_ticks = None

    if mode == 'acc':
        
        step = 0.01
        y_ticks = np.arange(0.8, 1.0 + step, step)
        
        ax.scatter(x_cont, train_batch_acc_np, label='Train Batch Acc', color = (0.0,0.0,0.0,0.02), s=20)

        ax.plot(x, train_acc[start-1:end], label='Train Acc', marker='o', color='steelblue')
        #ax.plot(x, train_C0_acc[start-1:end], label='C0 Acc', marker='o', color='seagreen')
        ax.plot(x, train_C1_acc[start-1:end], label='C1 Acc', marker='o', color='orange')

        ax.plot(x, test_acc[start-1:end], label='Test Acc', marker='o', linestyle=':', color='steelblue')

        ax.plot(x, val_acc[start-1:end], label='Validation Acc', marker='o', linestyle='--', color='steelblue')
        #ax.plot(x, val_C0_acc[start-1:end], label='Val C0 Acc', marker='o', linestyle='--', color='seagreen')
        ax.plot(x, val_C1_acc[start-1:end], label='Val C1 Acc', marker='o', linestyle='--', color='orange')
        
    elif mode == 'loss':
        
        step = y_max / 10
        y_ticks = np.arange(0, y_max + step, step)
        
        ax.scatter(x_cont, train_batch_loss_np, label='Train Batch Loss', color = (0.0,0.0,0.0,0.02), s=20)
        ax.plot(x, train_loss[start-1:end], label='Train Loss', marker='o', color='steelblue')
        #ax.plot(x, test_loss[start-1:end], label='Test Loss', marker='o', linestyle=':', color='steelblue')
        ax.plot(x, val_loss[start-1:end], label='Validation Loss', marker='o', color='orange')
        
    ax.set_ylim(y_ticks[0], y_ticks[-1])

    ax.legend(loc = loc)
    ax.set_axisbelow(True)
    ax.grid(color=3 * [0.88])

def plotCompare(checkpoints, mode='acc', y_lim_loss = 0.2, y_lim_acc = 90, loc_loss = 'upper right', loc_acc = 'lower right', max_n = -1, labels = None):

    if labels == None:
        labels = ["Model %02d" % (i+1) for i in range(len(checkpoints))]

    assert len(labels) == len(checkpoints)

    rcParams['font.size'] = '19'
    
    n = max_n

    for i, cp in enumerate(checkpoints):
        print("train time checkpoint #%d: %.2fh" % (i+1, sum(cp['train_time']) / 3600))
        if (len(cp['lr']) < n) or n < 0:
            n = len(cp['lr'])
    
    # -------------------------------------------------------------------------------------

    plt.figure(figsize=(16,9))
    ax = plt.axes()
    
    plt.setp(ax.spines.values(), color=3 * [0.5])
    ax.set_facecolor(3 * [0.99])
    ax.tick_params(axis='x', colors=3 * [0.3])
    ax.tick_params(axis='y', colors=3 * [0.3])
    
    ax.set_xlim(0, n+0.5)

    x = 1 + np.arange(n)
    y_ticks = None

    colors = ['steelblue', 'orange', 'seagreen']

    if mode == 'acc':

        step = (100 - y_lim_acc) / 10
        y_ticks = np.arange(y_lim_acc, 100 + step, step)

        for i, cp in enumerate(checkpoints):
            ax.plot(x, 100 * np.array(cp['val_acc'][:n]), label=labels[i] + ', Val Acc', marker='o', color=colors[i%len(colors)])
            ax.plot(x, 100 * np.array(cp['val_C1_acc'][:n]), label=labels[i] + r', Val $C_1$ Recall', marker='o', color=colors[i%len(colors)], linestyle='--')  

        ax.yaxis.set_major_formatter(ticker.PercentFormatter())

        ax.legend(loc = loc_acc, ncol=len(checkpoints))
        
    elif mode == 'loss' or mode == 'both':

        ax.set_ylabel('Loss', labelpad =20)
        
        step = y_lim_loss / 10
        y_ticks = np.arange(0, y_lim_loss + step, step)

        for i, cp in enumerate(checkpoints):
            ax.plot(x, cp['train_loss'][:n], label=labels[i] + ', Train', marker='o', color=colors[i%len(colors)])
            ax.plot(x, cp['val_loss'][:n], label=labels[i] + ', Validation', marker='o', color=colors[i%len(colors)], linestyle='--')

        if mode == 'loss':
            ax.legend(loc = loc_loss, ncol=len(checkpoints))


    if mode == 'both':

        ax2 = ax.twinx()

        for i, cp in enumerate(checkpoints):
            ax2.plot(x, 100 * np.array(cp['val_acc'][:n]), label=labels[i] + ', Val Overall', marker='o', color=colors[i%len(colors)], linestyle=':')
            ax2.plot(x, 100 * np.array(cp['val_C1_acc'][:n]), label=labels[i] + r', Val $C_1$', marker='o', color=colors[i%len(colors)], linestyle=':', alpha = 0.5)  

        ax2.yaxis.set_major_formatter(ticker.PercentFormatter())
        ax2.set_ylim(y_lim_acc, 100.0)
        ax2.tick_params(axis='y', colors=3 * [0.3])

        if mode == 'both':
            ax2.legend(loc = 'lower center', bbox_to_anchor=(0.5, 1.05), ncol=len(checkpoints), title=r'Overall Accuracy and $C_1$ Recall')
        else:
            ax2.legend(loc = loc_acc)

        ax2.set_ylabel('Accuracy', labelpad =20)

    
    
    ax.set_ylim(y_ticks[0], y_ticks[-1])

    if mode == 'both':
        ax.legend(loc = 'upper center', bbox_to_anchor=(0.5, -0.05), ncol=len(checkpoints), title="Train and Validation Loss")

    ax.set_axisbelow(True)
    ax.grid(color=3 * [0.88])


def plotLearningRate(checkpoint, y_max = 0.01, plot_orig=True, plot_lossreduction=True, loc='lower right'):

    rcParams['font.size'] = '19'
    
    start = 1
    end = len(checkpoint["lr"])
    step = y_max / 10
    y_ticks = np.arange(0, y_max + step, step)

    start_learning_rate = checkpoint['train_settings'][-1]['learning_rate']
    max_epochs = checkpoint['train_settings'][-1]['epochs']
    lr_schedule = MultiplicativeAnnealing(max_epochs)
    scheduled_lr = start_learning_rate * np.ones(end)
    for i in range(end):
        scheduled_lr[i+1:] *= lr_schedule(i+1)

    loss_reductions = [sum(batch_loss_reductions) / len(batch_loss_reductions) for batch_loss_reductions in checkpoint['train_batch_loss_reduction']]
    
    # -------------------------------------------------------------------------------------

    N = end - start + 1

    plt.figure(figsize=(16,9))
    ax = plt.axes()
    
    plt.setp(ax.spines.values(), color=3 * [0.5])
    ax.set_facecolor(3 * [0.99])
    ax.tick_params(axis='x', colors=3 * [0.3])
    ax.tick_params(axis='y', colors=3 * [0.3])
    
    ax.set_xlim(0.5, end+0.5)

    x = np.arange(start, end+1, 1)
    
    ax.plot(x, checkpoint["lr"][start-1:end], label='Actual LR', marker='o', linestyle='--', color='steelblue')

    if plot_orig:
        ax.plot(x, scheduled_lr, label='Scheduled LR', marker='o', linestyle='--', color='orange')
        
    if plot_lossreduction:
        ax2 = ax.twinx()
        ax2.plot(x, loss_reductions, label='Av. Loss Reduction', marker='o', linestyle=':', color='seagreen')
        ax2.yaxis.set_major_formatter(ticker.PercentFormatter())
        ax2.set_ylim(0.0, 2.0)
        ax2.tick_params(axis='y', colors=3 * [0.3])
        ax2.legend(loc = 'upper right')


    ax.set_ylim(y_ticks[0], y_ticks[-1])

    ax.legend(loc = loc)
    ax.set_axisbelow(True)
    ax.grid(color=3 * [0.88])


def plotPredictor(checkpoint, y_max = 0.0005):

    rcParams['font.size'] = '19'
    
    train_batch_loss = checkpoint['train_batch_loss']
    train_epoch_loss = checkpoint['train_loss']
    
    test_loss = checkpoint['test_loss']
    val_loss = checkpoint['val_loss']
    
    print("train time: ", sum(checkpoint['train_time']) / 3600, "h")
    
    start = 1
    end = len(train_batch_loss)
    # end = 1
    step = y_max / 10
    y_ticks = np.arange(0, y_max + step, step)

    # -------------------------------------------------------------------------------------

    N = end - start + 1

    train_batch_loss_np = np.array(train_batch_loss)[0:end+1].reshape(-1)

    plt.figure(figsize=(16,9))
    
    ax = plt.axes()
    
    plt.setp(ax.spines.values(), color=3 * [0.5])
    ax.set_facecolor(3 * [0.99])
    ax.tick_params(axis='x', colors=3 * [0.3])
    ax.tick_params(axis='y', colors=3 * [0.3])
    
    ax.set_ylim(y_ticks[0], y_ticks[-1])
    ax.set_xlim(0, end+0.5)

    x = np.arange(start, end+1, 1)
    x_cont = np.linspace(start-1, end, train_batch_loss_np.shape[0])

    ax.scatter(x_cont, train_batch_loss_np, label='Train Batch Loss', color = (0.0,0.0,0.0,0.04), s=20)
    ax.plot(x, train_epoch_loss[start-1:end], label='Train Loss', marker='o')
    ax.plot(x, test_loss[start-1:end], label='Test Loss', marker='o')
    ax.plot(x, val_loss[start-1:end], label='Validation Loss', marker='o')

    #plt.hlines(y_ticks, -1, end+1, color='lightgrey', linewidths=1, linestyles='dashed')
    #plt.xticks(np.arange(start,end+1).astype(int))
    #plt.yticks(y_ticks)

    ax.legend(loc = 'upper right')
    ax.set_axisbelow(True)
    ax.grid(color=3 * [0.88])