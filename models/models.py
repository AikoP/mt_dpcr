import torch
import torch.nn as nn
import torch.nn.functional as F

# Swish activation function
class Swish(torch.nn.Module):

    def __init__(self, beta = 0):
        super().__init__()
        self.beta = torch.nn.Parameter(torch.tensor([beta], dtype=torch.float), requires_grad=True)

    def forward(self, x):
        return x * torch.sigmoid(0.25 + 1.5 * torch.sigmoid(self.beta) * x)

# Mish activation function
class Mish(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x *( torch.tanh(F.softplus(x)))

# Squish activation function
class Squish(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.tanh(x) * torch.sqrt(F.softplus(x))

#Splash activation function (learnable)
class Splash(torch.nn.Module):

    def __init__(self, S = 4, init_shape='ReLU', ):
        super(Splash,self).__init__()

        self.S = S
        self.b = torch.nn.Parameter(torch.linspace(0, 3, steps=S), requires_grad=False)

        self.a_pos = torch.nn.Parameter(torch.zeros(S), requires_grad=True)
        # self.a_pos[0] = 1
        self.a_neg = torch.nn.Parameter(torch.zeros(S), requires_grad=True)
        # self.a_pos[0] = -0.07

        # if init_shape == 'ReLU':
        #     self.a_pos = torch.nn.Parameter(torch.zeros(S), requires_grad=True)
        #     self.a_pos[0] = 1
        #     self.a_neg = torch.nn.Parameter(torch.zeros(S), requires_grad=True)
        #     self.a_pos[0] = -0.07
        # else:
        #     self.a_pos = torch.nn.Parameter(torch.randn(S), requires_grad=True)
        #     self.a_neg = torch.nn.Parameter(torch.randn(S), requires_grad=True)

        # self.a_pos.requires_grad = True
        # self.a_neg.requires_grad = True

    def resu(self, x, b):
        return torch.tanh(x-b)*torch.sqrt(torch.square(x-b))

    def forward(self, x):

        b = torch.ones(x.size(), device=x.device).unsqueeze(-1) * self.b
        x_ex = x.unsqueeze(-1).repeat_interleave(self.S, dim=-1)
        z = torch.zeros(b.size(), device=x.device)
        
        h = torch.sum(torch.max(x_ex-b,z) * self.a_pos + torch.max(-x_ex-b,z) * self.a_neg, dim=-1)

        return h

def getModel(args = {},
        model_type = 'cnet_plus',
        model_cap = 'normal',
    ):

    args['model_type'] = args['model_type'] if 'model_type' in args else model_type
    args['model_cap'] = args['model_cap'] if 'model_cap' in args else model_cap

    model = None

    if args['model_type'] == 'semseg':
        model = Model_SemSeg(args)
    elif args['model_type'] == 'unet':
        model = Model_UNet(args)
    elif args['model_type'] == 'unet_plus':
        model = Model_UNetPlus(args)
    elif args['model_type'] == 'cnet' or args['model_type'] == 'cnet_plus':
        if args['model_cap'] == 'small':
            model = Model_CNet_small(args)
        elif args['model_cap'] == 'smaller':
            model = Model_CNet_smaller(args)
        else:
            model = Model_CNet(args)
    else:
        raise Exception('model type %s not supported!' % (args['model_type']))

    return model

def loadModel(checkpoint_path, device = torch.device('cpu'), checkpoint = -1):

    model_checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

    model = getModel(model_checkpoint['train_settings'][-1]['model_args'])

    state_dict = model_checkpoint['model_state_dict'][checkpoint]
    state_dict['base'] = nn.Parameter(torch.empty(0))   # backwards compatibility

    model.load_state_dict(state_dict)
    model.eval()

    return model.to(device)

def getConvLayer(in_channels, out_channels, args = {},
        d = 1,
        batch_norm = True,
        batch_norm_momentum = None,
        batch_norm_affine = False
    ):

    # insert default args
    args['batch_norm'] = args['batch_norm'] if 'batch_norm' in args else batch_norm
    args['batch_norm_momentum'] = args['batch_norm_momentum'] if 'batch_norm_momentum' in args else batch_norm_momentum
    args['batch_norm_affine'] = args['batch_norm_affine'] if 'batch_norm_affine' in args else batch_norm_affine

    activation = getActivation(args)

    if args['batch_norm']:
        if d == 1:
            return nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels, momentum=args['batch_norm_momentum'], affine=args['batch_norm_affine']),
                activation
            )
        elif d == 2:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels, momentum=args['batch_norm_momentum'], affine=args['batch_norm_affine']),
                activation
            )
    else:
        if d == 1:
            return nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                activation
            )
        elif d == 2:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                activation
            )

def getActivation(args={},
        activation_type = 'mish',
        activation_args = {}
    ):

    args['activation_type'] = args['activation_type'] if 'activation_type' in args else activation_type
    args['activation_args'] = args['activation_args'] if 'activation_args' in args else activation_args

    if args['activation_type'] == 'mish':
        return Mish()
    elif args['activation_type'] == 'relu':
        leaky = args['activation_args']['leaky'] if 'leaky' in args['activation_args'] else True
        if leaky:
            slope = args['activation_args']['slope'] if 'slope' in args['activation_args'] else 0.2
            return nn.LeakyReLU(negative_slope=slope)
        else:
            return nn.ReLU()
    elif args['activation_type'] == 'swish':
        if 'beta' in args['activation_args']:
            return Swish(beta = args['activation_args']['beta'])
        else:
            return Swish()
    elif args['activation_type'] == 'splash':
        return Splash()
    elif args['activation_type'] == 'squish':
        return Squish()

    return None

def knn(x, k):

    """
        x: (b x d x n) tensor
        k: integer
    """

    # # compute pairwise distances
    # inner = torch.matmul(x.transpose(2, 1), 2.0 * x)        # (b x n x n)
    # xx = torch.sum(torch.square(x), dim=1, keepdim=True)    # (b x 1 x n)
    # pairwise_distance = xx - inner + xx.transpose(2, 1)     # (b x n x n)

    # # print ("pairwise_distance.size():", pairwise_distance.size())
 
    # return pairwise_distance.topk(k=k+1, dim=-1, largest=False)[1][:,:,1:]   # (b x n x k)


    # dynamic tensor splitting? ---------------------------------------------------------------------

    b = x.size(0)
    d = x.size(1)
    n = x.size(2)

    GPU_FREE_MEM = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()

    # print ("free memory: %.3f GB" % (GPU_MAX_MEM / 10**9))

    q = int(((0.95 * GPU_FREE_MEM) / (4*b) - n*d)/(n*d + d))

    # print ("q =", q)

    if (n <= q):

        # compute pairwise distances
        inner = torch.matmul(x.transpose(2, 1), 2.0 * x)        # (b x n x n)
        xx = torch.sum(torch.square(x), dim=1, keepdim=True)    # (b x 1 x n)
        pairwise_distance = xx.transpose(2, 1) - inner + xx     # (b x n x n)

        return pairwise_distance.topk(k=k+1, dim=-1, largest=False)[1][:,:,1:]   # (b x n x k)

    else:

        topk = torch.empty((b, n, k), dtype=torch.long, device=x.device) # (b x n x k)

        for i in range(0, n, q):

            # compute pairwise distances
            inner = torch.matmul(x[:,:,i:i+q].transpose(2, 1), 2.0 * x)         # (b x q x n)
            # yy = torch.sum(torch.square(x[:,:,i:i+q]), dim=1, keepdim=True)   # (b x 1 x q)
            xx = torch.sum(torch.square(x), dim=1, keepdim=True)                # (b x 1 x n)
            pairwise_distance = xx[:,:,i:i+q].transpose(2, 1) - inner + xx                 # (b x q x n)

            # topk[:, i:i+q, :] = pairwise_distance.topk(k=k+1, dim=-1, largest=False)[1][:,:,1:].to(torch.device('cpu'))
            topk[:, i:i+q, :] = pairwise_distance.topk(k=k+1, dim=-1, largest=False)[1][:,:,1:]     # (b x q x k)

        # return topk.to(x.device)
        return topk


def get_graph_feature(x, rsize = 20, diff_features_only = False):

    """
        x: (b x d x n) tensor
        rsize: integer 
    """
    
    knn_idx = knn(x, k = rsize).unsqueeze(1).expand(-1, x.size(1), -1, -1)   # (b x n x k) -> (b x 1 x n x k) -> (b x d x n x k)

    x = x.unsqueeze(-1).expand(-1, -1, -1, rsize)   # (b x d x n) -> (b x d x n x 1) -> (b x d x n x k)
    f = x.gather(2, knn_idx)                        # (b x d x n x k)
    
    if diff_features_only:
        return f - x                        # (b x d x n x k)
    else:
        return torch.cat([f-x, x], dim=1)   # (b x 2*d x n x k)
        

    # below the old version (slower) ---------------------------------------

    # b, d, n = x.size()

    # idx_base = torch.arange(b, device=x.device).view(-1, 1, 1) * n   # (b x 1 x 1)

    # idx = (knn(x, k = rsize) + idx_base).view(-1)   # (b x n x k) -> (b * n * k)

    # x = x.transpose(2, 1).contiguous()          # (b, d, n)  -> (b, n, d)
    # feature = x.view(b*n, -1)[idx, :]           # (b, n, d)  -> (b * n, d) -> (b * n * k, d)
    # feature = feature.view(b, n, rsize, d)          # (b * n * k, d) -> (b, n, k, d)
    # x = x.view(b, n, 1, d).repeat(1, 1, rsize, 1)   # (b, n, d) -> (b, n, 1, d) -> (b, n, k, d)
    
    # feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2)  # (b, n, k, 2 * d) -> (b, 2 * d, n, k)
  
    # return feature      # (b, 2 * d, n, k)

# This base module implements the batch batch normalization phase, that tracks running means and variances
class BaseModule(nn.Module):

    def __init__(self, args,
            input_channels = 3,
            rsize = 10,
            output_channels = 2,
            dropout = 0,
            diff_features_only = False
        ):

        super(BaseModule, self).__init__()

        self.base = nn.Parameter(torch.empty(0))

        # insert defaults if missing
        args['rsize'] = args['rsize'] if 'rsize' in args else rsize
        args['input_channels'] = args['input_channels'] if 'input_channels' in args else input_channels
        args['output_channels'] = args['output_channels'] if 'output_channels' in args else output_channels
        args['dropout'] = args['dropout'] if 'dropout' in args else dropout
        args['diff_features_only'] = args['diff_features_only'] if 'diff_features_only' in args else diff_features_only

        self.args = args

        self.rsize = args['rsize']
        self.input_channels = args['input_channels']
        self.output_channels = args['output_channels']
        self.dropout = args['dropout']
        self.diff_features_only = args['diff_features_only']

    def eval_bn(self):
        """
            set this mode to compute averages of mean and variance for the BatchNorm layers
            run one epoch without backprops to accumulate average values for mean and variance from batches with fixed weights
        """

        for module in self.modules():
            if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
                module.track_running_stats = True
                module.reset_running_stats()

    def eval(self):
        """
            reset the batch norm calculation without resetting the accumulated values for mean and variance
        """

        super().eval()

        for module in self.modules():
            if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
                module.track_running_stats = False

        

    def train(self, mode = True):
        """
            reset the batch norm calculation without resetting the accumulated values for mean and variance (regardless of mode)
        """

        super().train(mode)

        for module in self.modules():
            if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
                module.track_running_stats = False

class Model_SemSeg(BaseModule):

    def __init__(self, args, emb_dims = 1024):

        super(Model_SemSeg, self).__init__(args)

        self.emb_dims = args['emb_dims'] if 'emb_dims' in args else emb_dims

        self.conv1 = getConvLayer(self.input_channels*2, 64, args=args, d=2)
        self.conv2 = getConvLayer(64, 64, args=args, d=2)

        self.conv3 = getConvLayer(64*2, 64, args=args, d=2)
        self.conv4 = getConvLayer(64, 64, args=args, d=2)

        self.conv5 = getConvLayer(64*2, 64, args=args, d=2)

        self.conv6 = getConvLayer(64*3, self.emb_dims, args=args, d=1)
        self.conv7 = getConvLayer(self.emb_dims+64*3, 512, args=args, d=1)
        self.conv8 = getConvLayer(512, 256, args=args, d=1)

        self.conv9 = nn.Conv1d(256, self.output_channels, kernel_size=1, bias=False)

    def forward(self, x):

        # batch_size = x.size(0)
        num_points = x.size(2)

        x = get_graph_feature(x, rsize=self.rsize)  # (batch_size, d, num_points) -> (batch_size, 2*d, num_points, k)
        x = self.conv1(x)                               # (batch_size, 2*d, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                               # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]            # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)+

        x = get_graph_feature(x1, rsize=self.rsize)             # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                               # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)                               # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]            # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, rsize=self.rsize)             # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)                               # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]            # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = torch.cat((x1, x2, x3), dim=1)              # (batch_size, 64*3, num_points)

        x = self.conv6(x)                               # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x = x.max(dim=-1, keepdim=True)[0]              # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)

        x = x.repeat(1, 1, num_points)                  # (batch_size, 1024, num_points)
        x = torch.cat((x, x1, x2, x3), dim=1)           # (batch_size, 1024+64*3, num_points)

        x = self.conv7(x)                               # (batch_size, 1024+64*3, num_points) -> (batch_size, 512, num_points)
        x = self.conv8(x)                               # (batch_size, 512, num_points) -> (batch_size, 256, num_points)
        x = self.conv9(x)                               # (batch_size, 256, num_points) -> (batch_size, 2, num_points)

        return x

class Model_UNet(BaseModule):

    def __init__(self, args):

        super(Model_UNet, self).__init__(args)

        self.conv1 = getConvLayer(self.input_channels*2, 64, args=args, d=2)
        self.conv2 = getConvLayer(64*2, 64, args=args, d=2)
        self.conv3 = getConvLayer(64*2, 128, args=args, d=2)

        self.conv4 = getConvLayer(128, 128, args=args, d=1)

        self.conv5 = getConvLayer(256, 128, args=args, d=1)
        self.conv6 = getConvLayer(128, 64, args=args, d=1)
        self.conv7 = getConvLayer(128, 64, args=args, d=1)
        self.conv8 = getConvLayer(128, 64, args=args, d=1)

        self.conv9 = nn.Conv1d(64, self.output_channels, kernel_size=1, bias=False)

    def forward(self, x):

        x = get_graph_feature(x, rsize=self.rsize)  # (batch_size, d, num_points) -> (batch_size, 2*d, num_points, k)
        x = self.conv1(x)                               # (batch_size, 2*d, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]            # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, rsize=self.rsize)             # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv2(x)                               # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]            # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, rsize=self.rsize)             # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                               # (batch_size, 64*2, num_points, k) -> (batch_size, 128, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]            # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = self.conv4(x3)                              # (batch_size, 128, num_points) -> (batch_size, 128, num_points)

        x = torch.cat((x, x3), 1)                       # (batch_size, 128, num_points) -> (batch_size, 256, num_points)
        x = self.conv5(x)                               # (batch_size, 256, num_points) -> (batch_size, 128, num_points)
        x = self.conv6(x)                               # (batch_size, 128, num_points) -> (batch_size, 64, num_points)

        x = torch.cat((x, x2), 1)                       # (batch_size, 64, num_points) -> (batch_size, 128, num_points)
        x = self.conv7(x)                               # (batch_size, 128, num_points) -> (batch_size, 64, num_points)

        x = torch.cat((x, x1), 1)                       # (batch_size, 64, num_points) -> (batch_size, 128, num_points)
        x = self.conv8(x)                               # (batch_size, 128, num_points) -> (batch_size, 64, num_points)

        x = self.conv9(x)                               # (batch_size, 64, num_points) -> (batch_size, out_channels, num_points)

        return x

class Model_UNetPlus(BaseModule):

    def __init__(self, args):

        super(Model_UNetPlus, self).__init__(args)

        self.conv11 = getConvLayer(self.input_channels*2, 64, args=args, d=2)
        self.conv12 = getConvLayer(64*2, 64, args=args, d=2)
        self.conv13 = getConvLayer(64*2, 128, args=args, d=2)
        self.conv14 = getConvLayer(128*2, 128, args=args, d=2)

        self.conv21 = getConvLayer(128, 64, args=args, d=1)
        self.conv221 = getConvLayer(192, 128, args=args, d=1)
        self.conv222 = getConvLayer(128, 64, args=args, d=1)
        self.conv231 = getConvLayer(256, 128, args=args, d=1)
        self.conv232 = getConvLayer(128, 64, args=args, d=1)

        self.conv31 = getConvLayer(128, 64, args=args, d=1)
        self.conv32 = getConvLayer(128, 64, args=args, d=1)

        self.conv411 = getConvLayer(128, 64, args=args, d=1)
        self.conv412 = nn.Conv1d(64, self.output_channels, kernel_size=1, bias=False)

    def forward(self, x):

        x = get_graph_feature(x, rsize=self.rsize)  # (batch_size, d, num_points) -> (batch_size, 2*d, num_points, k)
        x = self.conv11(x)                              # (batch_size, 2*d, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]            # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, rsize=self.rsize)             # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv12(x)                              # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]            # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, rsize=self.rsize)             # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv13(x)                              # (batch_size, 64*2, num_points, k) -> (batch_size, 128, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]            # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = get_graph_feature(x3, rsize=self.rsize)             # (batch_size, 128, num_points) -> (batch_size, 128*2, num_points, k)
        x = self.conv14(x)                              # (batch_size, 128*2, num_points, k) -> (batch_size, 128, num_points, k)
        x4 = x.max(dim=-1, keepdim=False)[0]            # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x1 = self.conv21(torch.cat((x1, x2), 1))        # (batch_size, 128, num_points) -> (batch_size, 64, num_points)

        x2 = self.conv221(torch.cat((x2, x3), 1))       # (batch_size, 192, num_points) -> (batch_size, 128, num_points)
        x2 = self.conv222(x2)                           # (batch_size, 128, num_points) -> (batch_size, 64, num_points)

        x3 = self.conv231(torch.cat((x3, x4), 1))       # (batch_size, 256, num_points) -> (batch_size, 128, num_points)
        x3 = self.conv232(x3)                           # (batch_size, 128, num_points) -> (batch_size, 64, num_points)

        x1 = self.conv31(torch.cat((x1, x2), 1))        # (batch_size, 128, num_points) -> (batch_size, 64, num_points)
        x2 = self.conv32(torch.cat((x2, x3), 1))        # (batch_size, 128, num_points) -> (batch_size, 64, num_points)

        x1 = self.conv411(torch.cat((x1, x2), 1))       # (batch_size, 128, num_points) -> (batch_size, 64, num_points)

        x = self.conv412(x1)                            # (batch_size, 64, num_points) -> (batch_size, out_channels, num_points)

        return x

class Model_CNet(BaseModule):

    def __init__(self, args):

        super(Model_CNet, self).__init__(args)

        self.dpR1 = nn.Dropout2d(p=self.dropout)
        self.convR1C1 = getConvLayer(self.input_channels, 64, args=args, d=2) if self.diff_features_only else getConvLayer(self.input_channels*2, 64, args=args, d=2)
        self.convR1C2 = getConvLayer(64, 64, args=args, d=2)

        self.dpR2 = nn.Dropout2d(p=self.dropout)
        self.convR2C1 = getConvLayer(64, 64, args=args, d=2) if self.diff_features_only else getConvLayer(64*2, 64, args=args, d=2)
        self.convR2C2 = getConvLayer(64, 64, args=args, d=2)

        self.dpR3 = nn.Dropout2d(p=self.dropout)
        self.convR3C1 = getConvLayer(64, 64, args=args, d=2) if self.diff_features_only else getConvLayer(64*2, 64, args=args, d=2)
        self.convR3C2 = getConvLayer(64, 64, args=args, d=2)

        self.dpR4 = nn.Dropout2d(p=self.dropout)
        self.convR4C1 = getConvLayer(64, 64, args=args, d=2) if self.diff_features_only else getConvLayer(64*2, 64, args=args, d=2)
        self.convR4C2 = getConvLayer(64, 64, args=args, d=2)

        self.convC1 = getConvLayer(256, 1024, args=args, d=1)
        self.convC2 = getConvLayer(1024, 448, args=args, d=1)
        self.convC3 = getConvLayer(512, 256, args=args, d=1)
        self.convC4 = getConvLayer(256, 96, args=args, d=1)
        self.convC5 = getConvLayer(128, 64, args=args, d=1)

        self.dp = nn.Dropout(p=self.dropout)

        self.convG11 = getConvLayer(1024, 64, args=args, d=1)
        self.convG12 = nn.Sequential(nn.Linear(128, 64, bias=False), getActivation(args=args))

        self.convG21 = getConvLayer(256, 32, args=args, d=1)
        self.convG22 = nn.Sequential(nn.Linear(64, 32, bias=False), getActivation(args=args))

        self.convC6 = nn.Conv1d(64, self.output_channels, kernel_size=1, bias=False)

    def forward(self, x):

        b = x.size(0)   # batch size
        # d = x.size(1)   # dimensionality of input
        n = x.size(2)   # number of points

        # EdgeConv ------------------------------------

        x = get_graph_feature(x, rsize=self.rsize, diff_features_only=self.diff_features_only)              # (b, d, n) -> (b, 2*d, n, F)
        x = self.dpR1(x)                                # (b, 2*d, n) -> (b, 2*d, n, F)
        x = self.convR1C1(x)                            # (b, 2*d, n, F) -> (b, 64, n, F)
        x = self.convR1C2(x)                            # (b, 64, n, F) -> (b, 64, n, F)
        r1 = x.max(dim=-1, keepdim=False)[0]            # (b, 64, n, F) -> (b, 64, n)

        # ---------------------------------------------

        # EdgeConv ------------------------------------

        x = get_graph_feature(r1, rsize=self.rsize, diff_features_only=self.diff_features_only)             # (b, 64, n) -> (b, 64*2, n, F)
        x = self.dpR2(x)                                # (b, 2*d, n) -> (b, 2*d, n, F)
        x = self.convR2C1(x)                            # (b, 64*2, n, F) -> (b, 64, n, F)
        x = self.convR2C2(x)                            # (b, 64, n, F) -> (b, 64, n, F)
        r2 = x.max(dim=-1, keepdim=False)[0]            # (b, 64, n, F) -> (b, 64, n)

        # ---------------------------------------------

        # EdgeConv ------------------------------------

        x = get_graph_feature(r2, rsize=self.rsize, diff_features_only=self.diff_features_only)             # (b, 64, n) -> (b, 64*2, n, F)
        x = self.dpR3(x)                                # (b, 2*d, n) -> (b, 2*d, n, F)
        x = self.convR3C1(x)                            # (b, 64*2, n, F) -> (b, 64, n, F)
        x = self.convR3C2(x)                            # (b, 64, n, F) -> (b, 64, n, F)
        r3 = x.max(dim=-1, keepdim=False)[0]            # (b, 64, n, F) -> (b, 64, n)

        # ---------------------------------------------

        # EdgeConv ------------------------------------

        x = get_graph_feature(r3, rsize=self.rsize, diff_features_only=self.diff_features_only)             # (b, 64, n) -> (b, 64*2, n, F)
        x = self.dpR4(x)                                # (b, 2*d, n) -> (b, 2*d, n, F)
        x = self.convR4C1(x)                            # (b, 64*2, n, F) -> (b, 64, n, F)
        x = self.convR4C2(x)                            # (b, 64, n, F) -> (b, 64, n, F)
        r4 = x.max(dim=-1, keepdim=False)[0]            # (b, 64, n, F) -> (b, 64, n)

        # ---------------------------------------------


        x = torch.cat((r1, r2, r3, r4), 1)              # (b, 256, n)
        x = self.convC1(x)                               # (b, 256, n) -> (b, 1024, n)


        # GlobConv ------------------------------------

        # GlobExt ----------
        

        x_glob = self.convG11(x)                        # (b, 1024, n) -> (b, 64, n)
        x_max = F.adaptive_max_pool1d(x_glob, 1)        # (b, 64, n) -> (b, 64, 1)
        x_max = x_max.view(b, -1)                       # (b, 64, 1) -> (b, 64)
        x_avg = F.adaptive_avg_pool1d(x_glob, 1)        # (b, 64, n) -> (b, 64, 1)
        x_avg = x_avg.view(b, -1)                       # (b, 64, 1) -> (b, 64)
        x_glob = torch.cat((x_max, x_avg), 1)           # (b, 128)
        x_glob = self.convG12(x_glob).unsqueeze(-1)     # (b, 128) -> (b, 64, 1)

        # ------------------

        x = self.convC2(x)                              # (b, 1024, n) -> (b, 384, n)

        x = torch.cat((x, x_glob.repeat(1, 1, n)), 1)   # (b, 448, n) -> (b, 512, n)

        x = self.convC3(x)                              # (b, 512, n) -> (b, 256, n)

        # ---------------------------------------------

        # GlobConv ------------------------------------

        # GlobExt ----------
        
        x_glob = self.convG21(x)                        # (b, 256, n) -> (b, 32, n)
        x_max = F.adaptive_max_pool1d(x_glob, 1)        # (b, 32, n) -> (b, 32, 1)
        x_max = x_max.view(b, -1)                       # (b, 32, 1) -> (b, 32)
        x_avg = F.adaptive_avg_pool1d(x_glob, 1)        # (b, 32, n) -> (b, 32, 1)
        x_avg = x_avg.view(b, -1)                       # (b, 32, 1) -> (b, 32)
        x_glob = torch.cat((x_max, x_avg), 1)           # (b, 64)
        x_glob = self.convG22(x_glob).unsqueeze(-1)     # (b, 64) -> (b, 32, 1)

        # ------------------

        x = self.convC4(x)                               # (b, 256, n) -> (b, 96, n)
        
        x = torch.cat((x, x_glob.repeat(1, 1, n)), 1)   # (b, 96, n) -> (b, 128, n)

        x = self.convC5(x)                               # (b, 128, n) -> (b, 64, n)

        # ---------------------------------------------


        x = self.dp(x)                                  # (b, 64, n) -> (b, 64, n))

        x = self.convC6(x)                              # (b, 64, n) -> (b, out_channels, n)
        

        return x

class Model_CNet_small(BaseModule):

    def __init__(self, args):

        super(Model_CNet_small, self).__init__(args)

        self.dpR1 = nn.Dropout2d(p=self.dropout)
        self.convR1C1 = getConvLayer(self.input_channels, 64, args=args, d=2) if self.diff_features_only else getConvLayer(self.input_channels*2, 64, args=args, d=2)
        self.convR1C2 = getConvLayer(64, 64, args=args, d=2)

        self.dpR2 = nn.Dropout2d(p=self.dropout)
        self.convR2C1 = getConvLayer(64, 64, args=args, d=2) if self.diff_features_only else getConvLayer(64*2, 64, args=args, d=2)
        self.convR2C2 = getConvLayer(64, 64, args=args, d=2)

        self.dpR3 = nn.Dropout2d(p=self.dropout)
        self.convR3C1 = getConvLayer(64, 64, args=args, d=2) if self.diff_features_only else getConvLayer(64*2, 64, args=args, d=2)
        self.convR3C2 = getConvLayer(64, 64, args=args, d=2)

        self.convC1 = getConvLayer(192, 512, args=args, d=1)
        self.convC2 = getConvLayer(512, 192, args=args, d=1)
        self.convC3 = getConvLayer(256, 128, args=args, d=1)

        self.dp = nn.Dropout(p=self.dropout)

        self.convG11 = getConvLayer(512, 64, args=args, d=1)
        self.convG12 = nn.Sequential(nn.Linear(128, 64, bias=False), getActivation(args=args))

        self.convOut = nn.Conv1d(128, self.output_channels, kernel_size=1, bias=False)

    def forward(self, x):

        b = x.size(0)   # batch size
        # d = x.size(1)   # dimensionality of input
        n = x.size(2)   # number of points


        # EdgeConv ------------------------------------

        x = get_graph_feature(x, rsize=self.rsize, diff_features_only=self.diff_features_only)              # (b, d, n) -> (b, 2*d, n, F)
        x = self.dpR1(x)                                # (b, 2*d, n) -> (b, 2*d, n, F)
        x = self.convR1C1(x)                            # (b, 2*d, n, F) -> (b, 64, n, F)
        x = self.convR1C2(x)                            # (b, 64, n, F) -> (b, 64, n, F)
        r1 = x.max(dim=-1, keepdim=False)[0]            # (b, 64, n, F) -> (b, 64, n)

        # ---------------------------------------------

        # EdgeConv ------------------------------------

        x = get_graph_feature(r1, rsize=self.rsize, diff_features_only=self.diff_features_only)             # (b, 64, n) -> (b, 64*2, n, F)
        x = self.dpR2(x)                                # (b, 2*d, n) -> (b, 2*d, n, F)
        x = self.convR2C1(x)                            # (b, 64*2, n, F) -> (b, 64, n, F)
        x = self.convR2C2(x)                            # (b, 64, n, F) -> (b, 64, n, F)
        r2 = x.max(dim=-1, keepdim=False)[0]            # (b, 64, n, F) -> (b, 64, n)

        # ---------------------------------------------

        # EdgeConv ------------------------------------
        

        x = get_graph_feature(r2, rsize=self.rsize, diff_features_only=self.diff_features_only)             # (b, 64, n) -> (b, 64*2, n, F)
        x = self.dpR3(x)                                # (b, 2*d, n) -> (b, 2*d, n, F)
        x = self.convR3C1(x)                            # (b, 64*2, n, F) -> (b, 64, n, F)
        x = self.convR3C2(x)                            # (b, 64, n, F) -> (b, 64, n, F)
        r3 = x.max(dim=-1, keepdim=False)[0]            # (b, 64, n, F) -> (b, 64, n)

        # ---------------------------------------------


        x = torch.cat((r1, r2, r3), 1)                  # (b, 192, n)
        x = self.convC1(x)                              # (b, 256, n) -> (b, 512, n)


        # GlobConv ------------------------------------

        # GlobExt ----------

        x_glob = self.convG11(x)                        # (b, 512, n) -> (b, 64, n)
        x_max = F.adaptive_max_pool1d(x_glob, 1)        # (b, 64, n) -> (b, 64, 1)
        x_max = x_max.view(b, -1)                       # (b, 64, 1) -> (b, 64)
        x_avg = F.adaptive_avg_pool1d(x_glob, 1)        # (b, 64, n) -> (b, 64, 1)
        x_avg = x_avg.view(b, -1)                       # (b, 64, 1) -> (b, 64)
        x_glob = torch.cat((x_max, x_avg), 1)           # (b, 128)
        x_glob = self.convG12(x_glob).unsqueeze(-1)     # (b, 128) -> (b, 64, 1)

        # ------------------

        x = self.convC2(x)                              # (b, 512, n) -> (b, 192, n)

        x = torch.cat((x, x_glob.repeat(1, 1, n)), 1)   # (b, 192, n) -> (b, 256, n)

        x = self.convC3(x)                              # (b, 256, n) -> (b, 128, n)

        # ---------------------------------------------


        x = self.dp(x)                                  # (b, 128, n) -> (b, 128, n))

        x = self.convOut(x)                              # (b, 128, n) -> (b, out_channels, n)


        return x

class Model_CNet_smaller(BaseModule):

    def __init__(self, args):

        super(Model_CNet_smaller, self).__init__(args)

        self.dpR1 = nn.Dropout2d(p=self.dropout)
        self.convR1C1 = getConvLayer(self.input_channels*1, 64, d=2, args=args) if self.diff_features_only else getConvLayer(self.input_channels*2, 64, d=2, args=args)
        self.convR1C2 = getConvLayer(64, 64, d=2, args=args)

        self.dpR2 = nn.Dropout2d(p=self.dropout)
        self.convR2C1 = getConvLayer(64*1, 64, d=2, args=args) if self.diff_features_only else getConvLayer(64*2, 64, d=2, args=args)
        self.convR2C2 = getConvLayer(64, 64, d=2, args=args)

        self.convC1 = getConvLayer(128, 256, d=1, args=args)
        self.convC2 = getConvLayer(256, 96, d=1, args=args)
        self.convC3 = getConvLayer(128, 64, d=1, args=args)

        self.dp = nn.Dropout(p=self.dropout)

        self.convG11 = getConvLayer(256, 32, d=1, args=args)
        self.convG12 = nn.Sequential(nn.Linear(64, 32, bias=False), getActivation(args=args))

        self.convOut = nn.Conv1d(64, self.output_channels, kernel_size=1, bias=False)

    def forward(self, x):

        b = x.size(0)   # batch size
        # d = x.size(1)   # dimensionality of input
        n = x.size(2)   # number of points


        # EdgeConv ------------------------------------

        x = get_graph_feature(x, rsize=self.rsize, diff_features_only=self.diff_features_only)              # (b, d, n) -> (b, 2*d, n, F)
        x = self.dpR1(x)                                # (b, 2*d, n) -> (b, 2*d, n, F)
        x = self.convR1C1(x)                            # (b, 2*d, n, F) -> (b, 64, n, F)
        x = self.convR1C2(x)                            # (b, 64, n, F) -> (b, 64, n, F)
        r1 = x.max(dim=-1, keepdim=False)[0]            # (b, 64, n, F) -> (b, 64, n)

        # ---------------------------------------------

        # EdgeConv ------------------------------------

        x = get_graph_feature(r1, rsize=self.rsize, diff_features_only=self.diff_features_only)             # (b, 64, n) -> (b, 64*2, n, F)
        x = self.dpR2(x)                                # (b, 2*d, n) -> (b, 2*d, n, F)
        x = self.convR2C1(x)                            # (b, 64*2, n, F) -> (b, 64, n, F)
        x = self.convR2C2(x)                            # (b, 64, n, F) -> (b, 64, n, F)
        r2 = x.max(dim=-1, keepdim=False)[0]            # (b, 64, n, F) -> (b, 64, n)

        # ---------------------------------------------


        x = torch.cat((r1, r2), 1)                      # (b, 128, n)
        x = self.convC1(x)                              # (b, 128, n) -> (b, 256, n)


        # GlobConv ------------------------------------

        # GlobExt ----------

        x_glob = self.convG11(x)                        # (b, 256, n) -> (b, 32, n)
        x_max = F.adaptive_max_pool1d(x_glob, 1)        # (b, 32, n) -> (b, 32, 1)
        x_max = x_max.view(b, -1)                       # (b, 32, 1) -> (b, 32)
        x_avg = F.adaptive_avg_pool1d(x_glob, 1)        # (b, 32, n) -> (b, 32, 1)
        x_avg = x_avg.view(b, -1)                       # (b, 32, 1) -> (b, 32)
        x_glob = torch.cat((x_max, x_avg), 1)           # (b, 64)
        x_glob = self.convG12(x_glob).unsqueeze(-1)     # (b, 64) -> (b, 32, 1)

        # ------------------

        x = self.convC2(x)                              # (b, 256, n) -> (b, 96, n)

        x = torch.cat((x, x_glob.repeat(1, 1, n)), 1)   # (b, 96, n) -> (b, 128, n)

        x = self.convC3(x)                              # (b, 128, n) -> (b, 64, n)

        # ---------------------------------------------


        x = self.dp(x)                                  # (b, 64, n) -> (b, 64, n))

        x = self.convOut(x)                              # (b, 64, n) -> (b, out_channels, n)


        return x