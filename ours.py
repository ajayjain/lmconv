from IPython import embed
import math
import pdb

import numpy as np
import torch 
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils import weight_norm as wn

from layers import * 
from utils import * 


class masked_conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3),
                 dilation=1, mask_type='B'):
        """2D Convolution with masked weight for autoregressive connection"""
        super(masked_conv2d, self).__init__()
        assert mask_type in ['A', 'B']
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.mask_type = mask_type

        # Pad to maintain spatial dimensions
        self.padding = ((dilation * (kernel_size[0] - 1)) // 2,
                   (dilation * (kernel_size[1] - 1)) // 2)

        # Conv parameters
        self.weight = Parameter(torch.Tensor(out_channels, in_channels, *kernel_size))
        self.bias = Parameter(torch.Tensor(out_channels))

        # Mask
        #         -------------------------------------
        #        |  1       1       1       1       1 |
        #        |  1       1       1       1       1 |
        #        |  1       1    1 if B     0       0 |   H // 2
        #        |  0       0       0       0       0 |   H // 2 + 1
        #        |  0       0       0       0       0 |
        #         -------------------------------------
        #  index    0       1     W//2    W//2+1
        assert self.weight.size(0) == out_channels
        assert self.weight.size(1) == in_channels
        assert self.weight.size(2) == kernel_size[0]
        assert self.weight.size(3) == kernel_size[1]
        mask = torch.ones(out_channels, in_channels, kernel_size[0], kernel_size[1])
        yc = kernel_size[0] // 2
        xc = kernel_size[1] // 2
        mask[:, :, yc, xc:] = 0
        mask[:, :, yc + 1:] = 0
        if mask_type == 'B':
            mask[:, :, yc, xc] = 1
        self.register_buffer('mask', mask)

        self.reset_parameters()

    def reset_parameters(self):
        # From PyTorch _ConvNd implementation
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        masked_weight = self.weight * self.mask
        return F.conv2d(x, masked_weight, self.bias, padding=self.padding, dilation=self.dilation)


class OurPixelCNNLayer_up(nn.Module):
    def __init__(self, nr_resnet, nr_filters, resnet_nonlinearity, kernel_size=(5,5),
                 weight_norm=True, two_stream=True):
        super(OurPixelCNNLayer_up, self).__init__()
        self.nr_resnet = nr_resnet
        self.two_stream = two_stream
        if weight_norm:
            conv_op = lambda cin, cout: wn(masked_conv2d(cin, cout, mask_type='B', kernel_size=kernel_size))
        else:
            conv_op = lambda cin, cout: masked_conv2d(cin, cout, mask_type='B', kernel_size=kernel_size)
        # stream from pixels above and to the left
        self.u_stream = nn.ModuleList([gated_resnet(nr_filters, conv_op, 
                                        resnet_nonlinearity, skip_connection=0) 
                                            for _ in range(nr_resnet)])
        
        # stream from pixels above and to the left, with skip connection
        if self.two_stream:
            self.ul_stream = nn.ModuleList([gated_resnet(nr_filters, conv_op, 
                                            resnet_nonlinearity, skip_connection=1) 
                                                for _ in range(nr_resnet)])

    def forward(self, u, ul=None):
        if self.two_stream:
            assert ul is not None
            u_list, ul_list = [], []
            for i in range(self.nr_resnet):
                u  = self.u_stream[i](u)
                ul = self.ul_stream[i](ul, a=u)
                u_list  += [u]
                ul_list += [ul]
            return u_list, ul_list

        u_list = []
        for i in range(self.nr_resnet):
            u  = self.u_stream[i](u)
            u_list  += [u]
        return u_list


class OurPixelCNNLayer_down(nn.Module):
    def __init__(self, nr_resnet, nr_filters, resnet_nonlinearity, kernel_size=(5,5),
                 weight_norm=True, two_stream=True):
        super(OurPixelCNNLayer_down, self).__init__()
        self.nr_resnet = nr_resnet
        self.two_stream = two_stream
        if weight_norm:
            conv_op = lambda cin, cout: wn(masked_conv2d(cin, cout, mask_type='B', kernel_size=kernel_size))
        else:
            conv_op = lambda cin, cout: masked_conv2d(cin, cout, mask_type='B', kernel_size=kernel_size)
        # stream from pixels above
        self.u_stream  = nn.ModuleList([gated_resnet(nr_filters, conv_op, 
                                        resnet_nonlinearity, skip_connection=1) 
                                            for _ in range(nr_resnet)])
        
        if self.two_stream:
            # stream from pixels above and to thes left
            self.ul_stream = nn.ModuleList([gated_resnet(nr_filters, conv_op, 
                                            resnet_nonlinearity, skip_connection=2) 
                                                for _ in range(nr_resnet)])

    def forward(self, u, u_list, ul=None, ul_list=None):
        if self.two_stream:
            assert ul is not None
            assert ul_list is not None

            for i in range(self.nr_resnet):
                u  = self.u_stream[i](u, a=u_list.pop())
                ul = self.ul_stream[i](ul, a=torch.cat((u, ul_list.pop()), 1))
            return u, ul

        for i in range(self.nr_resnet):
            u  = self.u_stream[i](u, a=u_list.pop())
        return u


class OurPixelCNN(nn.Module):
    def __init__(self, nr_resnet=5, nr_filters=80, nr_logistic_mix=10, 
                    resnet_nonlinearity='concat_elu', input_channels=3, kernel_size=(5,5),
                    max_dilation=2, weight_norm=True, two_stream=True):
        super(OurPixelCNN, self).__init__()
        self.two_stream = two_stream
        assert two_stream == False
        if resnet_nonlinearity == 'concat_elu':
            self.resnet_nonlinearity = lambda x : concat_elu(x)
        else:
            raise Exception('right now only concat elu is supported as resnet nonlinearity.')

        down_nr_resnet = [nr_resnet] + [nr_resnet + 1] * 2
        self.down_layers = nn.ModuleList([OurPixelCNNLayer_down(down_nr_resnet[i], nr_filters, self.resnet_nonlinearity, 
                                                kernel_size=kernel_size, weight_norm=weight_norm, two_stream=self.two_stream) for i in range(3)])

        self.up_layers   = nn.ModuleList([OurPixelCNNLayer_up(nr_resnet, nr_filters, self.resnet_nonlinearity,  
                                                kernel_size=kernel_size, weight_norm=weight_norm, two_stream=self.two_stream) for _ in range(3)])

        if weight_norm:
            conv_op_init = lambda cin, cout: wn(masked_conv2d(cin, cout, mask_type='A', kernel_size=kernel_size))
            conv_op = lambda cin, cout: wn(masked_conv2d(cin, cout, mask_type='B', kernel_size=kernel_size, dilation=max_dilation))
        else:
            conv_op_init = lambda cin, cout: masked_conv2d(cin, cout, mask_type='A', kernel_size=kernel_size)
            conv_op = lambda cin, cout: masked_conv2d(cin, cout, mask_type='B', kernel_size=kernel_size, dilation=max_dilation)

        # NOTE: In PixelCNN++, u_init can access a 2x3 region above each pixel
        self.u_init = conv_op_init(input_channels + 1, nr_filters)
        self.downsize_u_stream = nn.ModuleList([conv_op(nr_filters, nr_filters) for _ in range(2)])
        self.upsize_u_stream = nn.ModuleList([conv_op(nr_filters, nr_filters) for _ in range(2)])

        if self.two_stream:
            self.ul_init = conv_op_init(input_channels + 1, nr_filters)
            self.downsize_ul_stream = nn.ModuleList([conv_op(nr_filters, nr_filters) for _ in range(2)])
            self.upsize_ul_stream = nn.ModuleList([conv_op(nr_filters, nr_filters) for _ in range(2)])

        num_mix = 3 if input_channels == 1 else 10
        self.nin_out = nin(nr_filters, num_mix * nr_logistic_mix)
        self.init_padding = None

    def forward(self, x, sample=False):
        # similar as done in the tf repo :  
        if self.init_padding is None and not sample: 
            xs = [int(y) for y in x.size()]
            padding = Variable(torch.ones(xs[0], 1, xs[2], xs[3]), requires_grad=False)
            self.init_padding = padding.cuda() if x.is_cuda else padding
        
        if sample : 
            xs = [int(y) for y in x.size()]
            padding = Variable(torch.ones(xs[0], 1, xs[2], xs[3]), requires_grad=False)
            padding = padding.cuda() if x.is_cuda else padding
            x = torch.cat((x, padding), 1)
        
        x = x if sample else torch.cat((x, self.init_padding), 1)

        if self.two_stream:
            return self.forward_two_stream(x)

        return self.forward_one_stream(x)

    def forward_two_stream(self, x):
        ###      UP PASS    ###
        u_list  = [self.u_init(x)]
        ul_list = [self.ul_init(x)]
        for i in range(3):
            # resnet block
            u_out, ul_out = self.up_layers[i](u_list[-1], ul_list[-1])
            u_list  += u_out
            ul_list += ul_out

            if i != 2: 
                # downscale (only twice)
                u_list  += [self.downsize_u_stream[i](u_list[-1])]
                ul_list += [self.downsize_ul_stream[i](ul_list[-1])]

        ###    DOWN PASS    ###
        u  = u_list.pop()
        ul = ul_list.pop()
        
        for i in range(3):
            # resnet block
            u, ul = self.down_layers[i](u, u_list, ul, ul_list)

            # upscale (only twice)
            if i != 2 :
                u  = self.upsize_u_stream[i](u)
                ul = self.upsize_ul_stream[i](ul)

        x_out = self.nin_out(F.elu(ul))

        if not torch.isfinite(x_out).all().cpu().item():
            print("ERROR: NaN or Inf in returned tensor, embedding")
            embed()

        assert len(u_list) == len(ul_list) == 0, pdb.set_trace()

        return x_out
        
    def forward_one_stream(self, x):
        ###      UP PASS    ###
        u_list  = [self.u_init(x)]
        # resnet block and downscale
        u_list += self.up_layers[0](u_list[-1])
        u_list += [self.downsize_u_stream[0](u_list[-1])]
        u_list += self.up_layers[1](u_list[-1])
        u_list += [self.downsize_u_stream[1](u_list[-1])]
        u_list += self.up_layers[2](u_list[-1])

        ###    DOWN PASS    ###
        # resnet block and upscale
        u = u_list.pop()
        u = self.down_layers[0](u, u_list)
        u = self.upsize_u_stream[0](u)
        u = self.down_layers[1](u, u_list)
        u = self.upsize_u_stream[1](u)
        u = self.down_layers[2](u, u_list)

        x_out = self.nin_out(F.elu(u))

        if not torch.isfinite(x_out).all().cpu().item():
            print("ERROR: NaN or Inf in returned tensor, embedding")
            embed()

        return x_out
        
