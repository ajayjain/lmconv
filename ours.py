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


class OurPixelCNNLayer_up(nn.Module):
    def __init__(self, nr_resnet, nr_filters, resnet_nonlinearity, conv_op, kernel_size=(5,5),
                 weight_norm=True, two_stream=True):
        super(OurPixelCNNLayer_up, self).__init__()
        self.nr_resnet = nr_resnet
        self.two_stream = two_stream

        self.u_stream = nn.ModuleList([gated_resnet(nr_filters, conv_op, 
                                        resnet_nonlinearity, skip_connection=0) 
                                            for _ in range(nr_resnet)])
        
        if self.two_stream:
            self.ul_stream = nn.ModuleList([gated_resnet(nr_filters, conv_op, 
                                            resnet_nonlinearity, skip_connection=1) 
                                                for _ in range(nr_resnet)])

    def forward(self, u, ul=None, mask=None):
        if self.two_stream:
            assert ul is not None
            u_list, ul_list = [], []
            for i in range(self.nr_resnet):
                u  = self.u_stream[i](u, mask=mask)
                ul = self.ul_stream[i](ul, a=u, mask=mask)
                u_list  += [u]
                ul_list += [ul]
            return u_list, ul_list

        u_list = []
        for i in range(self.nr_resnet):
            u  = self.u_stream[i](u, mask=mask)
            u_list  += [u]
        return u_list


class OurPixelCNNLayer_down(nn.Module):
    def __init__(self, nr_resnet, nr_filters, resnet_nonlinearity, conv_op, kernel_size=(5,5),
                 weight_norm=True, two_stream=True):
        super(OurPixelCNNLayer_down, self).__init__()
        self.nr_resnet = nr_resnet
        self.two_stream = two_stream

        self.u_stream  = nn.ModuleList([gated_resnet(nr_filters, conv_op, 
                                        resnet_nonlinearity, skip_connection=1) 
                                            for _ in range(nr_resnet)])
        
        if self.two_stream:
            self.ul_stream = nn.ModuleList([gated_resnet(nr_filters, conv_op, 
                                            resnet_nonlinearity, skip_connection=2) 
                                                for _ in range(nr_resnet)])

    def forward(self, u, u_list, ul=None, ul_list=None, mask=None):
        if self.two_stream:
            assert ul is not None
            assert ul_list is not None

            for i in range(self.nr_resnet):
                u  = self.u_stream[i](u, a=u_list.pop(), mask=mask)
                ul = self.ul_stream[i](ul, a=torch.cat((u, ul_list.pop()), 1), mask=mask)
            return u, ul

        for i in range(self.nr_resnet):
            u  = self.u_stream[i](u, a=u_list.pop(), mask=mask)
        return u


class OurPixelCNN(nn.Module):
    def __init__(self, nr_resnet=5, nr_filters=80, nr_logistic_mix=10,
                    resnet_nonlinearity='concat_elu', input_channels=3, kernel_size=(5,5),
                    max_dilation=2, weight_norm=True, two_stream=True):
        super(OurPixelCNN, self).__init__()
        assert resnet_nonlinearity == 'concat_elu'
        self.two_stream = two_stream
        self.resnet_nonlinearity = lambda x : concat_elu(x)
        self.init_padding = None

        if weight_norm:
            conv_op_init = lambda cin, cout: wn(input_masked_conv2d(cin, cout, kernel_size=kernel_size))
            conv_op_dilated = lambda cin, cout: wn(input_masked_conv2d(cin, cout, kernel_size=kernel_size, dilation=max_dilation))
            conv_op = lambda cin, cout: wn(input_masked_conv2d(cin, cout, kernel_size=kernel_size))
        else:
            conv_op_init = lambda cin, cout: input_masked_conv2d(cin, cout, kernel_size=kernel_size)
            conv_op_dilated = lambda cin, cout: input_masked_conv2d(cin, cout, kernel_size=kernel_size, dilation=max_dilation)
            conv_op = lambda cin, cout: input_masked_conv2d(cin, cout, kernel_size=kernel_size)

        down_nr_resnet = [nr_resnet] + [nr_resnet + 1] * 2
        self.down_layers = nn.ModuleList([OurPixelCNNLayer_down(down_nr_resnet[i], nr_filters, self.resnet_nonlinearity, conv_op,
                                                kernel_size=kernel_size, weight_norm=weight_norm, two_stream=self.two_stream) for i in range(3)])

        self.up_layers   = nn.ModuleList([OurPixelCNNLayer_up(nr_resnet, nr_filters, self.resnet_nonlinearity, conv_op,
                                                kernel_size=kernel_size, weight_norm=weight_norm, two_stream=self.two_stream) for _ in range(3)])

        # NOTE: In PixelCNN++, u_init can access a 2x3 region above each pixel
        self.u_init = conv_op_init(input_channels + 1, nr_filters)
        self.downsize_u_stream = nn.ModuleList([conv_op_dilated(nr_filters, nr_filters) for _ in range(2)])
        self.upsize_u_stream = nn.ModuleList([conv_op_dilated(nr_filters, nr_filters) for _ in range(2)])

        if self.two_stream:
            self.ul_init = conv_op_init(input_channels + 1, nr_filters)
            self.downsize_ul_stream = nn.ModuleList([conv_op_dilated(nr_filters, nr_filters) for _ in range(2)])
            self.upsize_ul_stream = nn.ModuleList([conv_op_dilated(nr_filters, nr_filters) for _ in range(2)])

        num_mix = 3 if input_channels == 1 else 10
        self.nin_out = nin(nr_filters, num_mix * nr_logistic_mix)

    def forward(self, x, sample=False, mask_init=None, mask=None):
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
            return self.forward_two_stream(x, mask_init, mask)

        return self.forward_one_stream(x, mask_init, mask)

    def forward_two_stream(self, x, mask_init=None, mask=None):
        assert mask_init is None and mask is None, "Masking not implemented for 2 stream"

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
        
    def forward_one_stream(self, x, mask_init=None, mask=None):
        ###      UP PASS    ###
        u_list  = [self.u_init(x, mask=mask_init)]
        # resnet block and downscale
        u_list += self.up_layers[0](u_list[-1], mask=mask)
        u_list += [self.downsize_u_stream[0](u_list[-1], mask=mask)]
        u_list += self.up_layers[1](u_list[-1], mask=mask)
        u_list += [self.downsize_u_stream[1](u_list[-1], mask=mask)]
        u_list += self.up_layers[2](u_list[-1], mask=mask)

        ###    DOWN PASS    ###
        # resnet block and upscale
        u = u_list.pop()
        u = self.down_layers[0](u, u_list, mask=mask)
        u = self.upsize_u_stream[0](u, mask=mask)
        u = self.down_layers[1](u, u_list, mask=mask)
        u = self.upsize_u_stream[1](u, mask=mask)
        u = self.down_layers[2](u, u_list, mask=mask)

        x_out = self.nin_out(F.elu(u))

        if not torch.isfinite(x_out).all().cpu().item():
            print("ERROR: NaN or Inf in returned tensor, embedding")
            embed()

        return x_out
        
