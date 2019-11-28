import pdb
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils import weight_norm as wn

from layers import * 
from utils import * 


class masked_conv2d(nn.Conv2d):
    def __init__(self, num_filters_in, num_filters_out, kernel_size=(3,3),
                 mask_type='B', n_color=1):
        """2D Convolution with masked weight for Autoregressive connection"""
        # Pad to maintain spatial dimensions
        padding = (int((kernel_size[0] - 1) / 2),
                   int((kernel_size[1] - 1) / 2))
        super(masked_conv2d, self).__init__(
            num_filters_in, num_filters_out, kernel_size, padding=padding)
        assert mask_type in ['A', 'B']
        assert n_color == 1  # TODO: Add support for color channel masking
        self.mask_type = mask_type
        ch_out, ch_in, height, width = self.weight.size()

        # Mask
        #         -------------------------------------
        #        |  1       1       1       1       1 |
        #        |  1       1       1       1       1 |
        #        |  1       1    1 if B     0       0 |   H // 2
        #        |  0       0       0       0       0 |   H // 2 + 1
        #        |  0       0       0       0       0 |
        #         -------------------------------------
        #  index    0       1     W//2    W//2+1

        mask = torch.ones(ch_out, ch_in, height, width)
        yc = height // 2
        xc = width // 2
        mask[:, :, yc, xc:] = 0
        mask[:, :, yc + 1:] = 0
        if mask_type == 'B':
            mask[:, :, yc, xc] = 1
        self.register_buffer('mask', mask)

    def forward(self, x):
        self.weight.data *= self.mask
        return super(masked_conv2d, self).forward(x)


class OurPixelCNNLayer_up(nn.Module):
    def __init__(self, nr_resnet, nr_filters, resnet_nonlinearity):
        super(OurPixelCNNLayer_up, self).__init__()
        self.nr_resnet = nr_resnet
        conv_op = lambda cin, cout: masked_conv2d(cin, cout, mask_type='B', n_color=1)
        # stream from pixels above and to the left
        self.u_stream = nn.ModuleList([gated_resnet(nr_filters, conv_op, 
                                        resnet_nonlinearity, skip_connection=0) 
                                            for _ in range(nr_resnet)])
        
        # stream from pixels above and to the left, with skip connection
        self.ul_stream = nn.ModuleList([gated_resnet(nr_filters, conv_op, 
                                        resnet_nonlinearity, skip_connection=1) 
                                            for _ in range(nr_resnet)])

    def forward(self, u, ul):
        u_list, ul_list = [], []
        
        for i in range(self.nr_resnet):
            u  = self.u_stream[i](u)
            ul = self.ul_stream[i](ul, a=u)
            u_list  += [u]
            ul_list += [ul]

        return u_list, ul_list


class OurPixelCNNLayer_down(nn.Module):
    def __init__(self, nr_resnet, nr_filters, resnet_nonlinearity):
        super(OurPixelCNNLayer_down, self).__init__()
        self.nr_resnet = nr_resnet
        conv_op = lambda cin, cout: masked_conv2d(cin, cout, mask_type='B', n_color=1)
        # stream from pixels above
        self.u_stream  = nn.ModuleList([gated_resnet(nr_filters, conv_op, 
                                        resnet_nonlinearity, skip_connection=1) 
                                            for _ in range(nr_resnet)])
        
        # stream from pixels above and to thes left
        self.ul_stream = nn.ModuleList([gated_resnet(nr_filters, conv_op, 
                                        resnet_nonlinearity, skip_connection=2) 
                                            for _ in range(nr_resnet)])

    def forward(self, u, ul, u_list, ul_list):
        for i in range(self.nr_resnet):
            u  = self.u_stream[i](u, a=u_list.pop())
            ul = self.ul_stream[i](ul, a=torch.cat((u, ul_list.pop()), 1))
        
        return u, ul


class OurPixelCNN(nn.Module):
    def __init__(self, nr_resnet=5, nr_filters=80, nr_logistic_mix=10, 
                    resnet_nonlinearity='concat_elu', input_channels=3):
        super(OurPixelCNN, self).__init__()
        if resnet_nonlinearity == 'concat_elu' : 
            self.resnet_nonlinearity = lambda x : concat_elu(x)
        else : 
            raise Exception('right now only concat elu is supported as resnet nonlinearity.')

        self.nr_filters = nr_filters
        self.input_channels = input_channels
        self.nr_logistic_mix = nr_logistic_mix
        self.right_shift_pad = nn.ZeroPad2d((1, 0, 0, 0))
        self.down_shift_pad  = nn.ZeroPad2d((0, 0, 1, 0))

        assert input_channels == 1  # FIXME: temporary

        down_nr_resnet = [nr_resnet] + [nr_resnet + 1] * 2
        self.down_layers = nn.ModuleList([OurPixelCNNLayer_down(down_nr_resnet[i], nr_filters, 
                                                self.resnet_nonlinearity) for i in range(3)])

        self.up_layers   = nn.ModuleList([OurPixelCNNLayer_up(nr_resnet, nr_filters, 
                                                self.resnet_nonlinearity) for _ in range(3)])

        # TODO: Dilate convolutions to increase receptive field, as we no longer downsample
        self.downsize_u_stream = nn.ModuleList([masked_conv2d(nr_filters, nr_filters, mask_type='B', n_color=input_channels)
                                                 for _ in range(2)])

        self.downsize_ul_stream = nn.ModuleList([masked_conv2d(nr_filters, nr_filters, mask_type='B', n_color=input_channels)
                                                 for _ in range(2)])
        
        self.upsize_u_stream = nn.ModuleList([masked_conv2d(nr_filters, nr_filters, mask_type='B', n_color=input_channels)
                                              for _ in range(2)])
        
        self.upsize_ul_stream = nn.ModuleList([masked_conv2d(nr_filters, nr_filters, mask_type='B', n_color=input_channels)
                                              for _ in range(2)])

        # NOTE: In PixelCNN++, u_init can access a 2x3 region above each pixel, while this mask
        # only accesses a 1x3 region above the pixel and a pixel to the left
        self.u_init = wn(masked_conv2d(input_channels + 1, nr_filters, mask_type='A', n_color=input_channels))
        self.ul_init = wn(masked_conv2d(input_channels + 1, nr_filters, mask_type='A', n_color=input_channels))

        num_mix = 3 if self.input_channels == 1 else 10
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

        ###      UP PASS    ###
        x = x if sample else torch.cat((x, self.init_padding), 1)
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
            u, ul = self.down_layers[i](u, ul, u_list, ul_list)

            # upscale (only twice)
            if i != 2 :
                u  = self.upsize_u_stream[i](u)
                ul = self.upsize_ul_stream[i](ul)

        x_out = self.nin_out(F.elu(ul))

        assert len(u_list) == len(ul_list) == 0, pdb.set_trace()

        return x_out
        
