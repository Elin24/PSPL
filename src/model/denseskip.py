import torch
import torch.nn as nn
import numpy as np
import math
from model import common

def make_model(args, parent=False): 
    return DenseSkip(args)

class DenseBlock(nn.Module):
    def __init__(self, growth_rate, n_feat_in, n_layers, conv=common.default_conv):
        super(DenseBlock, self).__init__()

        kernel_size = 3 
        feat = n_feat_in 
        body = []

        for i in xrange(n_layers): 
            to_concat = False if (i==0) else True
            layer = common.DenseLayer(conv,feat,growth_rate,kernel_size,True,
                                    to_concat)
            #self.add_module('DenseLayer{}'.format(i+1),layer)
            body.append(layer)

            if i==0: feat = growth_rate
            else: feat += growth_rate

        self.body = nn.Sequential(*body)
        
    def forward(self, x):
        return self.body(x)

class DenseSkip(nn.Module): 
    def __init__(self, args, conv=common.default_conv): 
        super(DenseSkip, self).__init__()

        self.act = nn.ReLU(True)
        kernel_size = 3 
        growth_rate = args.growth_rate
        n_feats = args.n_feats
        n_denseblocks = args.n_denseblocks
        scale = args.scale[0]
        self.is_sub_mean = args.is_sub_mean

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)

        self.head = nn.Sequential(*[nn.Conv2d(in_channels=args.n_channel_in, out_channels=n_feats,
                             kernel_size=kernel_size,padding=1),
                    nn.ReLU(True)])

        self.dense_blocks = []
        for i in xrange(n_denseblocks): 
            db = DenseBlock(growth_rate, n_feats, args.n_layers)
            self.add_module('DenseBlock{}'.format(i+1), db)
            self.dense_blocks.append(db)

        self.bottleneck = nn.Conv2d(in_channels=n_feats*(n_denseblocks+1), 
                                    out_channels=n_feats*2, kernel_size=1,
                                     stride=1, padding=0, bias=False)

        self.tail = [common.Upsampler(nn.ConvTranspose2d, scale=scale, n_feat=n_feats*2,
                     act=self.act, bias=False, type='deconv')]
        self.tail = nn.Sequential(*self.tail)

        self.reconstruction = nn.Conv2d(in_channels=n_feats*2, out_channels=args.n_channel_out,
                                         kernel_size=kernel_size, stride=1, padding=1, bias=False)

        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

    def forward(self, x): 
        if self.is_sub_mean: 
            x = self.sub_mean(x)

        x = self.head(x)
        outs = [x]

        for db in self.dense_blocks: 
            x = db(x)
            
            outs.append(x)

        x = torch.cat(outs, 1)
        x = self.bottleneck(x)        
        x = self.tail(x)
        x = self.reconstruction(x)

        if self.is_sub_mean: 
            x = self.add_mean(x)

        return x