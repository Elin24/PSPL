import torch 
import torch.nn as nn 
import common
from itertools import izip 

def make_model(args, parent=False):
    return RDN(args)

class RDN(nn.Module): 
    def __init__(self, args, conv=common.default_conv): 
        super(RDN, self).__init__()

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)

        kernel_size = 3
        self.is_sub_mean = args.is_sub_mean

        self.conv1 = conv(args.n_channel_in, args.n_feats, kernel_size, bias=True)
        self.conv2 = conv(args.n_feats, args.n_feats, kernel_size, bias=True)

        self.RDBs = []
        for i in xrange(args.n_denseblocks): 
            RDB = common.RDB(args.n_feats,args.n_layers,args.growth_rate,conv,kernel_size,True)
            self.add_module('RDB{}'.format(i+1),RDB)
            self.RDBs.append(RDB)

        self.gff_1 = nn.Conv2d(args.n_feats*args.n_denseblocks, args.n_feats,
                                kernel_size=1, padding=0, bias=True)
        self.gff_3 = conv(args.n_feats, args.n_feats, kernel_size, bias=True)

        m_tail = [common.Upsampler(conv, args.scale[0], args.n_feats, act=False),
                    conv(args.n_feats, args.n_channel_out, kernel_size)]

        self.tail = nn.Sequential(*m_tail)
        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

    def forward(self, x): 
        if self.is_sub_mean: 
            x = self.sub_mean(x)

        F_minus = self.conv1(x)
        x = self.conv2(F_minus)
        to_concat = []

        for db in self.RDBs: 
            x = db(x)
            to_concat.append(x)

        x = torch.cat(to_concat, 1)
        x = self.gff_1(x)
        x = self.gff_3(x)
        x = x + F_minus

        self.down_feats = x

        out = self.tail(x)

        if self.is_sub_mean: 
            out = self.add_mean(out)

        return out 
    
    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        
        for (k1, p1), (k2, p2) in izip(state_dict.items(), own_state.items()): 
            if (k1.split('.')[0] == '0') or (k1.split('.')[0] == '5'): #do not copy shift mean layer
                continue

            if isinstance(p1, nn.Parameter): 
                p1 = p1.data
                
            try: 
                own_state[k2].copy_(p1) 
            except Exception: 
                raise RuntimeError('error')
                