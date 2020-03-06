from . import common 
import torch.nn as nn 

def make_model(args): 
    return SRResNet(args)

class SRResNet(nn.Module): 
    def __init__(self, args, conv=common.default_conv): 
        super(SRResNet, self).__init__()
        
        #kernel_size = 3 
        #scale = args.scale[0]
        #act = nn.LeakyReLU(negative_slope=0.2)
        
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)

        self.body = nn.Sequential(
            conv(args.n_colors, args.n_feats * 2, 9),
            nn.ReLU(inplace=True),
            conv(args.n_feats * 2, args.n_feats, 5),
            nn.ReLU(inplace=True),
            conv(args.n_feats, args.n_colors, 5)
        )

        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

    def forward(self, x): 
        x = self.sub_mean(x)

        x = self.body(x)

        x = self.add_mean(x)

        return x