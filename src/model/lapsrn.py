import torch 
import torch.nn as nn 
import math 
try:
    # Python 2
    from itertools import izip
except ImportError:
    # Python 3
    izip = zip
    xrange = range

def make_model(args): 
    return LapSRN(args)

class Conv_Block(nn.Module): 
    def __init__(self, n_layers, n_feats, negative_slope, kernel_size, upsample=True):
        super(Conv_Block, self).__init__()
        body = []
        tail = []

        for _ in xrange(n_layers):
            layer = nn.Conv2d(in_channels=n_feats, out_channels=n_feats,
                             kernel_size=kernel_size,stride=1,padding=1,bias=False)
            act = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
            body.extend([layer,act])

        if upsample: 
            upsampled = nn.ConvTranspose2d(in_channels=n_feats, out_channels=n_feats,
                                        kernel_size=4, stride=2, padding=1,bias=False)
            act = nn.LeakyReLU(negative_slope, inplace=True)
            tail.extend([upsampled, act])
            
        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(*tail)

    def forward(self, x): 
        self.down_feats = self.body(x)
        return self.tail(self.down_feats)

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        
        for (k1, p1), (k2, p2) in izip(state_dict.items(), own_state.items()): 
            if isinstance(p1, nn.Parameter): 
                p1 = p1.data
            try: 
                own_state[k2].copy_(p1) 
            except Exception: 
                raise RuntimeError('p1 dims = {}, p2 dims = {}'.format(p1.size(), p2.size()))

class LapSRN(nn.Module): 
    def __init__(self, args): 
        super(LapSRN, self).__init__()

        kernel_size = 3 
        self.scale = args.scale[0]
        
        head = [nn.Conv2d(in_channels=args.n_colors, out_channels=args.n_feats, 
                        kernel_size=kernel_size, stride=1, padding=1, bias=False),
                nn.LeakyReLU(args.negative_slope, True)]
        self.head = nn.Sequential(*head)

        self.feats_branch, self.images_branch, self.residuals_branch = [], [], []

        n_iters = 1 if (self.scale==1) else int(math.log(self.scale, 2))
        for i in xrange(n_iters): 
            feat_branch = Conv_Block(args.n_layers, args.n_feats, args.negative_slope,
                                        kernel_size, upsample=not(self.scale==1))
            if not (self.scale==1):
                img_branch = nn.ConvTranspose2d(in_channels=args.n_colors if (i==0) \
                                                else args.n_colors,
                                                out_channels=args.n_colors,
                                                kernel_size=4, stride=2, padding=1,
                                                bias=False)
            else: 
                img_branch = nn.Conv2d(in_channels=args.n_colors if (i==0) else args.n_colors,
                                        kernel_size=kernel_size,out_channels=args.n_colors,
                                        stride=1,padding=1,bias=True)

            res_branch = nn.Conv2d(in_channels=args.n_feats, out_channels=args.n_colors,
                                    kernel_size=kernel_size,stride=1,padding=1,bias=False)
            
            self.add_module('img_branch_{}'.format(i+1), img_branch)
            self.add_module('residual_branch_{}'.format(i+1), res_branch)
            self.add_module('feat_branch_{}'.format(i+1),feat_branch)

            self.feats_branch.append(feat_branch)
            self.images_branch.append(img_branch)
            self.residuals_branch.append(res_branch)

    def forward(self, x):
        fx = self.head(x)

        self.features = []
        self.down_feats = None 

        for feat,img,res in izip(self.feats_branch,self.images_branch,
                                self.residuals_branch):
            fx = feat(fx)
            ix = img(x)
            rx = res(fx)

            x = rx + ix

            self.features.append(fx)

            if self.down_feats is None: 
                self.down_feats = feat.down_feats

        return x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        
        for (k1, p1), (k2, p2) in izip(state_dict.items(), own_state.items()): 
            if isinstance(p1, nn.Parameter): 
                p1 = p1.data
            try: 
                own_state[k2].copy_(p1) 
            except Exception: 
                raise RuntimeError('error; {}, {}, {}, {}'.format(k1, k2, p1.size(), p2.size()))