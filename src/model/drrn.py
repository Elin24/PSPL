import torch.nn as nn 
from . import common

def make_model(args): 
    return DRRN(args)

class DRRN(nn.Module): 
    def __init__(self, args, conv=common.default_conv): 
        super(DRRN, self).__init__()
        
        kernel_size = 3 
        self.n_layers = args.n_layers 
        self.is_residual = args.n_colors == args.n_colors 

        self.head = conv(args.n_colors,args.n_feats,kernel_size,bias=False)
        self.conv1 = conv(args.n_feats,args.n_feats,kernel_size,bias=False)
        self.conv2 = conv(args.n_feats,args.n_feats,kernel_size,bias=False)
        self.tail = conv(args.n_feats,args.n_colors,kernel_size,bias=False)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x): 
        residual = x
        inputs = self.head(self.act(x))
        out = inputs

        for _ in range(self.n_layers): 
            out = self.conv2(self.act(self.conv1(self.act(out))))
            out += inputs

        self.features = [self.act(out)]
        out = self.tail(self.features[0])

        return out + residual if (self.is_residual) else out
    