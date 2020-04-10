# -*- coding: utf-8 -*-

import glob
import os
import PIL
from PIL import Image
from shutil import copyfile

# Image.resize(size, PIL.Image.BICUBIC)

def mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir

def mkdata(root, scale):
    hrdir = mkdir(os.path.join(f'datasetx{scale}', 'benchmark', root, 'HR'))
    lrdir = mkdir(os.path.join(f'datasetx{scale}', 'benchmark', root, 'LR_bicubic', f'X{scale}'))
    for imgp in os.listdir(root):
        hrimgpath = os.path.join(hrdir, imgp)
        spimgp = os.path.splitext(imgp)
        lrimgpath = os.path.join(lrdir, f'{spimgp[0]}x{scale}{spimgp[1]}')
        
        imgp = os.path.join(root, imgp)
        hrimg = Image.open(imgp)
        w, h = hrimg.size
        nw = w if w % scale == 0 else (w - w % scale)
        nh = h if h % scale == 0 else (h - h % scale)
        if nw == w and nh == h:
            copyfile(imgp, hrimgpath)
        else:
            hrimg = hrimg.resize((nw, nh), PIL.Image.BICUBIC)
            hrimg.save(hrimgpath)
        nw, nh = nw // scale, nh // scale
        lrimg = hrimg.resize((nw, nh), PIL.Image.BICUBIC)
        lrimg.save(lrimgpath)
    print(f'{root} - {scale} : Ok.')


if __name__ == '__main__':
    for root in ['Set5', 'Set14', 'B100', 'Urban100']:
        for scale in [2,3,4]:
            mkdata(root, scale)