# Pixel-level Self-Paced Learning for Super-Resolution

This is  an official implementaion of the paper **Pixel-level Self-Paced Learning for Super-Resolution**, which has been accepted by ICASSP 2020.

[[arxiv](https://arxiv.org/abs/2003.03113)][[PDF](https://arxiv.org/pdf/2003.03113)]

trained model files: [Baidu Pan](https://pan.baidu.com/s/1ZDqJbn0kxAqEmkvMSUMD9g)(code: v0be)

## Requirements

This code is forked from [thstkdgus35/EDSR-PyTorch](https://github.com/thstkdgus35/EDSR-PyTorch). In the light of its README, following libraries are required:

- Python 3.6+ (Python 3.7.0 in my experiments)
- PyTorch >= 1.0.0 (1.1.0 in my experiments)
- numpy
- skimage
- imageio
- matplotlib
- tqdm

## Core Parts

![pspl framework](images/flow.png)

Detail code can be found in [Loss.forward](https://github.com/Elin24/PSPL/blob/2deb17d4bcf7db17463238e143ca94e438e51e2a/src/loss/__init__.py#L60), which can be simplified as:

```python
# take L1 Loss as example

import torch
import torch.nn as nn
import torch.nn.functional as F
from . import pytorch_ssim

class Loss(nn.modules.loss._Loss):
    def __init__(self, spl_alpha, spl_beta, spl_maxVal):
        super(Loss, self).__init__()
        self.loss = nn.L1Loss()
        self.alpha = spl_alpha
        self.beta = spl_beta
        self.maxVal = spl_maxVal

    def forward(self, sr, hr, step):
        # calc sigma value
        sigma = self.alpha * step + self.beta
        # define gauss function
        gauss = lambda x: torch.exp(-((x+1) / sigma) ** 2) * self.maxVal
        # ssim value
        ssim = pytorch_ssim.ssim(hr, sr, reduction='none').detach()
        # calc attention weight
        weight = gauss(ssim).detach()
        nsr, nhr = sr * weight, hr * weight
        # calc loss
        lossval = self.loss(nsr, nhr)
        return lossval
```

the library pytorch_ssim is focked from [Po-Hsun-Su/pytorch-ssim](https://github.com/Po-Hsun-Su/pytorch-ssim) and rewrite some details for adopting it to our requirements.

Attention weight values change according to *SSIM Index* and *steps*:
![attention values](images/attention.png)

## Citation

If you find this project useful for your research, please cite:

```bibtex
@inproceedings{lin2020pixel,
  title={Pixel-Level Self-Paced Learning For Super-Resolution}
  author={Lin, Wei and Gao, Junyu and Wang, Qi and Li, Xuelong},
  booktitle={ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2020},
  pages={2538-2542}
}
```
