# Introduction

This repository contains the training code for @rostepifanov's part of the LOWER MATH team solution for [AortaSeg24 Challenge](https://aortaseg24.grand-challenge.org/) (team's final test phase score is [TBA], securing [TBA] place). The @mkotyushev's part of the solution could be found in [this](https://github.com/mkotyushev/aorta) repository.

# Solution overview

The solution is based on the U-Net++ architecture with the following modifications:
- Model is converted to 3D by replacing 2D convolutions with 3D ones and modifying the architecture accordingly
- Additional heads are added along with default U-Net++ head, accumulating activations from lower levels of the network

See more details on training process, used augmentations & other hyperparameters in the [TBA] report paper.

# Reproducing the results

To train model on 1996 seed run the following command:

```
python train.py --verbose -dp ./data -ni 1000 --njobs 3 -bone timm-efficientnetv2-m -e 15 -mn model_tranm_seed0.pth -mlr 0.001 -lr 0.001 -dpr 0.1 -bs 3 --seed 1996
```
