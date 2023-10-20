# PyTorch implementation of UNet# (UNet_sharp) 

This repository contains code for a image segmentation model based on [Multi-scale context UNet-like network with redesigned skip connections for medical image segmentation] implemented in PyTorch. （Continuously updated in the future）

## Requirements
- pytorch1.7
- torchio<=0.18.20
- python>=3.6

## Installation
1. Install PyTorch.
```sh
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
```
2. Install pip packages.
```sh
pip install -r requirements.txt
```

## Training on [2018 Data Science Bowl](https://www.kaggle.com/c/data-science-bowl-2018) dataset
1. Download dataset from [here](https://www.kaggle.com/c/data-science-bowl-2018/data) to inputs/ and unzip. The file structure is the following:
```
inputs
└── data-science-bowl-2018
    ├── stage1_train
    |   ├── 00ae65...
    │   │   ├── images
    │   │   │   └── 00ae65...
    │   │   └── masks
    │   │       └── 00ae65...            
    │   ├── ...
    |
    ...
```
2. Preprocess. (update soon)
```sh
python preprocess_dsb2018.py
```
3. Train the model. (update soon)
```sh
python train.py --dataset dsb2018_96 --arch UNet_sharp
```
4. Evaluate. (update soon)
```sh
python val.py --name dsb2018_96_UNet_sharp_woDS


## reference
https://github.com/4uiiurz1/pytorch-nested-unet
https://github.com/ZJUGiveLab/UNet-Version
