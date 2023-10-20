# PyTorch implementation of UNet# (UNet_sharp)

This repository contains code for a image segmentation model based on [Multi-scale context UNet-like network with redesigned skip connections for medical image segmentation] implemented in PyTorch.

[**NEW**] Add support for multi-class segmentation dataset.

[**NEW**] Add support for PyTorch 1.x.

## Requirements
- PyTorch 1.x

## Installation
1. Create an anaconda environment.
```sh
conda create -n=<env_name> python=3.6 anaconda
conda activate <env_name>
```
2. Install PyTorch.
```sh
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
```
3. Install pip packages.
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
2. Preprocess.
```sh
python preprocess_dsb2018.py
```
3. Train the model.
```sh
python train.py --dataset dsb2018_96 --arch UNet_sharp
```
4. Evaluate.
```sh
python val.py --name dsb2018_96_UNet_sharp_woDS
