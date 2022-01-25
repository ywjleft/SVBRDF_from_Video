# Deep Reflectance Scanning: Recovering Spatially-varying Material Appearance from a Flash-lit Video Sequence

## Introduction

This repository provides a reference implementation for the CGF paper "Deep Reflectance Scanning: Recovering Spatially-varying Material Appearance from a Flash-lit Video Sequence".

More information (including a copy of the paper) can be found at http://share.msraig.info/DeepRefScan/DeepRefScan.htm.

## Citation
If you use our code or models, please cite:

```
@article{https://doi.org/10.1111/cgf.14387,
author = {Ye, Wenjie and Dong, Yue and Peers, Pieter and Guo, Baining},
title = {Deep Reflectance Scanning: Recovering Spatially-varying Material Appearance from a Flash-lit Video Sequence},
journal = {Computer Graphics Forum},
volume = {40},
number = {6},
pages = {409-427},
year = {2021}
}
```

----------------------------------------------------------------
## Usage

### System requirements
   - Linux system (tested on Ubuntu 18.04).
   - An NVidia GPU (tested on Titan X).
   - CUDA toolkit (tested with version 10.0).
   - Tensorflow (tested with version 1.14.0).
   - Python 3 (tested with version 3.6.9).
   - gcc (tested with version 7.5.0).

### Run the test
To run standard SVBRDF reconstruction of 1024x1024 resolution, use test_1k.py. 

To run high-resolution SVBRDF reconstruction with an extra guidance photo, use test_bigmap.py. 

Please refer to the scripts to find the format of input data. We also provide example input data on the project page. 

The trained weights are also provided on the project page. 

### Run the training
#### Prepare training data
Due to copyright, we cannot provide Adobe SVBRDF dataset. The Inria SVBRDF dataset can be found at https://team.inria.fr/graphdeco/projects/multi-materials/.

For Inria data, please use utils/tile_svbrdf.py to convert it into tilable numpy array files (.npy). 

If you have access to Adobe data, please convert and organize it in a similar way. Each SVBRDF should be converted to a numpy file of 4096x4096x12 uint8 array, storing SVBRDF maps in the order of normal, diffuse, roughness, specular in the last channel. 

You also need to pre-generate Perlin noise maps by utils/gen_noise.py. 

Please note that, you need to set the data paths in the scripts. A small modification on the data reading part may also be needed to match your organization of training data. 

#### Train the auto-alignment model
To train the flow model for adjacent frames, run "python -m flow.train_adjacent".

To train the flow model for residual refinement, run "python -m flow.train_distant".

#### Train the SVBRDF reconstruction model
The SVBRDF model training runs with the cooperation of a model trainer and a data loader/generator. 

The loader transfers generated data to the trainer by disk files. It is highly recommended to create a virtual memory disk for the data exchangement. It is recommended to have at least 64GB memory to run the training with a virtual memory disk. 

To start the training, run "python -m svbrdf.train_svbrdf_loader" and "python -m svbrdf.train_svbrdf_trainer" simultaneously. 

## Acknowlegements
The flow model is partially from https://github.com/daigo0927/pwcnet. 

The SVBRDF model is partially from https://github.com/valentin-deschaintre/multi-image-deepNet-SVBRDF-acquisition. 

## Contact
You can contact Wenjie Ye (ywjleft@163.com) if you have any problems.
