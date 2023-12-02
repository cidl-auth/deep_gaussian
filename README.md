# END-TO-END TRAINABLE GAUSSIAN FILTERING FOR ELECTROCARDIOGRAM SIGNAL CLASSIFICATION USING DEEP LEARNING

:page_with_curl: [**[Paper]**](https://ieeexplore.ieee.org/abstract/document/10290075)
:rocket: [**[Demo]**](https://colab.research.google.com/github/cidl-auth/deep_gaussian/blob/main/demo.ipynb)&nbsp;

Official PyTorch implementation of the paper **"END-TO-END TRAINABLE GAUSSIAN FILTERING FOR ELECTROCARDIOGRAM SIGNAL CLASSIFICATION USING DEEP LEARNING"**
by Angelos Nalmpantis, Nikolaos Passalis and Anastasios Tefas (EUSIPCO 2023)

## Overview
This repository contains a practical tool that employs Gaussian filters to denoise an input signal and extract its high-order derivatives. The cut-off frequency is incorporated into the learning process in an end-to-end manner, enabling the model to select the appropriate amount of denoising via gradient-based optimization. The layer applying the Gaussian filters can be incorporated into any DL pipeline that is used for time-series analysis.

<img src="https://github.com/cidl-auth/deep_gaussian/blob/main/figures/method.png"/>

### Example

```python
In  [1]: import torch

In  [2]: from deep_gaussian_filtering import DeepGaussianFilter

In  [3]: x = torch.rand(10,200) # batch_size, time_steps

In  [4]: gaussian_filter = DeepGaussianFilter(filter_size=11, sigma=1., order=2)
                
In  [5]: y = gaussian_filter(x)

In  [6]: y.shape
Out [6]: torch.Size([10, 3, 200])

```

<img src="https://github.com/cidl-auth/deep_gaussian/blob/main/figures/input_output.png" width="600" height="600" />

## Citation
If you use this code or find our work otherwise useful, please consider citing our paper:
```
@inproceedings{nalmpantis2023end,
  title={End-to-End Trainable Gaussian Filtering for Electrocardiogram Signal Classification Using Deep Learning},
  author={Nalmpantis, Angelos and Passalis, Nikolaos and Tefas, Anastasios},
  booktitle={2023 31st European Signal Processing Conference (EUSIPCO)},
  pages={1060--1064},
  year={2023},
  organization={IEEE}
}
```
