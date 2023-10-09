# FLatS: Principled Out-of-Distribution Detection with Feature-Based Likelihood Ratio Score

This repository contains the code for our paper [FLatS: Principled Out-of-Distribution Detection with Feature-Based Likelihood Ratio Score](#) by [Haowei Lin](https://linhaowei1.github.io/) and [Yuntian Gu]([github.com](https://github.com/guyuntian)).

## Quick Links

- [Overview](#overview)
- [Requirements](#requirements)
- [Training](#training)
- [Evaluation](#evaluation)
- [Bugs or Questions?](#bugs-or-questions)
- [Acknowledgements](acknowledgements#)
- [Citation](#citation)

## Overview

Detecting out-of-distribution (OOD) instances is crucial for NLP models in practical applications. Although numerous OOD detection methods exist, most of them are empirical. Backed by theoretical analysis, this paper advocates for the measurement of the "OOD-ness" of a test case $\boldsymbol{x}$ through the **likelihood ratio** between out-distribution $\mathcal P_{\textit{out}}$ and in-distribution $\mathcal P_{\textit{in}}$. We argue that the state-of-the-art (SOTA) feature-based OOD detection methods, such as Maha and KNN, are suboptimal since they only estimate in-distribution density $p_{\textit{in}}(\boldsymbol{x})$. To address this issue, we propose **FLatS**, a principled solution for OOD detection based on likelihood ratio. Moreover, we demonstrate that FLatS can serve as a general framework capable of enhancing other OOD detection methods by incorporating out-distribution density $p_{\textit{out}}(\boldsymbol{x})$ estimation. Experiments show that FLatS establishes a new SOTA on popular benchmarks. 

## Requirements

First, install PyTorch by following the instructions from [the official website](https://pytorch.org/). Please use the correct 1.6.0 version corresponding to your platforms/CUDA versions to faithfully reproduce our results. PyTorch version higher than `1.6.0` should also work. For example, if you use Linux and **CUDA11** ([how to check CUDA version](https://varhowto.com/check-cuda-version/)), install PyTorch by the following command,

```
pip install torch==1.6.0+cu110 -f https://download.pytorch.org/whl/torch_stable.html
```

If you instead use **CUDA** `<11` or **CPU**, install PyTorch by the following command,

```
pip install torch==1.6.0
```

Then run the following script to install the remaining dependencies,

```
pip install -r requirements.txt
```

We use [faiss](https://github.com/facebookresearch/faiss) to run fast K-nearest search algorithm, 

## Training

In the following section, we describe how to train the TPLR model by using our code.

**Data**

Before training and evaluation, please download the datasets (CIFAR-10, CIFAR-100, TinyImageNet). The default working directory is set as ``~/data`` in our code. You can modify it according to your need.

**Pre-train Model**

We use the pre-train DeiT model provided by [MORE](https://github.com/k-gyuhak/MORE). Please download it and save the file as ``./deit_pretrained/best_checkpoint.pth``

**Training scripts**

We provide all the example training scripts to run TPLR. e.g., for C10-5T, train the network using this command:

```bash
bash scripts/deit_C10_5T.sh
```

For the results in the paper, we use Nvidia GeForce RTX2080Ti GPUs with CUDA 10.2. Using different types of devices or different versions of CUDA/other software may lead to slightly different performance.

## Evaluation

Continue with the C10-5T example. Once you finished training, come back to the root directory and simply run this command:

```bash
bash scripts/deit_C10_5T_eval.sh
```

The results for the first sequence with `seed=2023` will be saved in `./data/seq0/seed2023/progressive_main_2023`.

## Bugs or questions?

If you have any questions related to the code or the paper, feel free to email [Haowei](mailto:linhaowei@pku.edu.cn). If you encounter any problems when using the code, or want to report a bug, you can open an issue. Please try to specify the problem with details so we can help you better and quicker!

## Acknowledgements

We thank this repo: [Code for OOD Detection Research in NLP](https://github.com/lancopku/Avg-Avg) for providing an extendable framework for continual learning. We use their code structure as a reference when developing this code base.

## Citation

Please cite our paper if you use this code or part of it in your work:

```bibtex
@inproceedings{lin2023flats,
      title={FLatS: Principled Out-of-Distribution Detection with Feature-Based Likelihood Ratio Score}, 
      author={Lin, Haowei and Gu, Yuntian},
      booktitle={Empirical Methods in Natural Language Processing (EMNLP)},
      year={2023}
}
```