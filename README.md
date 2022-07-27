# Local-Mixup
This repo contains the code (notebooks, scripts) for Local Mixup, as presented in our paper "Preventing Manifold Intrusion with Locality: Local Mixup" (https://arxiv.org/abs/2201.04368).
We also provide the dataset of the two coiling spirals used in our paper. Note that this dataset can also be generated by the user with different values of noise. 

## Requirements and Installation:
1. A Pytorch installation https://pytorch.org
2. Python version 3.8.1 (lower versions might not work)
3. (optional) CIFAR10 dataset,CIAFAR100 dataset, SVHN, Fashion MNIST from https://pytorch.org/vision/stable/datasets.html.

## Experiments
### Low dimension (spirals dataset)
We provide:
- **Generate_Spiral_dataset.ipynb** used to generate the spirals dataset.    
- **Spiral_Experiment.ipynb** used to carried out the spirals dataset.
- **data\spirals_datasets** the folder that contains the spirals dataset.
    
### High Dimension
#### Lipschitz Lower Bound
We provide the script **cifar10_lipschitz.py** used to compute $Q(D)$ for Mixup, Local Mixup and Vanilla.
#### CIFAR10 Resnet18
We provide the notebook **cifar.ipynb** used to compute the error rates on Cifar10.
#### Fashion-MNIST DenseNet
We provide the script **fashionmnist.py** used to compute the error rates on Fashion MNIST.
#### SVHN Lenet
We provide the script **svhn.py** used to compute the error rates on MNIST.

