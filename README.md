# [ICASSP2025] Binary Stochastic Flip Optimization for Training Binary Neural Networks
---
This is official code for "Binary Stochastic Flip Optimization for Training Binary Neural Networks" published in ICASSP2025.

## Abstract
For deploying deep neural networks on edge devices with limited resources, binary neural networks (BNNs) have attracted significant attention, due to their computational and memory efficiency.
However, once a neural network is binarized, finetuning it on edge devices becomes challenging because most conventional training algorithms for BNNs are designed for use on centralized servers and require storing real-valued parameters during training.
To address this limitation, this paper introduces binary stochastic flip optimization (BinSFO), a novel training algorithm for BNNs.
BinSFO employs a parameter update rule based on Boolean operations, eliminating the need to store real-valued parameters and thereby reducing memory requirements and computational overhead.
In experiments, we demonstrated the effectiveness and memory efficiency of BinSFO in finetuning scenarios on six image classification datasets.
BinSFO performed comparably to conventional training algorithms
with a 70.7% smaller memory requirement.


## Preparation
The code requires pre-trained weights of AdamBNN and BCDNet. In addition, if using Caltech-101/256, Oxford Pets/Flowers datasets, the code requires that they are downloaded. These assets should be stored under `${PATH}/model` or `${PATH}/dataset` and `${PATH}` should be listed in `.storage_path.txt`. The pre-trained weights are available on their official GitHub.
- Requirements
  - python 3.8.9 and requirements.txt

## How to use
The model training process, which includes the proposed method, is defined in `main.py`, and can be executed as follows with the command:
```
mpirun -np ${num_gpu} python main.py --batch_size=${batchsize} --dataset=${dataset} --epochs=${epocgs} --lr=${lr} --model=${model} --algorithm=${training_algorithm} --test
```
Note that due to differences in operating environments, the performance obtained may vary from the values reported in the literature.

### Training Algorithm
You can use each method used in the paper by setting the arguments `--training_space`, `--mask_type`, and `--scheduler_type` as below:

| Algorithm        | `--algorithm` |
| ------------------- | ------------------ |
| BinSFO | `binary`           |
| Real-space SGD/Adam   | `real`             |
| Binary optimizer   | `bop`             |

| Hypermask            | `--mask_type`  |
| --------------------- | -------------  |
| EWM mask  | `EWM`  |
| WPM mask | `WPM` |
| Random mask        | `RAND`       |

| Adaptive Temperature Scheduler  | `--scheduler_type`  |
| --------------------- | -------------  |
| Constant              | `const`        |
| Auto Scheduler              | `gaussian`        |
| Auto Scheduler using XNOR-Nets              | `scaled-gaussian`        |

### Architecture
You can select models from `AdamBNN`, `BCDNet`, and `MLP`. If using MLP, the arguments `--hid_dim` and `--depth` should be specified.

### Dataset

| Dataset       | `--dataset`  |
| ------------- | ------------ |
| MNIST         | `MNIST`        |
| CIFAR-10      | `CIFAR10`      |
| CIFAR-100     | `CIFAR100`     |
| Caltech-101      | `Caltech101`      |
| Caltech-256      | `Caltech256`      |
|Oxford Flowers| `OxfordFlowers`|
|Oxford Pets| `OxfordPets`|
