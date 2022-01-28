Born-Infeld (BI) for AI: Energy-Conserving Descent (ECD) for Optimization
====
This repository contains the code for the BBI optimizer, introduced in the paper _Born-Infeld (BI) for AI: Energy-Conserving Descent (ECD) for Optimization_. [2201.11137](http://arxiv.org/abs/2201.11137).
It is implemented using Pytorch.

The repository also includes the code needed to reproduce all the experiments presented in the paper. In particular:

- The BBI optimizer is implemented in the file `inflation.py`.

- The jupyter notebooks with the synthetic experiments are in the folder `synthetic`. All the notebooks already include the output, and text files with results are also included in the folder. In particular
    - The notebook `ackley.ipynb` can be used to reproduce the results in Sec. 4.1.
    - The notebook `zakharov.ipynb` can be used to reproduce the results in Sec. 4.2.
    - The notebook `multi_basin.ipynb` can be used to reproduce the results in Sec. 4.3.

- The ML benchmarks described in Sec. 4.5 can be found in the folders `CIFAR` and `MNIST`. The notebooks already include some results that can be inspected, but not all the statistics that builds up the results in Table 2. In particular:
    - *CIFAR* : The notebook `CIFAR-notebook.ipynb` uses hyperopt to estimate the best hyperparameters for each optimizer and then runs a long run with the best estimated hyperparamers. The results can be analyzed with the notebook `analysis-cifar.ipynb`, which can also be used to generate more runs with the best hyperparameters to gather more statistics. The subfolder `results` already includes some runs that can be inspected.

    - *MNIST*: The notebooks `mnist_scan_BBI.ipynb` and `mnist_scan_SGD.ipynb` perform a grid scan using BBI and SGD, respectively and gather some small statistics. All the results are within the notebooks themselves.

- The PDE experiments can be run by running the script `script-PDE.sh` as
    ```
    bash script-PDE.sh
    ```
    This will solve the PDE outlined in Sec. 4.4 and App. C multiple times with the same initialization. The hyperparameters are also kept fixed and can be obtained from the script itself. In particular:
    - `feature 1` means that an L2 regularization is added to the loss.
    - `seed` specifies the seed, which fixes the initialization of the network. The difference between the different runs then is only due to the random bounces, which are not affected by this choice of the seed. 

    The folder `results` already includes some runs.
    The runs performed in this way are not noisy, i.e. the set of points sampled from the domain is kept fixed. To randomly change the points every "epoch" (1000 iterations), edit the file `experiments/PDE_PoissonD.py` by changing line 134 to `self.update_points = True`.


The code has been tested with Python 3.9, Pytorch 1.10, hyperopt 0.2.5. We ran the synthetic experiments and MNIST on a six-core i7-9850H CPU with 16 GB of RAM, while we ran the CIFAR and PDE experiments on a pair of GPUs. We tested both on a pair of NVIDIA GeForce RTX 2080 Ti and on a pair of NVIDIA Tesla V100-SXM2-16GB GPUs, coupled with 32 GB of RAM and AMD EPYC 7502P CPUs.

The Resnet-18 code (in `experiments/models`) and the `utils.py` helper functions are adapted from https://github.com/kuangliu/pytorch-cifar (MIT License).
    
