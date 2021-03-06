Finding hot start conditions for AC Power Flow based on Newton-Raphson Algorithm with 1D CNNs. Preprint: https://arxiv.org/abs/2004.09342v1

With DCPF as input, the trained 1D CNNs predict hot start conditions (voltage magnitude and phase at each bus) that minize NR iterations and solution time. Power flow code written in Matlab with Matpower, and CNN implementation and training written in Julia. Currently only considering load demand variations in data generation. 

`pf/` contains Matlab code for 1) creating and sampling from P, Q variations, 2) pf (dc, ac cold start, ac hot start) computations and 3) saving and loading existing data. Replace the default `runpf.m` and `newtonpf.m` with the ones here to return PQ mismatches as `mpc.mismatch`.

`train/` contains Julia code for 1) implementing and training an 1D CNN with provided dataset (and bash script for training on remote clusters) and 2) saving and loading existing model.

`old/` contains old power flow code in Julia, and old NN implementations (MLP) and hyperparameter options. Now unmaintained.

Case files from https://github.com/MATPOWER/matpower.
I will rewrite the `pf` code in Julia, once a Julia native power flow solver package is finished.
