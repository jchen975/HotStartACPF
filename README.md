# DC-Power-Flow-Correction
Finding good hot-start values for AC power flow with an MLP and DC power flow results, written in Matlab and Julia. Currently only considering load demand variations in data generation. 

`./pf/` contains Matlab code for 1) creating and sampling from P, Q variations, 2) pf (dc, ac cold start, ac hot start) computation and 3) saving and loading existing data. Replace the default `runpf.m` and `newtonpf.m` with the ones here to return PQ mismatches as `mpc.mismatch`.

`./train` contains Julia code for 1) training an MLP with provided dataset (and bash script for training on remote clusters), 2) plotting results and 3) saving and loading existing model.

`./old/` contains old power flow code in Julia. 

Case files from https://github.com/MATPOWER/matpower.
