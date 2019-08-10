# DC-Power-Flow-Correction
Correcting DC power flow results with an MLP, written entirely in Julia. Work still in progress: currently only considering load demand variations in data generation. 

`load_data.jl` performs 1) sampling from P, Q variations, 2) pf computation in parallel and 3) saving and loading existing data.

`train.jl` performs 1) training an MLP with provided dataset, 2) plotting results and 3) saving and loading existing model.

`run_load.jl` and `run_train.jl`: run `load_data.jl` and `train.jl` on command line; bash scripts committed but will be changed to test different parameters

`hot_start_acpf.jl`: run acpf again with hot start values from dcpf + mlp inference. Only conducted basic tests for this file.

To find the case files, please see https://github.com/MATPOWER/matpower and https://github.com/lanl-ansi/PowerModels.jl/tree/master/test/data/matpower. 
