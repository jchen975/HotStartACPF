# DC-Power-Flow-Correction
Correcting DC power flow results with a MLP, written entirely in Julia. Work still in progress: currently only considering load demand variations in data generation. 

`case_general_parallel.jl`: no longer maintained and originally meant to be a single source file to run everything at once. 

`load_data.jl` performs 1) sampling from P, Q variations, 2) pf computation in parallel and 3) saving and loading existing data
`train.jl` performs 1) training an MLP with provided dataset, 2) plotting results and 3) saving and loading existing model

To find the case files, please see https://github.com/MATPOWER/matpower and https://github.com/lanl-ansi/PowerModels.jl/tree/master/test/data/matpower. 
