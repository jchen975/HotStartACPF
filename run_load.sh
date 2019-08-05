#!/bin/bash

# overwrite existing run_load.log or create a new one
echo "------------------------------------------------------------------------" > run_load.log
echo "************ Starting program: $(date) ************" >> run_load.log
echo "------------------------------------------------------------------------" >> run_load.log

# From run_load.jl:
# Running on command line (assuming load_data.jl is in current directory):
# julia run_load.jl <case name> <number of samples> <number of workers> <force generating new dataset Y/N>

lscpu >> run_load.log  # print computer info to log file

N=10000
nprocs=30

julia run_load.jl case30 $N $nprocs Y >> run_load.log
julia run_load.jl case118 $N $nprocs Y >> run_load.log
julia run_load.jl case300 $N $nprocs Y >> run_load.log
julia run_load.jl case2869 $N $nprocs Y >> run_load.log
julia run_load.jl case13659 $N $nprocs Y >> run_load.log

echo "------------------------------------------------------------------------" >> run_load.log
echo "************ Finished program: $(date) ************" >> run_load.log
echo "------------------------------------------------------------------------" >> run_load.log
