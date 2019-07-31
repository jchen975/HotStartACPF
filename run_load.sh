#!/bin/bash

echo "------------------------------------------------------------------------" >> run_load.log
echo "************ Starting program: $(date) ************" >> run_load.log
echo "------------------------------------------------------------------------" >> run_load.log

# From run_load.jl:
# Running on command line (assuming load_data.jl is in current directory):
# julia run_load.jl <case name> <number of samples> <number of workers> <force generating new dataset Y/N>

julia run_load.jl case30 12000 40 Y >> run_load.log
julia run_load.jl case118 12000 40 Y >> run_load.log
julia run_load.jl case300 12000 40 Y >> run_load.log
julia run_load.jl case2869 12000 40 Y >> run_load.log
julia run_load.jl case13659 12000 40 Y >> run_load.log

echo "------------------------------------------------------------------------" >> run_load.log
echo "************ Finished program: $(date) ************" >> run_load.log
echo "------------------------------------------------------------------------" >> run_load.log
