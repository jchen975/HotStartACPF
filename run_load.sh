#!/bin/bash

# overwrite existing run_load.log or create a new one
echo "------------------------------------------------------------------------" > run_load.log
echo >> run_load.log
echo "************ Starting program: $(date) ************" >> run_load.log
echo "CPU Info" >> run_load.log
lscpu | grep -i "Model name" >> run_load.log  # print computer info to log file
echo >> run_load.log
echo "------------------------------------------------------------------------" >> run_load.log

module load gcc/7.3.0 julia/1.1.1  # enable julia

# From run_load.jl:
# Running on command line (assuming load_data.jl is in current directory):
# julia run_load.jl <case name> <number of samples> <number of workers> <force generating new dataset Y/N>

N=10000
nprocs=30
casefiles=("case30" "case118" "case300" "case2869" "case13659")

for case in ${casefiles[*]}
do
    julia run_load.jl "$case" "$N" "$nprocs" Y >> run_load.log
done

echo "------------------------------------------------------------------------" >> run_load.log
echo >> run_load.log
echo "************ Finished program: $(date) ************" >> run_load.log
echo >> run_load.log
echo "------------------------------------------------------------------------" >> run_load.log
