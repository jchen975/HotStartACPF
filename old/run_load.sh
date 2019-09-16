#!/bin/bash

# overwrite existing run_load.log or create a new one
echo "------------------------------------------------------------------------" > run_load.log
echo >> run_load.log
echo "************ Starting program: $(date) ************" >> run_load.log
echo "CPU Info" >> run_load.log
lscpu | grep "Model name:" | sed -r 's/Model name:\s{1,}//g'  >> run_load.log
echo >> run_load.log
echo "------------------------------------------------------------------------" >> run_load.log

module load gcc/7.3.0 julia/1.1.0  # enable julia

# From load_data.jl:
# Running on command line (assuming load_data.jl is in current directory):
# julia load_data.jl <case name> <number of samples> <number of workers> <force generating new dataset Y/N>

N=10000
# nprocs=40
# casefiles=("case30" "case118" "case118" "case2869")

julia -p 40 load_data.jl case118 "$N" Y >> case118_load.log
julia -p 40 load_data.jl case300 "$N" Y >> case300_load.log
julia -p 40 load_data.jl case2869 "$N" Y >> case2869_load.log


# for case in ${casefiles[*]}
# do
#     # julia load_data.jl "$case" "$N" "$nprocs" Y >> run_load.log
#     julia load_data.jl "$case" "$N" 3 Y >> run_load.log
#     echo >> run_load.log
#     julia load_data.jl "$case" "$N" 4 Y >> run_load.log
#     echo >> run_load.log
#     julia load_data.jl "$case" "$N" 7 Y >> run_load.log
#     echo >> run_load.log
#     julia load_data.jl "$case" "$N" 8 Y >> run_load.log
#     echo >> run_load.log
#     julia load_data.jl "$case" "$N" 15 Y >> run_load.log
#     echo >> run_load.log
#     julia load_data.jl "$case" "$N" 23 Y >> run_load.log
#     echo >> run_load.log
#     julia load_data.jl "$case" "$N" 31 Y >> run_load.log
#     echo >> run_load.log
# done

echo "------------------------------------------------------------------------" >> run_load.log
echo >> run_load.log
echo "************ Finished program: $(date) ************" >> run_load.log
echo >> run_load.log
echo "------------------------------------------------------------------------" >> run_load.log
