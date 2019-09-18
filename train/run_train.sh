#!/bin/bash

# overwrite existing run_train.log or create a new one
echo "------------------------------------------------------------------------" > run_train.log
echo >> run_train.log
echo "************ Starting program: $(date) ************" >> run_train.log
echo ">> CPU Info" >> run_train.log
lscpu | grep "Model name:" | sed -r 's/Model name:\s{1,}//g' >> run_train.log
echo >> run_train.log
echo ">> GPU Info" >> run_train.log
nvidia-smi --query-gpu=name --format=csv | grep -v "name" >> run_train.log
echo >> run_train.log
echo "------------------------------------------------------------------------" >> run_train.log

module load gcc/7.3.0 julia/1.1.1 cuda/10.0.130 cudnn/7.5 # enable julia, CUDA

# From run_train.jl:
# Running on command line (assuming train.jl is in current directory):
# julia run_train.jl <case name> <second hidden layer Y/N> <retrain Y/N> <learning rate> <epochs> <batch size>`
# julia run_train.jl <case name> -d

casefiles=("case30" "case118" "case300" "case2869")

for case in ${casefiles[*]}
do
    echo ">> Training model for $case at $(date)" >> run_train.log
    julia run_train.jl "$case" -d >> run_train.log
done

echo "------------------------------------------------------------------------" >> run_train.log
echo >> run_train.log
echo "************ Finished program: $(date) ************" >> run_train.log
echo >> run_train.log
echo "------------------------------------------------------------------------" >> run_train.log