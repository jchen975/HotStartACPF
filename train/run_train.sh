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

module load nixpkgs/16.09 gcc/7.3.0 cuda/10.0.130 cudnn/7.6 julia/1.2.0 # enable julia, CUDA

export LD_LIBRARY_PATH="${EBROOTCUDA}/lib64:${LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH="${EBROOTCUDNN}/lib64:${LD_LIBRARY_PATH}"

# julia cudatest.jl >> run_train.log
# julia train.jl "case30" "0.2" "conv" "Y" "Y" >> run_train.log
julia train.jl "case118" "0.1" "conv" "Y" "Y" >> run_train.log 2>&1
# julia train.jl "case2869pegase" "0.2" "conv" "Y" "Y" >> run_train.log

# casefiles=("case30" "case89pegase" "case145" "case118" "case145" "case2869pegase");
# T=("0.1" "0.15" "0.2" "0.25" "0.3");
# Lambda=("1.0" "2.5" "5.0" "7.5" "10.0");
#
# for case in ${casefiles[*]}
# do
#     echo "********* Training models for $case at $(date) **********" >> run_train.log
#     for t in ${T[*]}
#     do
#         for l in ${Lambda[*]}
#         do
#             echo "  >> T = $t, lambda = $l"  >> run_train.log
#             # julia train.jl "$case" "$t" "" >> run_train.log
#         done
#     done
# done

echo "------------------------------------------------------------------------" >> run_train.log
echo >> run_train.log
echo "************ Finished program: $(date) ************" >> run_train.log
echo >> run_train.log
echo "------------------------------------------------------------------------" >> run_train.log
