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

module load nixpkgs/16.09 gcc/7.3.0 cuda/10.0.130 cudnn/7.6 julia/1.3.0 # enable julia, CUDA

export LD_LIBRARY_PATH="${EBROOTCUDA}/lib64:${LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH="${EBROOTCUDNN}/lib64:${LD_LIBRARY_PATH}"

casefiles=("case30" "case118");
T=("0.1" "0.15" "0.2" "0.25" "0.3");
for c in ${casefiles[*]}
do
    for t in ${T[*]}
    do
        echo ">> T = $t"  >> run_train.log 2>&1
        julia train.jl "$c" "$t" "conv" "2" "retrain" "" >> run_train.log 2>&1
    done
done


T2=("0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8")
for t in ${T2[*]}
do
    echo ">> T = $t"  >> run_train.log 2>&1
    julia train.jl "case2869pegase" "$t" "conv" "2" "retrain" "" >> run_train.log 2>&1
done

for t in ${T2[*]}
do
    echo ">> T = $t"  >> run_train.log 2>&1
    julia train.jl "case300" "$t" "conv" "2" "retrain" "failmode" >> run_train.log 2>&1
done


echo "------------------------------------------------------------------------" >> run_train.log
echo >> run_train.log
echo "************ Finished program: $(date) ************" >> run_train.log
echo >> run_train.log
echo "------------------------------------------------------------------------" >> run_train.log
