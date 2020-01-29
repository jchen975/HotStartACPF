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

case="case118";
T1=("2000" "3000" "4000" "5000" "6000");
for t in ${T1[*]}
do
    echo ">> T = $t"  >> run_train.log 2>&1
    julia train.jl "$case" "$t" "retrain" "" >> run_train.log 2>&1
done

case="case2869pegase";
T2=("3000" "4000" "5000" "6000" "7000" "8000" "9000" "10000" "11000" "12000");
for t in ${T2[*]}
do
   echo ">> T = $t"  >> run_train.log 2>&1
   julia train.jl "$case" "$t" "retrain" "" >> run_train.log 2>&1
done

case="case300";
T3=("2000" "3000" "4000" "5000" "6000" "7000" "8000" "9000" "10000");
for t in ${T3[*]}
do
    echo ">> T = $t"  >> run_train.log 2>&1
    julia train.jl "$case" "$t" "retrain" "failmode" >> run_train.log 2>&1
done


echo "------------------------------------------------------------------------" >> run_train.log
echo >> run_train.log
echo "************ Finished program: $(date) ************" >> run_train.log
echo >> run_train.log
echo "------------------------------------------------------------------------" >> run_train.log
