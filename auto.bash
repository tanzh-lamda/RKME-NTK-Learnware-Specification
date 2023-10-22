#!/usr/bin/env bash

# shellcheck disable=SC1090
source ~/anaconda3/etc/profile.d/conda.sh
conda activate nt

param='--mode auto --n_uploaders 50 --n_users 50 -K 48 --auto_param data_id --spec rbf --data cifar10'
folder="$(date +%s)"
mkdir -p "./log/${folder}"
echo "The output is redirected to log/${folder} with token ${folder}"

num=8
if [ $# -eq 1 ]
  then
    num=$1
fi

for ((i=0;i<num;i++))
do
# shellcheck disable=SC2086
nohup python main.py --id ${i} --cuda_idx ${i} ${param} > "./log/${folder}/auto_${i}.log" 2>&1 &
echo $! >> "./log/${folder}/.save_pid"
done