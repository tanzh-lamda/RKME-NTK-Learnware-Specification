#!/usr/bin/env bash

# shellcheck disable=SC1090
source ~/anaconda3/etc/profile.d/conda.sh
conda activate learnware_remote

param='--mode auto --n_uploaders 50 --n_users 50 -K 50'
folder="$(date +%s)"
mkdir -p "./log/${folder}"

for num in {0..6}
do
# shellcheck disable=SC2086
nohup python main.py --id ${num} ${param} > "./log/${folder}/auto_${num}.log" 2>&1 &
echo $! > ./log/save_pid.txt
done