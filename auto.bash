source ~/anaconda3/etc/profile.d/conda.sh
conda activate learnware_remote

comm='--mode auto --n_uploaders 50 --n_users 50 -K 50'

for num in {0..6}
do
nohup python main.py --id ${num} ${comm} > "./log/auto_${num}.log" 2>&1 &
done