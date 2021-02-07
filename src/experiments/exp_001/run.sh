export CUDA_VISIBLE_DEVICES=0,1,2
root=$(dirname $(dirname $(pwd)))
name=$(basename $(pwd))
python main.py exp_param.WORK_DIR=$(pwd) exp_param.ROOT=${root} exp_param.exp_name=${name} > log.txt
