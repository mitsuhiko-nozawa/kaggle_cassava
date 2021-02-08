root=$(dirname $(dirname $(dirname $(pwd))))
name=$(basename $(pwd))
python main.py \
exp_param.WORK_DIR=$(pwd) \
exp_param.ROOT=${root} \
exp_param.exp_name=${name} \
hydra.run.dir=. hydra.output_subdir=null hydra/job_logging=disabled hydra/hydra_logging=disabled

#nohup sh -c 'while ps -p 40199 > /dev/null; do sleep 60; done; sh run.sh &