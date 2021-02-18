# 全実験のlogを出させてまとめる

pwd=$(pwd)

files=$(dirname $(pwd))"/experiments/*"
for filepath in $files; do
    name=$(basename $filepath)
    if [ "$name" = "_template" ]
    then 
        echo "is template"
    elif [ "$name" = "make_exp.sh" ]
    then
        echo "is make_exp.sh"
    elif [ "$name" = "pred_val.sh" ]
    then
        echo "is make_exp.sh"
    else
        cd $filepath
        echo $filepath
        root_=$(dirname $(dirname $(dirname $(pwd))))
        name_=$(basename $(pwd))
        python main.py \
        exp_param.WORK_DIR=$(pwd) \
        exp_param.ROOT=${root_} \
        exp_param.exp_name=${name_} \
        exp_param.debug=False \
        exp_param.train_flag=True \
        exp_param.infer_flag=False \
        exp_param.log_flag=False \
        train_param.epochs=1 \
        train_param.batch_size=64 \
        train_param.do_retrain=True \
        hydra.run.dir=. hydra.output_subdir=null hydra/job_logging=disabled hydra/hydra_logging=disabled
        cd $pwd
    fi

done
echo complete