#!/bin/sh
#SBATCH --job-name=pretraining_for_Segmentation
#SBATCH --output=./pretraining/pretrained_endovit_models/EndoViT_for_Segmentation/output/out.txt # Standard output of the script (Can be absolute or relative path). %A adds the job id to the file name so you can launch the same script multiple times and get different logging files
#SBATCH --error=./pretraining/pretrained_endovit_models/EndoViT_for_Segmentation/output/err.txt
#SBATCH --time=0-96:00:00 # Limit on the total run time (format: days-hours:minutes:seconds)
#SBATCH --gres=gpu:1 # Number of GPUs if needed
#SBATCH --cpus-per-task=12 # Number of CPUs (Don't use more than 24 per GPU)
#SBATCH --mem=120G # Memory in GB (Don't use more than 126G per GPU)

# # activate corresponding environment
# source ~/miniconda3/etc/profile.d/conda.sh
# conda deactivate
# conda activate endovit

CURRENT=$(date +"%Y_%m_%d_%H_%M_%S")

echo $CURRENT

EXP_NAME=endovit_sagemaker_$CURRENT

# set output paths
OUT_DIR=../output/$EXP_NAME

echo "Output directory: $OUT_DIR"
SAVE_BEST_MODEL_AT=$OUT_DIR/endovit_seg.pth

#WANDB_RUN_NAME=pretraining_for_Segmentation

if [ ! -d ${OUT_DIR} ] ; then
    mkdir -p ${OUT_DIR}
fi

# set input paths
DATA_DIR=../sample-data
CONFIG_PATH=./pretraining_config_local.yml

# if you don't want to use wandb remove the last 2 arguments below
# NOTE: before using wandb you will have to log into wandb
python ./mae/main_pretrain.py \
    --config ${CONFIG_PATH} \
    --data_path ${DATA_DIR} \
    --val_data_path ${DATA_DIR} \
    --train_datasets_to_take sample_segmentation \
    --val_datasets_to_take sample_validation \
    --output_dir ${OUT_DIR} \
    --log_dir ${OUT_DIR}/tensorboard_logs \
    --save_best_model_at ${SAVE_BEST_MODEL_AT}
#    1>${OUT_DIR}/out.txt 2>${OUT_DIR}/err.txt   
#    --use_wandb \
#    --wandb_run_name ${WANDB_RUN_NAME} \