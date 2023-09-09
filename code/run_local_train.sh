
# - Note
# - if matplot error: https://github.com/ultralytics/yolov5/issues/11417#issuecomment-1518920181

# source /home/ec2-user/anaconda3/bin/activate JupyterSystemEnv
# conda activate pytorch_p310


python local_train.py --epochs 5 --lr 0.001 --seed 123 --model-dir "../local_model" --data-dir "../Spleen3D/processed/train" --num-gpus 1

