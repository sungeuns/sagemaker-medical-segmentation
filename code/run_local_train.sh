
# - Note
# - if matplot error: https://github.com/ultralytics/yolov5/issues/11417#issuecomment-1518920181
# - pip install matplotlib --no-cache-dir --force-reinstall

# source /home/ec2-user/anaconda3/bin/activate JupyterSystemEnv
# conda activate pytorch_p310

# Run train job into the local
python local_train.py --epochs 5 --lr 0.001 --seed 123 --model-dir "../local_model" --data-dir "../Spleen3D/processed/train" --num-gpus 1

