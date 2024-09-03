
## Training EndoVit on SageMaker

원본 코드
- EndoVit: https://github.com/DominikBatic/EndoViT


### 워크샵 진행

- 01-prepare-sample-data.ipynb
  - 아래 `데이터 준비` 및 `모델 준비` 과정을 직접 하지 않고 쉽게 진행하려면 해당 노트북의 cell 을 실행하면 됩니다.
- 02-endovit-training-on-sagemaker.ipynb
  - 데이터 준비 후 SageMaker의 managed training을 사용하여 학습을 진행 해 봅니다.


### 데이터 준비

워크샵 진행은 segmentation 용도 데이터만 활용하며, 미리 준비된 샘플 데이터를 사용합니다.
- tar.gz 파일을 받고 압축을 풀고, S3에 파일을 업로드합니다.

```
mkdir -p sample-data; cd sample-data

# Training sample 다운로드
wget https://[CF_DIST_ID].cloudfront.net/endo-vit/sample_segmentation.tar.gz
wget https://[CF_DIST_ID].cloudfront.net/endo-vit/sample_validation.tar.gz

# 압축 해제
tar zxvf sample_segmentation.tar.gz
tar zxvf sample_validation.tar.gz

# training sample을 S3에 업로드 (1.3GB, 2560개)
aws s3 cp --recursive sample_segmentation/ s3://[prefix]/

# validation sample을 S3에 업로드 (141MB, 256개)
aws s3 cp --recursive sample_validation/ s3://[prefix]/
```

### 모델 준비

```
mkdir -p ./pt-models

# Pretrainig weight
wget -O ./pt-models/mae_pretrain_vit_base_full.pth https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base_full.pth

# CF 파일이 제공되면 아래와 같이 받을 수도 있음.
# cd pt-models; wget https://[CF_DIST_ID].cloudfront.net/endo-vit/mae_pretrain_vit_base_full.pth

# S3에 모델 업로드
aws s3 cp mae_pretrain_vit_base_full.pth s3://[prefix]/
```


### 로컬 Training 테스트 진행

로컬 테스트를 위한 환경을 구성합니다.

```
source /home/ec2-user/anaconda3/bin/activate JupyterSystemEnv
conda create --name endovit-sagemaker python=3.9
conda activate endovit-sagemaker
cd endovit-code; pip install -r requirements.txt
```

원본 repo에 있는 패키지 리스트를 설치해도 되지만 시간이 오래 걸리기 때문에 필요한 패키지만 설치합니다. `requirements.txt`에는 아래 패키지를 사용합니다.

```
torch==1.13.0
torchvision==0.14.0
timm==1.0.8
wandb==0.17.6
ConfigArgParse==1.7
numpy==1.23.1
tensorboard==2.10.1
torchmetrics==0.10.1
matplotlib==3.6.1
```

training을 진행하기 전에 파라미터가 적합하게 들어갔는지 확인하고, 학습을 진행합니다.

```
./pretrain_script_local.sh
```

만일 Multi-GPU를 활용한 distributed training을 하려면 아래 스크립트로 가능합니다.

```
./pretrain_script_local_multigpu.sh
```

### SageMaker에서 학습 진행

- `02-endovit-training-on-sagemaker.ipynb` 노트북을 실행해서 SageMaker training job을 테스트해 볼 수 있습니다.
- SageMaker에서 실행되는 스크립트는 `pretrain_script_sm.sh` 파일을 참고합니다.
- `instance_type` 에서 GPU가 1개인 instance를 사용하면 자동으로 단일 GPU 기반 학습이 진행되며 GPU가 여러 개인 instance를 사용하면 자동으로 Multi-GPU 기반 학습이 진행됩니다.



---------


## 기존 코드 테스트


### Prepare datasets

SageMaker notebook에서 terminal 을 열고 아래와 같이 데이터를 준비합니다.

```
git clone https://github.com/DominikBatic/EndoViT
cd EndoViT

source /home/ec2-user/anaconda3/bin/activate JupyterSystemEnv

# 전체 의존성을 맞추다보니 ENV 생성하는데 시간 오래 걸림.
conda env create -f conda_environment.yml

# conda env list
conda activate endovit

# datasets/Cholec80 에 데이터 다운로드 (~97GB)
python ./datasets/Cholec80/download_cholec80.py --data_rootdir ./datasets/

# datasets/Endo700k 에 데이터 준비 (~230GB)
python ./datasets/Cholec80/prepare_cholec80.py

```

준비된 데이터 통계

```
Summary:
----------
               Cholec80_for_Segmentation:     178054
     Cholec80_for_ActionTripletDetection:     169555
   Cholec80_for_SurgicalPhaseRecognition:      86304
                 Cholec80_for_Validation:      57071
```


### Prepare pre-trained model

```
mkdir -p ./pretraining/mae/ImageNet_pretrained_models

# Pretrainig weight
wget -O ./pretraining/mae/ImageNet_pretrained_models/mae_pretrain_vit_base_full.pth https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base_full.pth

# Fine-tuning weight
wget -O ./pretraining/mae/ImageNet_pretrained_models/mae_pretrain_vit_base.pth https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth
```


### Start pre-training

Segmentation 관련 예시

```
conda activate endovit
pip install configargparse timm wandb

./pretraining/pretrained_endovit_models/EndoViT_for_Segmentation/pretrain_script_dev.sh
```

Tensorboard log 확인 방법
```
pip install tensorboard
tensorboard --logdir [로그 경로]

# 접속 주소
https://[NOTEBOOK_NAME].notebook.[REGION].sagemaker.aws/proxy/6006/

# 예시
https://sm-notebook-g5.notebook.us-west-2.sagemaker.aws/proxy/6006/
```
