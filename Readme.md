## SageMaker with medical image segmentation

해당 코드는 [monai](https://github.com/Project-MONAI/MONAI) 를 활용하여 medical image 를 ML 알고리즘으로 학습하고 추론하는 것을 SageMaker로 수행하는 예시입니다. 노트북 순서대로 진행하면 됩니다.

- 원본 코드는 [여기](https://github.com/aws-samples/amazon-sagemaker-medical-imaging-with-monai/blob/main/Segmentation/MONAI_BYOS_spleen_segmentation_3D_Demo.ipynb) 를 참고하여 수정하였습니다.


### 01. 데이터 준비

- 샘플 데이터셋을 다운로드 받고, local 환경에서 학습을 진행 해 봅니다.
- `code/local_train.py` 를 참고해 주세요. (실행 예시 : `code/run_local_train.sh`)
- terminal 에서 아래 명령어를 활용하여 쉽게 EC2 서버처럼 사용할 수 있습니다.

```
source /home/ec2-user/anaconda3/bin/activate JupyterSystemEnv
conda env list
conda activate pytorch_p310
```

### 02. Sagemaker managed training을 활용한 학습

- 로컬 환경에서 하는 학습을 SageMaker에서 진행하는 것으로 테스트 해 봅니다.


### 03. SageMaker endpoint를 활용한 배포 및 테스트

- 학습이 완료된 model 을 배포하고 테스트를 진행합니다.

### 참고사항

- `lifecycle_configuration` 에 있는 파일들을 sagemaker notebook에 적용하여 여러 추가적인 기능들을 활용해 볼 수 있습니다.

