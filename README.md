# OCR_CRNN
CRNN 기반의 OCR 한글 모델 Training 

- CRNN_training.py: ~ 모델 학습 및 모델(가중치) 저장
- CRNN_train_test.ipynb: ~ 모델 학습 및 모델(가중치) 저장, loss 추이 그래프, 모델 test

CRNN_train_test.ipynb를 통해 코드를 구성하였고,
<br>
최종 모델 학습은 CRNN_train_test.ipynb를 바탕으로 한 CRNN_training.py를 터미널에서 실행해 백그라운드에서도 학습이 진행될 수 있게 하였습니다.

실제 학습은 CRNN_training.py 코드를 실행하면 되고,
<br>
모델 및 코드 테스트는 CRNN_train_test.ipynb 코드를 실행하면 됩니다.

<br>
본 README는 CRNN_train_test.ipynb 기준으로 작성되었습니다.

<br>
<br>


# 0. 디렉토리 구조
다음과 같은 디렉토리 구조로 이루어져 있습니다.

```bash
OCR_CRNN/
├── printed/
│     ├── 03343000.png
│     ├── 03343001.png
│     │   ...   
│     └── 03385349.png
│
├── utils/
│     ├── bboxes.py
│     ├── losses.py
│     ├── model.py
│     └── training.py           
│    
├── CRNN_train_test.ipynb
├── CRNN_training.py
├── CRNN_model_2_v1.h5
├── CRNN_model_2_v2.h5
├── CRNN_weights_2_v1.h5
├── CRNN_weights_2_v2.h5
├── crnn_data.py
├── crnn_model.py
├── crnn_utils.py
├── ssd_data.py
├── korean_printed_sentence.json
├── NanumBarunGothic.ttf
└── requirements.txt
```

# 1. 패키지 설치하기

```bash
$ pip install -r requirements.txt
```
위 커맨드를 실행해 모델 학습에 필요한 패키지를 설치합니다.
CRNN_train_test.ipynb에 해당 코드가 포함되어 있지 않기 때문에 따로 실행해주어야 합니다.

<br>

# 2. 학습 데이터 구축

https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100
![image](https://user-images.githubusercontent.com/48647199/206628317-10d44c2d-e5c4-4f93-89cd-73eeb2aaa202.png)

학습 데이터는 AI Hub의 '한국어 글자체 이미지' 데이터셋을 이용했습니다.
<br>
해당 데이터셋은 "손글씨", "인쇄체", "실사 데이터"로 구성되어 있고, 이 중 "인쇄체" 데이터는 '글자', '단어', '문장' 데이터로 구분되어 있습니다.

프로젝트에서 OCR 모델에 input으로 들어갈 데이터가 계약서이기 때문에,
<br>
현재까지 학습에선 '인쇄체' 데이터 중 '문장' 데이터 40,304개를 사용해 학습을 진행했습니다.
<br>
<br>
추후 학습은 '인쇄체_문장' 데이터에 대한 Data Augmentation이나 '손글씨' 데이터, 그리고 custom data를 바탕으로 fine tuning 및 transfer learning을 진행해 모델의 loss를 더 줄일 생각입니다.

 
<br>


# 3. GTUtility 객체 생성
미리 업로드 해놓은 이미지 파일들과 JSON 파일을 이용해 GTUtility 객체를 생성합니다.
<br>
이미지 개수가 많은 탓에 해당 셀은 약 3~4분정도 소요됩니다.



<br>

# 4. Target Value (라벨) 생성

생성된 GTUtility 객체를 이용해 target value(라벨)을 생성합니다.
<br>
해당 과정을 간략하게 설명하면 다음과 같습니다.
<br>

##### 1) GTUtility에서 text값을 가져옵니다.
##### 2) 1)에서 가져온 text들을 하나의 리스트로 합쳐줍니다.
##### 3) 리스트 안의 문자열을 문자 단위로 잘라주고, 딕셔너리에 넣어 중복을 제거합니다.
##### 4) 한글이 아닌 문자는 공백으로 바꿔주고, 다시 공백은 제거합니다.
##### 5) 리스트를 문자열 형태로 바꿔주고, 계약서에 자주 사용되는 공백, 숫자, .,:()[]<>"\'_ 등의 기호들을 문자열에 추가시켜줍니다.




<br>
<br>



<br>

# 5. 사용자 모델 학습(training)
여기서부턴 gpu 환경에서만 가능하다.

### 커맨드
```bash
$ cd deep-text-recognition-benchmark
$ CUDA_VISIBLE_DEVICES=0 python3 train.py 
  --train_data ../data/data_lmdb_release/training 
  --valid_data ../data/data_lmdb_release/validation 
  --select_data basic-skew 
  --batch_ratio 0.5-0.5 
  --Transformation TPS 
  --FeatureExtraction VGG 
  --SequenceModeling BiLSTM 
  --Prediction CTC 
  --data_filtering_off  
  --valInterval 100 
  --batch_size 128 
  --batch_max_length 50 
  --workers 6 
  --distributed 
  --imgW 400;
```
<br>

### 학습 옵션
train.py의 옵션을 커스텀해 학습 가능하다.
- `--train_data` : path to training dataset
- `--valid_data` : path to validation dataset
- `--select_data`: directories to use as training dataset(default = 'basic-skew')
- `--batch_ratio` 
- `--Transformation` : choose one - None|TPS
- `--FeatureExtraction`: choose one - VGG|RCNN|ResNet 
- `--SequenceModeling`: choose one - None|BiLSTM
- `--Prediction` : choose one - CTC|Attn
- `--data_filtering_off` : skip data filtering when creating LmdbDataset
- `--valInterval` : Interval between each validation
- `--workers` :  number of data loading workers
- `--distributed`
- `--imgW` : the width of the input image
- `--imgH` : the height of the input image

<br>

### 학습 결과
- ocr_dtrb/deep-text-recognition-benchmark/saved_models 디렉토리에 학습시킨 모델별 `log_train.txt`, `best_accuracy.pth`, `best_norem_ED.pth` 파일이 저장된다. 
- log_train.txt에서는 iteration마다 best_accuracy와 loss 값이 어떻게 변하는지 확인 가능하다.
![](https://velog.velcdn.com/images/goinggoing/post/3d5391c9-7d6f-48fd-b4e6-cda1d13ddf61/image.png)
- best_accuracy.pth 파일을 이용해 evaluation과 demo가 가능하다. 

<br>

# 5. 사용자 모델 테스트(evaluation)


- 본 학습에서는 training data : validation data = 2:1 비율로 설정했기 때문에 test data 생성과 테스트 과정을 생략했다. 
- test 과정을 진행하고 싶다면 1~3단계에서 테스트 데이터도 생성/가공하면 된다.
  
### 커맨드
```shell script
$ CUDA_VISIBLE_DEVICES=0 python3 test.py 
  --eval_data ../data/data_lmdb_release/evaluation 
  --benchmark_all_eval 
  --Transformation TPS 
  --FeatureExtraction VGG 
  --SequenceModeling None 
  --Prediction CTC 
  --saved_model saved_models/Test-TPS-VGG-None-CTC-Seed/best_accuracy.pth 
  --data_fil1tering_off 
  --workers 2 
  --batch_size 128 
  --imgW 400;
```
- 위 커맨드는 테스트 문장데이터로 lmdb 데이터셋을 생성하여 data_lmdb_release/evaluation 경로로 저장했다고 가정했다. 
- 가장 정확도가 높았던 학습 모델인 Test-TPS-VGG-None-CTC-Seed를 테스트에 사용했다. 다운로드 받아 사용해볼 수 있다. (용량이 커 구글 드라이브로 첨부) [Test-TPS-VGG-None-CTC-Seed](https://drive.google.com/file/d/16JvCdkkEKum7CaFH4TkAVu1YWMC3NQ9_/view?usp=sharing)
- 직접 학습시켜 새롭게 저장된 모델도 사용할 수 있다.

<br>

### 테스트 옵션
학습 시에 사용한 옵션들을 거의 동일하게 사용할 수 있다. 
- `--eval_data` : path to evaluation dataset
- `--benchmark_all_eval` : evaluate 3 benchmark evaluation datasets
- `--saved_model` : path to saved_model to evaluation

<br>

# 6. 학습 모델 데모

- `--Transformation`, `--FeatureExtraction`, `--SequenceModeling`, `--Prediction` 옵션을 이용해 각 스테이지에서 사용할 모듈을 결정한다. 
- 학습 시에 같은 모듈을 사용했더라도 설정한 옵션에 따라 accuracy와 loss가 다를 수 있다. 학습한 모델 중 데모를 시도할 모델은 `--saved_model` 옵션으로 지정할 수 있다.
- `--image_folder` 옵션으로 데모 쓰일 디렉토리 경로를 지정한다.

<br>

### 커맨드
```shell script
$ CUDA_VISIBLE_DEVICES=0 python3 demo.py 
  --Transformation TPS   
  --FeatureExtraction VGG   
  --SequenceModeling None   
  --Prediction CTC  
  --image_folder ../data/demo_image   
  --saved_model saved_models/Test-TPS-VGG-None-CTC-Seed/best_accuracy.pth;
```
- saved_models 디렉토리에 학습시킨 모델 중 가장 정확도 높았던 모델을  다운로드 받아 사용해볼 수 있다. (용량이 커 구글 드라이브로 첨부) [Test-TPS-VGG-None-CTC-Seed](https://drive.google.com/file/d/16JvCdkkEKum7CaFH4TkAVu1YWMC3NQ9_/view?usp=sharing)
- 예시는 saved_models/Test-TPS-VGG-None-CTC-Seed 디렉토리를 만들고 위의 모델을 다운 받아 이용한 데모이다. saved_models 디렉토리에 저장되는 학습 모델 경로로 지정하면 직접 학습시킨 다른 모델로도 가능하다.

- 데모를 위해 나눔고딕, 맑은 고딕, 굴림 폰트가 사용된 문장 이미지 데이터를 data/demo_image 디렉토리에 첨부해두었다. 다른 문장 이미지로도 가능하다. 




<br>

# Acknowledgements
[deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark) , [TextRecognitionDataGenerator](https://github.com/Belval/TextRecognitionDataGenerator)
