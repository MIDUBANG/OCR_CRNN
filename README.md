# ✏️ OCR_CRNN
> CRNN 기반의 OCR 한글 모델 Training 

- __CRNN_training.py__: ~ 모델 학습 및 모델(가중치) 저장
- __CRNN_train_test.ipynb__: ~ 모델 학습 및 모델(가중치) 저장, loss 추이 그래프, 모델 test

*CRNN_train_test.ipynb* 를 통해 코드를 구성하였고,
<br>
최종 모델 학습은 *CRNN_train_test.ipynb* 를 바탕으로 한 *CRNN_training.py* 를 터미널에서 실행해 백그라운드에서도 학습이 진행될 수 있게 하였습니다.

실제 학습은 *CRNN_training.py* 코드를 실행하면 되고,
<br>
모델 및 코드 테스트는 *CRNN_train_test.ipynb* 코드를 실행하면 됩니다.

<br>

본 README는 *CRNN_train_test.ipynb* 기준으로 작성되었습니다.

<br>
<br>


# 0. 디렉토리 구조
다음과 같은 디렉토리 구조로 이루어져 있습니다.

```
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

<br>

# 1. 패키지 설치하기

```bash
$ pip install -r requirements.txt
```
위 커맨드를 실행해 모델 학습에 필요한 패키지를 설치합니다.
*CRNN_train_test.ipynb*에 해당 코드가 포함되어 있지 않기 때문에 따로 실행해주어야 합니다.

<br>

# 2. 학습 데이터 구축

https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100
![image](https://user-images.githubusercontent.com/48647199/206628317-10d44c2d-e5c4-4f93-89cd-73eeb2aaa202.png)

학습 데이터는 AI Hub의 '한국어 글자체 이미지' 데이터셋을 이용했습니다.
<br>
해당 데이터셋은 **손글씨**, **인쇄체**, **실사 데이터** 로 구성되어 있고, 이 중 **인쇄체** 데이터는 _글자_, _단어_, _문장_  데이터로 구분되어 있습니다.

프로젝트에서 OCR 모델에 input으로 들어갈 데이터가 계약서이기 때문에,
<br>
현재까지 학습에선 **인쇄체** 데이터 중 _문장_  데이터 40,304개를 사용해 학습을 진행했습니다.
<br>
<br>
추후 학습은 _인쇄체_문장 데이터에 대한 Data Augmentation_ 이나 _손글씨 데이터_, 그리고 _계약서 이미지를 바탕으로 전처리를 진행한 custom data_ 를 이용해 fine tuning 및 transfer learning을 진행해 모델의 loss를 더 줄일 생각입니다.

 
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

# 5. Dataset Split

```python
gt_util_train, gt_util_val = gt_util.split(0.8)
```
Train : Validation = 8 : 2 의 비율로 나눠줍니다.
<br>
비율을 변경하고 싶으면 `gt_util.split()` 함수 안의 파라미터를 원하는 train set 비율로 설정해주면 됩니다.

<br>
<br>

# 6. OCR 모델 학습

<br>

## (1) Model의 input parameter 정의
```python
input_width = 256
input_height = 32
batch_size = 128

input_shape = (input_width, input_height, 1)
```

input 이미지의 `width`와 `height`, 그리고 `batch size`를 설정합니다.
<br>
본 학습에서의 input은 문장 데이터였기 때문에 width를 height보다 크게 설정해주었습니다.
<br>
batch size는 본인의 학습 환경이나 모델 성능에 따라 변경해주면 됩니다

<br>

## (2) 동결 Layer층 설정
```python
freeze = ['conv1_1',
          'conv2_1',
          'conv3_1', 'conv3_2', 
          #'conv4_1',
          #'conv5_1',
          #'conv6_1',
          #'lstm1',
          #'lstm2'
         ]
```
fine tuning을 위해 동결할 Layer층을 설정해줍니다.
이 또한 모델 성능에 맞게 조절해주면 됩니다.

<br>

## (3) 모델 정의 및 학습 모델의 version명 정의
```python
model, model_pred = CRNN(input_shape, len(korean_dict))
experiment = 'crnn_korean_test'
```
실제 학습에선 version명은 'crnn_korean_v1', 'crnn_korean_v2' 등으로 바꿔가며 설정해주었습니다.

<br>

## (4) InputGenerator 생성
```python
max_string_len = model_pred.output_shape[1]

gen_train = InputGenerator(gt_util_train, batch_size, korean_dict, input_shape[:2], 
                           grayscale=True, max_string_len=max_string_len, concatenate=False)
gen_val = InputGenerator(gt_util_val, batch_size, korean_dict, input_shape[:2], 
                         grayscale=True, max_string_len=max_string_len, concatenate=False)
```

<br>

## (5) 가중치 loading
```python
model.load_weights('./CRNN_weights_2_v2.h5')
```
이전에 진행했던 학습의 가중치를 load해 transfer learning을 진행합니다.
<br>
학습 중에 저장한 가중치를 불러와도 되고, 따로 저장한 가중치를 불러와도 됩니다.
<br>
본 코드에선 따로 저장한 가중치를 load해왔습니다.

<br>

## (6) 모델 학습 과정 저장
```python
checkdir = './checkpoints/' + time.strftime('%Y%m%d%H%M') + '_' + experiment
if not os.path.exists(checkdir):
    os.makedirs(checkdir)

with open(checkdir+'/source.py','wb') as f:
    source = ''.join(['# In[%i]\n%s\n\n' % (i, In[i]) for i in range(len(In))])
    f.write(source.encode())
```
위에서 설정한 모델 version명을 토대로 directory를 생성해 학습 과정을 저장합니다.
<br>
<br>
만약 *CRNN_train_test.ipynb* 가 아닌 *CRNN_training.py* 로 학습을 진행한다면,
<br>
하단의 코드 블럭은 삭제해야 합니다.

<br>

## (7) Optimizer 설정
```python
optimizer = SGD(learning_rate=0.0001, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
```
Optimizer를 설정합니다.
<br>
본 코드에선 optimizer로 `SGD`를 사용했으나, 필요에 따라 `Adam`과 같은 다른 모델도 사용할 수 있습니다.
<br>
만일 다른 모델을 사용할 경우, 따로 코드를 구현하거나 라이브러리를 로드해야 합니다.
<br>

`learning rate`는 0.001부터 0.0001까지 값을 변경해가면서 학습을 진행했습니다.
<br>
본인의 상황에 맞게 값을 변경하면서 사용하면 됩니다.
<br>
본 학습에선 따로 값을 변경하지 않았으나, 필요한 경우 `decay`나 `momentum`의 값을 변경할 수도 있습니다.

<br>

## (8) (2)에서 설정한 Layer층의 가중치 동결
```python
for layer in model.layers:
    layer.trainable = not layer.name in freeze
```

<br>

## (9) 모델 Compile
```python
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizer)
```
loss 모델로는 `ctc loss`를 사용하였으나, 이 또한 변경 가능합니다.

<br>

## (10) 모델 학습
```python
from keras.callbacks import ModelCheckpoint, EarlyStopping

hist = model.fit(gen_train.generate(), 
                steps_per_epoch=gt_util_train.num_objects // batch_size,
                epochs=1000,
                validation_data=gen_val.generate(), 
                validation_steps=gt_util_val.num_objects // batch_size,
                callbacks=[
             ModelCheckpoint(checkdir+'/weights.{epoch:03d}.h5', verbose=1, save_weights_only=True),
             #ModelSnapshot(checkdir, 100),
             Logger(checkdir),
            EarlyStopping(monitor='val_loss', mode='auto', restore_best_weights=True, verbose=1, patience=20)
          ],
          initial_epoch=0)
```
모델 학습을 진행합니다.
<br>
`epochs` 값을 변경할 수 있습니다.
<br>
callback 함수들 또한 설정해주었는데,
<br>
`ModelCheckpoint` 함수를 통해 하나의 epoch가 끝날 때마다 해당 epoch의 모델의 가중치를 저장해주었고,
<br>
`EarlyStopping` 함수를 통해 20 epochs 동안 validation loss가 감소하지 않는다면 더이상 학습을 진행할 필요가 없다고 판단해 학습을 중단하도록 했습니다.

<br>
해당 셀 구동 시, 
<br>
<br>

***'[ WARN:6@537.712] global /io/opencv/modules/imgcodecs/src/loadsave.cpp (239) findDecoder imread_('./printed/03384889.png'): can't open/read file: check file path/integrity'***

<br>
와 같은 warning meassage가 뜨는데,
<br>
이는 JSON 파일에서 문장 데이터가 아닌 다른 이미지 데이터들에 대한 정보가 제거되지 않았기 때문에 뜨는 메세지로, 무시하면 됩니다.
<br>
JSON 파일에 대한 작업을 진행했으나 완벽하게 정리되지 않아 추후 이 부분을 보완할 생각입니다.

<br>

# 7. 학습 결과 확인
<br>

## Loss graph 확인

```python
loss = hist.history['loss']
val_loss = hist.history['val_loss']

epochs = range(len(loss))
plt.figure(figsize=(15,10))
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b',label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
```

해당 코드를 통해 epoch에 따른 loss값의 추이를 확인할 수 있다.


<br>

# 8. 학습 모델 저장
<br>
## (1) Model 저장
```python
model.save('CRNN_model_test.h5')
```
학습한 model 자체를 저장합니다. 파라미터에는 모델이 저장될 경로와 파일명을 설정해주면 됩니다.

<br>
## (2) Weight 저장
```python
model.save_weights('CRNN_weights_test.h5')
```
학습한 model의 weight(가중치)를 저장합니다. 이 또한 파라미터에는 가중치가 저장될 경로와 파일명을 설정해주면 됩니다.

<br>

# 9. 학습 모델 Test
<br>

## (1) 한글 설정
```python
import matplotlib as mpl

# 유니코드 깨짐현상 해결
mpl.rcParams['axes.unicode_minus'] = False

# 나눔고딕 폰트 적용
plt.rcParams["font.family"] = 'NanumGothic'
```

model 적용 결과가 한글이기 때문에 유니코드 깨짐 현상을 해결하였고,
<br>
학습 환경에 따라 폰트가 깨져서 나오기도 해 따로 폰트를 적용해주었다.

<br>

## (2) 데이터 test
```python
g = gen_val.generate()
d = next(g)

res = model_pred.predict(d[0]['image_input'])

mean_ed = 0
mean_ed_norm = 0

plot_name = 'crnn_korean'

for i in range(32):
    chars = [alphabet[c] for c in np.argmax(res[i], axis=1)]
    gt_str = d[0]['source_str'][i]
    res_str = decode(chars)
    
    ed = editdistance.eval(gt_str, res_str)
    ed_norm = ed / len(gt_str)
    mean_ed += ed
    mean_ed_norm += ed_norm
    
    img = d[0]['image_input'][i][:,:,0].T
    plt.figure(figsize=[10,1.03])
    plt.imshow(img, cmap='gray', interpolation=None)
    ax = plt.gca()
    #plt.text(0, 45, '%s' % (''.join(chars)) )
    plt.text(0, 60, 'GT: %-24s RT: %-24s %0.2f' % (gt_str, res_str, ed_norm))
    
    plt.show()
```

본 코드에선 학습에 사용하지 않은 validation set에서 이미지 데이터를 가져와 test를 진행했습니다.
<br>
추후 validation set에 없는 custom data 또한 적용해 결과를 확인할 수 있도록 코드를 작성할 계획입니다.

<br>

# 10. 최종 디렉토리 구조
학습 후에는 디렉토리가 다음과 같이 형식으로 변경됩니다.

```
OCR_CRNN/
├── checkpoints/
│     ├── 202211302056_crnn_korean_v1
│     │     ├── history.csv
│     │     ├── log.csv
│     │     ├── weights.001.h5
│     │     ├── weights.002.h5
│     │     ├── ...
│     │     └── weights.227.h5
│     ├── 202212011003_crnn_korean_v2
│     │     ├── history.csv
│     │     ├── log.csv
│     │     ├── weights.001.h5
│     │     ├── weights.002.h5
│     │     ├── ...
│     │     └── weights.037.h5
│     ├── ...
│     └── 202212061250_crnn_korean_2_v1
│           ├── history.csv
│           ├── log.csv
│           ├── weights.001.h5
│           ├── weights.002.h5
│           ├── ...
│           └── weights.524.h5
│
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
├── CRNN_model_test.h5
├── CRNN_weights_2_v1.h5
├── CRNN_weights_2_v2.h5
├── CRNN_weights_test.h5
├── crnn_data.py
├── crnn_model.py
├── crnn_utils.py
├── ssd_data.py
├── korean_printed_sentence.json
├── NanumBarunGothic.ttf
└── requirements.txt
```

<br>
<br>

<hr>

# Reference
https://github.com/mvoelk/ssd_detectors
