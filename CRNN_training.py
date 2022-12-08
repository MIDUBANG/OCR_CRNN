#!/usr/bin/env python
# coding: utf-8

# In[1]:


# # 1. 라이브러리 로딩

# In[6]:


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"



# How to check if the code is running on GPU or CPU?

import tensorflow as tf
import keras.backend.tensorflow_backend as tfback

print("tf.__version__ is", tf.__version__)
print("tf.keras.__version__ is:", tf.keras.__version__)

#GPU 사용
def _get_available_gpus():
    """Get a list of available gpu devices (formatted as strings).

    # Returns
        A list of available GPU devices.
    """
    #global _LOCAL_DEVICES
    if tfback._LOCAL_DEVICES is None:
        devices = tf.config.list_logical_devices()
        tfback._LOCAL_DEVICES = [x.name for x in devices]
    return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]

tfback._get_available_gpus = _get_available_gpus


# How to check if Keras is using GPU?

from keras import backend as K

K.tensorflow_backend._get_available_gpus()

# In[7]:


import numpy as np
import matplotlib.pyplot as plt
import os
import editdistance
import pickle
import time

from keras.optimizers import SGD, Adam

from crnn_model import CRNN
from crnn_data import InputGenerator
from crnn_utils import decode
from utils.training import Logger, ModelSnapshot


# # 2. Target Value(라벨) 생성

# ## (1) JSON file load

# In[8]:


import json

with open('./korean_printed_sentence.json', encoding='UTF8') as f:
  printed = json.load(f)


# ## (2) GTUtility 생성

# In[ ]:


import numpy as np
import json
import os

from ssd_data import BaseGTUtility

class GTUtility(BaseGTUtility):
    """Utility for COCO-Text dataset.

    # Arguments
        data_path: Path to ground truth and image data.
        validation: Boolean for using training or validation set.
        polygon: Return oriented boxes defined by their four corner points.
            Required by SegLink...
        only_with_label: Add only boxes if text labels are available.
    """
    
    def __init__(self, data_path, validation=False, polygon=False, only_with_label=True):
        test = False

        self.data_path = data_path
        gt_path = data_path
        image_path = os.path.join(data_path, 'printed/')
        self.gt_path = gt_path
        self.image_path = image_path
        self.classes = ['Background', 'Text']

        self.image_names = []
        self.data = []
        self.text = []
        
        with open(os.path.join(gt_path, 'korean_printed_sentence.json'), encoding='UTF8') as f:
            gt_data = json.load(f)

        image_list = list(data['id'] for data in printed['images'])

        for img_id in image_list: # images

            if len(img_id) > 0:
                image_name = next((item['file_name'] for item in gt_data['images'] if item['id'] == img_id), None)
                img_width = next((item['width'] for item in gt_data['images'] if item['id'] == img_id), None)
                img_height = next((item['height'] for item in gt_data['images'] if item['id'] == img_id), None)

                boxes = []
                text = []
                
                box = np.array([0, 0, 1, 1])
                boxes.append(box)

                txt = next((item['text'] for item in gt_data['annotations'] if item['id'] == img_id), None)
                text.append(txt)
                    
                if len(boxes) == 0:
                    continue
                
                boxes = np.asarray(boxes)
                    
                # append classes
                boxes = np.concatenate([boxes, np.ones([boxes.shape[0],1])], axis=1)
                
                self.image_names.append(image_name)
                self.data.append(boxes)
                self.text.append(text)

                print(image_name)
                
        self.init()

    def randomSplit(self, split=0.8):
      gtu1 = BaseGTUtility()
      gtu1.gt_path = self.gt_path
      gtu1.image_path = self.image_path
      gtu1.classes = self.classes

      gtu2 = BaseGTUtility()
      gtu2.gt_path = self.gt_path
      gtu2.image_pth = self.image_path
      gtu2.classes = self.classes

      n = int(round(split * len(self.image_names)))

      idx = np.arange(len(self.image_names))

      np.random.seed(0)

      np.random.shuffle(idx)

      train = idx[:n]
      val = idx[n:]

      gtu1.image_names = [self.image_names[t] for t in train]
      gtu2.image_names = [self.image_names[v] for v in val]

      gtu1.data = [self.data[t] for t in train]
      gtu2.data = [self.data[v] for v in val]

      if hasattr(self, 'text'):
        gtu1.text = [self.text[t] for t in train]
        gtu2.text = [self.text[v] for v in val]

      gtu1.init()
      gtu2.init()

      return gtu1, gtu2


if __name__ == '__main__':
    gt_util = GTUtility('./', validation=False, polygon=True, only_with_label=True)
    print(gt_util.data)



# ## (3) 생성된 객체의 text 값 가져오기

# In[6]:


text = gt_util.text

text


# ## (4) 여러 개의 리스트를 하나의 리스트로

# In[7]:


from itertools import chain

text = list(chain(*text))


# ## (5) 리스트 안 문자열을 문자 단위로 잘라 counting

# In[8]:


from sklearn.feature_extraction.text import CountVectorizer

vect = CountVectorizer(analyzer='char').fit(text)


# In[9]:


vect.vocabulary_.keys()


# ## (6) dict_key 타입을 list 로

# In[10]:


charset = list(vect.vocabulary_.keys())


# ## (7) 한글 딕셔너리 생성

# In[11]:


import re

pattern = '[^가-힣]' #한글이 아닌 문자는 공백으로 바꿔준다
charset_dict = [re.sub(pattern, "", char) for char in charset]


# ## (8) chatset_dict에서는 공백 제거

# In[12]:


charset_dict2 = [x for x in charset_dict if x!= '']


# In[13]:


len(charset_dict2)


# ## (9) 리스트를 str 형태로

# In[14]:


to_str = "".join(charset_dict2)


# In[15]:


to_str


# In[16]:


# 최종
printed_dict = to_str


# In[17]:


# 최종 라벨
korean_dict = printed_dict + '0123456789' + ' ' + '.,:()[]<>"\'_'

korean_dict


# # 3. Train / Validation 나누기

# In[18]:


gt_util_train, gt_util_val = gt_util.split(0.8)


# In[19]:


# 잘 분리됐는지 확인
len(gt_util_train.data)


# # 4. Model Train

# ## (1)  Model - input parameter 정의

# In[20]:


input_width = 256
input_height = 32
batch_size = 128

input_shape = (input_width, input_height, 1)


# ## (2) Fine Tuning 위해 동결할 layer층

# In[21]:

'''
v1: conv1_1, conv2_1, conv3_1, conv3_2
v2: conv1_1, conv2_1
v3: conv1_1

[ver2]
v1: 동결층X
v2: conv1_1
'''


freeze = ['conv1_1',
          #'conv2_1',
          #'conv3_1', 'conv3_2', 
          #'conv4_1',
          #'conv5_1',
          #'conv6_1',
          #'lstm1',
          #'lstm2'
         ]


# ## (3) 모델이 저장될 버전의 이름 정의

# In[22]:


model, model_pred = CRNN(input_shape, len(korean_dict))
experiment = 'crnn_korean_2_v2'


# ## (4) input을 위한 generator 생성

# In[23]:


alphabet = korean_dict

alphabet


# In[24]:


max_string_len = model_pred.output_shape[1]
print(str(max_string_len))

# InputGenerator 코드 수정 -> grayscale을 gray_scale로 고쳤다 다시 grayscale로
gen_train = InputGenerator(gt_util_train, batch_size, alphabet, input_shape[:2], 
                           grayscale=True, max_string_len=max_string_len, concatenate=False)
gen_val = InputGenerator(gt_util_val, batch_size, alphabet, input_shape[:2], 
                         grayscale=True, max_string_len=max_string_len, concatenate=False)


# ## (5) 이미 학습한 가중치를 모델에 로딩 -> Transfer Learning

# In[ ]:


# 처음 학습이므로 지금은 제외
model.load_weights('./CRNN_weights_2_v1.h5')

# ## (6) 학습 과정 저장

# In[25]:


checkdir = './checkpoints/' + time.strftime('%Y%m%d%H%M') + '_' + experiment
if not os.path.exists(checkdir):
    os.makedirs(checkdir)
'''
with open(checkdir+'/source.py','wb') as f:
    source = ''.join(['# In[%i]\n%s\n\n' % (i, In[i]) for i in range(len(In))])
    f.write(source.encode())
'''

# ## (7) Optimizer 설정

# In[26]:


'''
v1: SGD, learning_rate=0.001
v2: SGD, learning_rate=0.0001
v3: Adam, learning_rate=0.0005 -> loss 너무 높아져서 다시 SGD로, learning_rate=0.0005

[ver2]
v1: learning_rate = 0.0005
v2: learning_rate = 0.0001
'''
optimizer = SGD(learning_rate=0.0001, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
#optimizer = Adam(lr=0.0005, epsilon=0.001,decay=1e-5, clipnorm=1.)


# ## (8) 위에서 선언한 Layer들의 가중치 동결

# In[27]:

'''
for layer in model.layers:
    layer.trainable = not layer.name in freeze
'''

# ## (9) model 컴파일

# In[28]:


model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizer)


# ## (10) model 학습

# In[ ]:


'''
[ver2]
v1: 3000 -> 524 epochs
'''


from keras.callbacks import ModelCheckpoint, EarlyStopping

hist = model.fit(gen_train.generate(), # batch_size here?
                steps_per_epoch=gt_util_train.num_objects // batch_size,
                epochs=3000,
                validation_data=gen_val.generate(), # batch_size here?
                validation_steps=gt_util_val.num_objects // batch_size,
                callbacks=[
             ModelCheckpoint(checkdir+'/weights.{epoch:03d}.h5', verbose=1, save_weights_only=True),
             #ModelSnapshot(checkdir, 100),
             Logger(checkdir),
            EarlyStopping(monitor='val_loss', mode='auto', restore_best_weights=True, verbose=1, patience=25)
          ],
          initial_epoch=0)


# # 5. Train Result

# ## (1) 학습 결과(loss) 그래프로 확인

# In[ ]:


loss = hist.history['loss']
val_loss = hist.history['val_loss']

epochs = range(len(loss))
plt.figure(figsize=(15,10))
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b',label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


# ## (2) 학습 결과(accuracy) 그래프로 확인

# In[ ]:





# # 6. 학습한 Model 저장

# ## [Method 1] 모델 자체를 저장

# 모델 전체를 파일로 저장하고, 불러오는 방법

# In[ ]:


model.save('CRNN_model_2_v2.h5') #저장될 디렉토리의 경로 저장

#new_model = tf.keras.models.load_model('/content/sample_data/checkpoints/202211281921_crnn_korean_v1/weights.021900.h5')

#test_loss, test_acc = new_model.evaluate(x,  y, verbose=2)


# ## [Method 2] 가중치를 저장

# 가중치만 파일로 저장하고, 불러오는 방법

# In[ ]:


model.save_weights('CRNN_weights_2_v2.h5')

'''
new_model= tf.keras.models.Sequential([
    tf.keras.layers.Dense(32, input_dim=x.shape[1]),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(len(np.unique(y)), activation='softmax')
])

new_model.load_weights('iris_weight')
test_loss, test_acc = new_model.evaluate(x,  y, verbose=2)
'''
