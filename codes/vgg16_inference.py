from keras.models import *
from keras.layers import *
from keras_vggface.vggface import VGGFace
from keras.preprocessing.image import *
from keras.optimizers import *
from keras.callbacks import *
from keras.applications import *
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
VGGFace_resnet50_model=VGGFace(model='vgg16', include_top=False, input_shape=(256, 256, 3), pooling='avg')
for layer in VGGFace_resnet50_model.layers:
    layer.trainable=False
    
input_tensor = Input((256, 256, 3))
outputs = VGGFace_resnet50_model(input_tensor)
model = Model(input_tensor, outputs, name='vgg16')

from keras.preprocessing import image
from keras_vggface.vggface import VGGFace
from keras_vggface import utils
import numpy as np
#生成器generator
def generator_test(file_list, batch_size):
    if batch_size>len(file_list):
        batch_size=len(file_list)
    while True:
        count = 0
        x = []
        for i,path in enumerate(file_list):
            x_temp = image.load_img(path)
            x_temp = image.img_to_array(x_temp)
            x_temp = np.expand_dims(x_temp, axis=0)
            x_temp= utils.preprocess_input(x_temp, version=2)
            count += 1
            x.append(x_temp)
            if count % batch_size == 0 and count != 0:
                x = np.array(x)
                x = x.reshape(batch_size, 256, 256, 3)
                yield x
                x = []
                
import os
from tqdm import tqdm
import pandas as pd
ids=pd.read_csv('/users/seria/TensorFlood/data/seeta_omg/omg_seeta_sub/omg_submit.csv',header=None)[0].values
# ids=np.array(list(map(lambda x:x[0].split(' ')[0],ids)))
idict={}

def name_correct(n):
    temp_n=n.split('_')
    new_names=None
    assert len(temp_n)==5 or len(temp_n)==4
    if len(temp_n)==5:
        new_names=temp_n[0]+'_'+temp_n[1]+'_'+temp_n[3]
    elif len(temp_n)==4:
        new_names=temp_n[0]+'_'+temp_n[2]
    return new_names

for i in ids:
    t=name_correct(i)
    if t not in idict:
        idict[t]=['/users/seria/TensorFlood/data/seeta_omg/omg_seeta_sub/'+i]
    else:
        idict[t].append('/users/seria/TensorFlood/data/seeta_omg/omg_seeta_sub/'+i)

predict_path='dataset/new_submit_vgg16_vec/'
for k in tqdm(idict):
    df_cnn=model.predict_generator(generator_test(idict[k],16),(len(idict[k])//16)+1,verbose=0)
    df_cnn=df_cnn[0:len(idict[k]),:]
    df_cnn=pd.DataFrame(df_cnn)
    df_cnn.to_csv(predict_path+k+'.csv',index=None)