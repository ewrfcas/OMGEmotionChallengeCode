import pandas as pd
import numpy as np
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# best_epoch:91
# best_a_ccc: 0.16(0.19)
# best_v_ccc: 0.2516(0.2633)

# fin
# best_epoch:196
# best_a_ccc:0.8370
# best_v_ccc:0.8449

def correct(train_y, pred_val_y):
    train_std = np.std(train_y)
    val_std = np.std(pred_val_y)
    mean = np.mean(pred_val_y)
    pred_val_y = np.array(pred_val_y)
    pred_val_y = mean + (pred_val_y - mean) * train_std / val_std
    return pred_val_y

# return data & length
def padding_context(df, context=0, max_length=None):
    if max_length==None:
        max_length=27+13*context
    data=np.zeros((df.shape[0],max_length))
    videos=df['video'].values
    df=df['index'].values
    df=list(map(lambda x:json.loads(x),df))
    df_origin=df.copy()
    # add context
    if context!=0:
        for i in range(context,data.shape[0]):
            for j in range(1,context+1):
                if videos[i-j]==videos[i]:
                    df[i]=df_origin[i-j]+df[i]
    
    # zero padding
    for i,d in enumerate(df):
        if len(d)<=max_length:
            data[i,0:len(d)]=d
        else:
            data[i,:]=d[0:max_length]
    length=[]
    data=data.astype(np.int32)
    for i in range(data.shape[0]):
        length.append(len(np.where(data[i,:]!=0)[0]))
    length=np.array(length)
    
    return data,length

import pandas as pd
import numpy as np
import calccc

import pandas as pd
import numpy as np

df_train=pd.read_csv('dataset/train_label.csv')
df_test=pd.read_csv('dataset/val_label.csv')

# split train/val
sp=pd.read_csv('dataset/val_id.csv',header=None).values
sp=np.unique(np.array(list(map(lambda x:x[0].split('_')[0],sp))))
train_index=[]
val_index=[]
videos=df_train['video'].values
for i,v in enumerate(videos):
    if v.split('_')[0] in sp:
        val_index.append(i)
    else:
        train_index.append(i)
df_val=df_train.iloc[val_index,:]
df_train=df_train.iloc[train_index,:]
df_train.head()

y_train=df_train[['arousal','valence']].values
y_val=df_val[['arousal','valence']].values
y_test=df_test[['arousal','valence']].values

import pickle
pkl_file = open('dataset/mono_waveforms_omg_tr.pkl','rb')
data = pickle.load(pkl_file)
pkl_file.close()

from tqdm import tqdm
context=0
max_length=600000
# train
X_train=[]
videos=df_train['video'].values
utts=df_train['utterance'].values
for i in tqdm(range(len(videos))):
    x_temp=data[videos[i]+'_'+utts[i].split('.')[0]+'.wav']
    if len(x_temp)>=max_length:
        x_temp=x_temp[0:max_length]
    else:
        x_temp=np.concatenate((x_temp,np.zeros(((max_length-len(x_temp)),))))
    X_train.append(x_temp)
X_train=np.array(X_train)

# val
X_val=[]
videos=df_val['video'].values
utts=df_val['utterance'].values
for i in tqdm(range(len(videos))):
    x_temp=data[videos[i]+'_'+utts[i].split('.')[0]+'.wav']
    if len(x_temp)>=max_length:
        x_temp=x_temp[0:max_length]
    else:
        x_temp=np.concatenate((x_temp,np.zeros(((max_length-len(x_temp)),))))
    X_val.append(x_temp)
X_val=np.array(X_val)

import pickle
pkl_file = open('dataset/mono_waveforms_omg_val.pkl','rb')
data = pickle.load(pkl_file)
pkl_file.close()

# test
X_test=[]
videos=df_test['video'].values
utts=df_test['utterance'].values
for i in tqdm(range(len(videos))):
    x_temp=data[videos[i]+'_'+utts[i].split('.')[0]+'.wav']
    if len(x_temp)>=max_length:
        x_temp=x_temp[0:max_length]
    else:
        x_temp=np.concatenate((x_temp,np.zeros(((max_length-len(x_temp)),))))
    X_test.append(x_temp)
X_test=np.array(X_test)
X_train=(X_train/32768.)*256.
X_val=(X_val/32768.)*256.
X_test=(X_test/32768.)*256.

# split to 1

X_train=np.concatenate((X_train,X_val,X_test),axis=0)
y_train=np.concatenate((y_train,y_val,y_test))

X_val=X_train
y_val=y_train

del data
import gc
gc.collect()
print(X_train.shape,X_test.shape)

import aud_CNN_model as ACM
from keras.optimizers import *
epoch_num=200
batch_size=64
timesteps=max_length
model=ACM.model(timesteps)
# optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.99, epsilon=1e-08, decay=0.00015)
optimizer = SGD(lr=0.003,momentum=0.9,nesterov=True,decay=0.001)
model.compile(loss='mae', optimizer=optimizer)

# load weights
# model.load_weights('models/AudCNN_context'+str(context)+'_weights_fin.h5')

best_ccc=0
best_epoch=0
early_stop=100
no_improve=0

n_batch = int(len(X_train) / batch_size)
for epoch in range(epoch_num):
    # training step
    index=np.arange(X_train.shape[0])
    np.random.shuffle(index)
    X_train=X_train[index,:]
    y_train=y_train[index,:]
    sum_loss=0
    last_train_str=""
    for i in range(n_batch):
        x=X_train[i*batch_size:(i+1)*batch_size,:]
        y1=y_train[i*batch_size:(i+1)*batch_size,0]
        y2=y_train[i*batch_size:(i+1)*batch_size,1]
        
        loss_value=model.train_on_batch(x,[y1,y2])
        sum_loss+=(loss_value[0])
        last_train_str = "\r[epoch:%d/%d, steps:%d/%d]-loss:%.4f" % \
              (epoch+1, epoch_num, i+1, n_batch, sum_loss/(i+1))
        print(last_train_str, end='      ', flush=True)
    
    # validating for mse & ccc
    prediction=model.predict(X_val)
    prediction_a=np.reshape(prediction[0],(1,-1))[0]
    prediction_v=np.reshape(prediction[1],(1,-1))[0]
    a_ccc, _ = calccc.ccc(y_val[:,0], prediction_a)
    a_ccc2,_ = calccc.ccc(y_val[:,0], correct(y_train[:,0], prediction_a))
    v_ccc, _ = calccc.ccc(y_val[:,1], prediction_v)
    v_ccc2,_ = calccc.ccc(y_val[:,1], correct(y_train[:,1], prediction_v))
    last_val_str=" [validate]-accc:%.4f(%.4f)-vccc:%.4f(%.4f)" % (a_ccc,a_ccc2,v_ccc,v_ccc2)
    print(last_train_str+last_val_str, end='      ', flush=True)
    if epoch>=99 and (a_ccc+v_ccc)/2.>best_ccc:
        best_ccc=(a_ccc+v_ccc)/2.
        best_epoch=epoch+1
        model.save_weights('models/AudCNN_weights_fin.h5')
#     prediction=model.predict(X_test)
#     prediction_a=np.reshape(prediction[0],(1,-1))[0]
#     prediction_v=np.reshape(prediction[1],(1,-1))[0]
#     a_ccc, _ = calccc.ccc(y_test[:,0], prediction_a)
#     a_ccc2,_ = calccc.ccc(y_test[:,0], correct(y_train[:,0], prediction_a))
#     v_ccc, _ = calccc.ccc(y_test[:,1], prediction_v)
#     v_ccc2,_ = calccc.ccc(y_test[:,1], correct(y_train[:,1], prediction_v))
#     print(last_train_str+last_val_str+" [test]-accc:%.4f(%.4f)-vccc:%.4f(%.4f)" % (a_ccc,a_ccc2,v_ccc,v_ccc2),\
#               end='      ', flush=True)
    print('\n')
print('best_ccc:',best_ccc)
print('best_epoch:',best_epoch)