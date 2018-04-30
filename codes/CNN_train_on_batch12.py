import CNN_model as CN
import RNN_model as R
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.utils import *
import keras.backend as K
import numpy as np
import os
from keras.metrics import *
import calccc
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# best epoch:27

# fin
# best_epoch:99
# best_a_ccc:0.7435
# best_v_ccc:0.8380

def huber_loss(y_true, y_pred, delta=0.5):
    return tf.losses.huber_loss(y_true,y_pred,delta=delta) 

def correct(train_y, pred_val_y):
    train_std = np.std(train_y)
    val_std = np.std(pred_val_y)
    mean = np.mean(pred_val_y)
    pred_val_y = np.array(pred_val_y)
    pred_val_y = mean + (pred_val_y - mean) * train_std / val_std
    return pred_val_y

import pandas as pd
#生成器generator with padding
def generator_test(file_list, batch_size, shuffle=False, random_seed=None, max_length=64):
    while True:
        if shuffle:
            if random_seed!=None:
                random_seed+=1
                np.random.seed(random_seed)
            index=np.arange(file_list.shape[0])
            np.random.shuffle(index)
            file_list=file_list[index]
        count = 0
        x, y = [], []
        for i,path in enumerate(file_list):
            x_temp= pd.read_csv(path)
            x_temp=x_temp.values
            if x_temp.shape[0]<max_length:
                x_temp=np.concatenate((x_temp,np.zeros((max_length-x_temp.shape[0],x_temp.shape[1]))),axis=0)
            else:
#                 x_temp=x_temp[0:max_length,:]
#                 # 固定间隔取帧
#                 index_temp = np.linspace(0, x_temp.shape[0], max_length, dtype=int, endpoint=False)
#                 x_temp=x_temp[index_temp,:]

                # 固定取5帧间隔
                index_temp=np.arange(x_temp.shape[0]//5)*5
                if len(index_temp)>=max_length:
                    index_temp=index_temp[0:max_length]
                    x_temp=x_temp[index_temp,:]
                else:
                    x_temp=x_temp[index_temp,:]
                    x_temp=np.concatenate((x_temp,np.zeros((max_length-x_temp.shape[0],x_temp.shape[1]))),axis=0)
            count += 1
            x.append(x_temp)
            if count % batch_size == 0 and count != 0:
                x = np.array(x)
                x = x.reshape(batch_size, max_length, -1).astype("float32")
                yield x
                x = []         
import os
df_train=pd.read_csv('../OMG_text_aud/dataset/train_label.csv')
df_test=pd.read_csv('../OMG_text_aud/dataset/val_label.csv')

# split train/val
sp=pd.read_csv('../OMG_text_aud/dataset/val_id.csv',header=None).values
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

X_train1=df_train['video'].values
X_train2=df_train['utterance'].values
X_train=[]
for i in range(X_train1.shape[0]):
    X_train.append(X_train1[i]+'_'+X_train2[i].split('.')[0].split('_')[-1])
X_train=np.array(X_train)

X_val1=df_val['video'].values
X_val2=df_val['utterance'].values
X_val=[]
for i in range(X_val1.shape[0]):
    X_val.append(X_val1[i]+'_'+X_val2[i].split('.')[0].split('_')[-1])
X_val=np.array(X_val)

X_test1=df_test['video'].values
X_test2=df_test['utterance'].values
X_test=[]
for i in range(X_test1.shape[0]):
    X_test.append(X_test1[i]+'_'+X_test2[i].split('.')[0].split('_')[-1])
X_test=np.array(X_test)

# remove_list
remove_list=['cbb64e001_14','9354b482f_1_9','2c09795bb_1_40']
remove_index=[]
for i,id in enumerate(X_train):
    if id in remove_list:
        print('remove',id)
        remove_index.append(i)
remove_index=np.array(remove_index)

X_train=np.array(list(map(lambda x:'dataset/new_trainset_vgg16_vec/'+x+'.csv',X_train)))
X_val=np.array(list(map(lambda x:'dataset/new_valset_vgg16_vec/'+x+'.csv',X_val)))
X_test=np.array(list(map(lambda x:'dataset/new_testset_vgg16_vec/'+x+'.csv',X_test)))

y_train=np.delete(y_train, remove_index, axis=0)
X_train=np.delete(X_train, remove_index, axis=0)

# split to 1

X_train=np.concatenate((X_train,X_val,X_test),axis=0)
y_train=np.concatenate((y_train,y_val,y_test))

X_val=X_train
y_val=y_train

from keras.callbacks import *
epoch_num=90
batch_size=16
max_length=64
best_a_ccc=0
best_v_ccc=0
best_ccc=0
best_epoch=0
best_a_epoch=0
best_v_epoch=0
# parameter
hidden_size=256
timesteps=64
attention_size=32
filters=64

model=CN.model(timesteps, 512, hidden_size, filters, ac='sigmoid+tanh')
# model=R.model(timesteps,512,hidden_size)
# optimizer = Adam(lr=0.005, beta_1=0.5, beta_2=0.95, epsilon=1e-08)
optimizer = SGD(lr=0.005, momentum=0.5, nesterov=True, decay=0.001)
model.compile(loss='mae', optimizer=optimizer)

# # circle training
# model.load_weights('models/CNN_weights12.h5')

n_batch = int(len(X_train) / batch_size)
for epoch in range(epoch_num):
    # training step
    index=np.arange(len(X_train))
    np.random.shuffle(index)
    X_train=X_train[index]
    y_train=y_train[index,:]
    sum_loss=0
    last_train_str=""
    for i in range(n_batch):
        file_list=X_train[i*batch_size:(i+1)*batch_size]
        label_list=y_train[i*batch_size:(i+1)*batch_size,:]
        x, y1, y2 = [], [], []
        for j,path in enumerate(file_list):
            x_temp= pd.read_csv(path).values
            if x_temp.shape[0]<max_length:
                x_temp=np.concatenate((x_temp,np.zeros((max_length-x_temp.shape[0],x_temp.shape[1]))),axis=0)
            else:
#                 x_temp=x_temp[0:max_length,:]
#                 # 固定间隔取帧
#                 index_temp = np.linspace(0, x_temp.shape[0], max_length, dtype=int, endpoint=False)
#                 x_temp=x_temp[index_temp,:]

                # 随机取连续帧
                index_temp=np.arange(x_temp.shape[0]//5)*5
                rand_index=np.random.random(len(index_temp))*5
                index_temp=index_temp+rand_index.astype(int)
                if len(index_temp)>=max_length:
                    index_temp=index_temp[0:max_length]
                    x_temp=x_temp[index_temp,:]
                else:
                    x_temp=x_temp[index_temp,:]
                    x_temp=np.concatenate((x_temp,np.zeros((max_length-x_temp.shape[0],x_temp.shape[1]))),axis=0)
            x.append(x_temp)
            y1.append(label_list[j,0])
            y2.append(label_list[j,1])
        x = np.array(x)
        x = x.reshape(batch_size, max_length, -1).astype("float32")
        y1 = np.array(y1)
        y2 = np.array(y2)
        
        loss_value=model.train_on_batch(x,[y1,y2])
        sum_loss+=(loss_value[0])
        last_train_str = "\r[epoch:%d/%d, steps:%d/%d]-loss:%.4f" % \
              (epoch+1, epoch_num, i+1, n_batch, sum_loss/(i+1))
        print(last_train_str, end='      ', flush=True)
    
    # validating for mse & ccc
    prediction=model.predict_generator(generator_test(X_val, batch_size), X_val.shape[0]//batch_size+1, verbose=0)
    prediction_a=np.reshape(prediction[0],(1,-1))[0][0:X_val.shape[0]]
    prediction_v=np.reshape(prediction[1],(1,-1))[0][0:X_val.shape[0]]
    a_ccc, _ = calccc.ccc(y_val[:,0], prediction_a)
    a_ccc2,_ = calccc.ccc(y_val[:,0], correct(y_train[:,0], prediction_a))
    v_ccc, _ = calccc.ccc(y_val[:,1], prediction_v)
    v_ccc2,_ = calccc.ccc(y_val[:,1], correct(y_train[:,1], prediction_v))
    last_val_str=" [validate]-accc:%.4f(%.4f)-vccc:%.4f(%.4f)" % (a_ccc,a_ccc2,v_ccc,v_ccc2)
    print(last_train_str+last_val_str, end='      ', flush=True)
    if a_ccc+v_ccc>best_ccc and epoch>=69:
        best_ccc=v_ccc+a_ccc
        best_epoch=epoch+1
        model.save_weights('models/CNN_weights12_fin.h5')
#     prediction=model.predict_generator(generator_test(X_test, batch_size), X_test.shape[0]//batch_size+1, verbose=0)
#     prediction_a=np.reshape(prediction[0],(1,-1))[0][0:X_test.shape[0]]
#     prediction_v=np.reshape(prediction[1],(1,-1))[0][0:X_test.shape[0]]
#     a_ccc, _ = calccc.ccc(y_test[:,0], prediction_a)
#     a_ccc2,_ = calccc.ccc(y_test[:,0], correct(y_train[:,0], prediction_a))
#     v_ccc, _ = calccc.ccc(y_test[:,1], prediction_v)
#     v_ccc2,_ = calccc.ccc(y_test[:,1], correct(y_train[:,1], prediction_v))
#     print(last_train_str+last_val_str+" [test]-accc:%.4f(%.4f)-vccc:%.4f(%.4f)" % (a_ccc,a_ccc2,v_ccc,v_ccc2),\
#               end='      ', flush=True)
    print('\n')
print('best_ccc:',best_ccc)
print('best_epoch:',best_epoch)