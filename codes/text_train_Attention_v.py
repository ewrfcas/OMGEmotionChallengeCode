import pandas as pd
import numpy as np
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

#best epoch:118
#best v_ccc:0.1965(0.2190)

#best(256) epoch:192
#best v_ccc:0.2221

# fin
# best_epoch:150
# best_v_ccc:0.4665

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

# load data
df_train=pd.read_csv('dataset/text_train_twitter.csv')
df_test=pd.read_csv('dataset/text_val_twitter.csv')
y_train=pd.read_csv('dataset/train_label.csv')
y_test=pd.read_csv('dataset/val_label.csv')

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
y_val=y_train.iloc[val_index,:]
y_train=y_train.iloc[train_index,:]

y_train=y_train[['arousal','valence']].values
y_val=y_val[['arousal','valence']].values
y_test=y_test[['arousal','valence']].values

# only v
y_train=y_train[:,1]
y_val=y_val[:,1]
y_test=y_test[:,1]

# context and padding
context=0
max_length=27+13*context
X_train,len_train = padding_context(df_train,context)
X_val,len_val = padding_context(df_val,context)
X_test,len_test = padding_context(df_test,context)
print(X_train.shape, len_train.shape,y_train.shape)
print(X_val.shape, len_val.shape,y_val.shape)
print(X_test.shape, len_test.shape,y_test.shape)
print('context:',context)

# # split to 1

# X_train=np.concatenate((X_train,X_val,X_test),axis=0)
# len_train=np.concatenate((len_train,len_val,len_test))
# y_train=np.concatenate((y_train,y_val,y_test))

# X_val=X_train
# len_val=len_train
# y_val=y_train

# load embedding matrix
import gensim
print('loading word2vec...')
model=gensim.models.KeyedVectors.load_word2vec_format('../NLPData/word2vec_twitter_model.bin',binary=True,\
                                                        unicode_errors='ignore')
embedding_matrix=model.vectors
print(embedding_matrix.shape)
embedding_matrix=np.concatenate((np.zeros((1,embedding_matrix.shape[1])),embedding_matrix,\
                                 np.random.random((1,embedding_matrix.shape[1]))/2-0.25),axis=0)
print(embedding_matrix.shape)
print('loading over')

import text_Attention_model as TAM
from keras.optimizers import *
epoch_num=150
batch_size=64
timesteps=max_length
dim=embedding_matrix.shape[1]
head=8
head_size=64
model=TAM.model(timesteps, dim, embedding_matrix, head, head_size)
# optimizer = Adam(lr=0.005, beta_1=0.9, beta_2=0.98, epsilon=1e-08, decay=0.001)
optimizer = SGD(lr=0.005,momentum=0.5,nesterov=True,decay=0.001)
model.compile(loss='mae', optimizer=optimizer)

# circle training
# model.load_weights('models/TextAttention_context0_weights.h5')

# context0 v val:0.3590 test:0.1759(0.1972)
# context1 v val:0.3339 test:0.1780(0.2080)
# context2 v val:0.3215 test:0.0956(0.1060)
# context3 v val:0.3891 test:0.0922(0.1040)
# context4 v val:0.3196 test:0.1157(0.1371)
 
best_ccc=0
best_epoch=0
seed=2018
n_batch = int(len(X_train) / batch_size)
for epoch in range(epoch_num):
    # training step
    index=np.arange(X_train.shape[0])
    np.random.seed(seed+epoch)
    np.random.shuffle(index)
    X_train=X_train[index,:]
    len_train=len_train[index]
    y_train=y_train[index]
    sum_loss=0
    last_train_str=""
    for i in range(n_batch):
        x=X_train[i*batch_size:(i+1)*batch_size,:]
        l=len_train[i*batch_size:(i+1)*batch_size]
        y=y_train[i*batch_size:(i+1)*batch_size]
        loss_value=model.train_on_batch([x,l],y)
        sum_loss+=(loss_value)
        last_train_str = "\r[epoch:%d/%d, steps:%d/%d] - loss:%.4f" % \
              (epoch+1, epoch_num, i+1, n_batch, sum_loss/(i+1))
        print(last_train_str, end='      ', flush=True)
    
    # validating for mse & ccc
    prediction=model.predict([X_val,len_val])
    prediction_v=np.reshape(prediction,(1,-1))[0]
    v_ccc, _ = calccc.ccc(y_val, prediction_v)
    v_ccc2,_ = calccc.ccc(y_val, correct(y_train, prediction_v))
    last_val_str=" [validate] - vccc:%.4f(%.4f)" % (v_ccc,v_ccc2)
    print(last_train_str+last_val_str, end='      ', flush=True)
    if v_ccc>best_ccc and epoch>99:
        best_ccc=v_ccc
        best_epoch=epoch+1
        model.save_weights('models/TextAttention_weights_fin.h5')
    prediction=model.predict([X_test,len_test])
    prediction_v=np.reshape(prediction,(1,-1))[0]
    v_ccc, _ = calccc.ccc(y_test, prediction_v)
    v_ccc2,_ = calccc.ccc(y_test, correct(y_train, prediction_v))
    print(last_train_str+last_val_str+" [test] - vccc:%.4f(%.4f)" % (v_ccc,v_ccc2),\
              end='      ', flush=True)
    print('\n')
print('best_ccc:',best_ccc)
print('best_epoch:',best_epoch)