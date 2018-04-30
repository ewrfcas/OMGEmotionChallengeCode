import tensorflow as tf
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import regularizers

def model(timesteps=64, dim=512, unit=256, filters=128, ac='sigmoid',mode='pred'):
    inputs = Input((timesteps, dim))
    x1 = Conv1D(filters, 5, strides=1, padding='valid', input_shape=(timesteps, dim),name='conv1')(inputs)
    x1 = BatchNormalization(name='bn1')(x1)
    x1 = Activation('relu')(x1)
    x1 = GlobalMaxPooling1D()(x1)

    x2 = Conv1D(filters, 4, strides=1, padding='valid', input_shape=(timesteps, dim),name='conv2')(inputs)
    x2 = BatchNormalization(name='bn2')(x2)
    x2 = Activation('relu')(x2)
    x2 = GlobalMaxPooling1D()(x2)

    x3 = Conv1D(filters, 3, strides=1, padding='valid', input_shape=(timesteps, dim),name='conv3')(inputs)
    x3 = BatchNormalization(name='bn3')(x3)
    x3 = Activation('relu')(x3)
    x3 = GlobalMaxPooling1D()(x3)
    
    x4 = Conv1D(filters, 2, strides=1, padding='valid', input_shape=(timesteps, dim),name='conv4')(inputs)
    x4 = BatchNormalization(name='bn4')(x4)
    x4 = Activation('relu')(x4)
    x4 = GlobalMaxPooling1D()(x4)
    
    x = Concatenate()([x1, x2, x3, x4])
    x = Dense(256, activation='relu',name='dense1')(x)
    if mode=='feat':
        output=x
    else:
        x = Dropout(0.25)(x)
        if ac == 'tanh':
            output = Dense(1, activation='tanh',name='last_dense')(x)
        elif ac == 'tanh+sigmoid' or ac == 'sigmoid+tanh':
            x1 = Dense(1, activation='sigmoid',name='last_dense_1')(x)
            x2 = Dense(1, activation='tanh',name='last_dense_2')(x)
            output = [x1, x2]
        else:
            output = Dense(1, activation='sigmoid',name='last_dense')(x)

    return Model(inputs=inputs, outputs=output)