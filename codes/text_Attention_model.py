from multihead_attention import Attention
from position_embedding import Position_Embedding
import tensorflow as tf
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import regularizers


def model(timesteps=27, dim=300, embedding_matrix=None, head=8, head_size=64, output_type='pred'):
    inputs = Input((timesteps,))
    input_len = Input((None,))
    x = Embedding(embedding_matrix.shape[0],dim,weights=[embedding_matrix], input_length=timesteps,
                  mask_zero=False,trainable=False,name='embedding')(inputs)
    x = Position_Embedding()(x)
    x = Attention(head, head_size)([x, x, x, input_len, input_len])
    x = GlobalAveragePooling1D()(x)
    x = Dense(256, activation='relu',name='dense1')(x)
    if output_type=='feat':
        output = x
    else:
        x = Dropout(0.3)(x)
    #     x1 = Dense(1, activation='sigmoid', name='last_dense_1')(x)
        x2 = Dense(1, activation='tanh', name='last_dense_2')(x)
        output = x2

    return Model(inputs=[inputs, input_len], outputs=output)