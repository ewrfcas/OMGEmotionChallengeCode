# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 11:23:01 2018

@author: seria

                    _ooOoo_
                  o888888888o
                 o88`_ . _`88o
                 (|  0   0  |)
                 O \   。   / O
              _____/`-----‘\_____
            .’   \||  _ _  ||/   `.
            |  _ |||   |   ||| _  |
            |  |  \\       //  |  |
            |  |    \-----/    |  |
             \ .\ ___/- -\___ /. /
         ,--- /   ___\<|>/___   \ ---,
         | |:    \    \ /    /    :| |
         `\--\_    -. ___ .-    _/--/‘
   ===========  \__  NOBUG  __/  ===========

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange
import time
import os
import csv
import h5py
import pandas as pd
import random as rand
import numpy as np
import tensorflow as tf
import scipy.ndimage as ndimg
from skimage import transform
from scipy.stats import pearsonr
from model import select_model
from tensorflow.python.framework import graph_util as gu
from tensorflow.python.ops.losses.losses_impl import Reduction
import tensorflow.contrib.layers as layers
from tensorflow.python.client import device_lib
import sys
import re

class RNN_Seq2Sng(object):
    dev_modes = ('train', 'comtrain', 'test', 'finetune', 'freeze')
    param = {}

    '''-----------------------------------------------------
    Settings for developer and relevant directories
    -----------------------------------------------------'''
    # Model directory (where fnn pb file lives)
    param['fnn_path'] = '../model/seq-54605.pb'
    # How are we gonna make use of this model
    param['setting'] = {'devel_mode': 'finetune',
                        'fnn_ckpt': '../model/noa-54960',
                        'rnn_ckpt': '../data/seeta_omg/omg_seeta_tr/vf-test-ep20',
                        'rnn_step': 0,
                        'model_path': '../model/lstm-101.pb'}
    # Name of FNN output
    param['output_node'] = 'map/TensorArrayStack/TensorArrayGatherV3:0'
    # Where the training dataset lives
    param['data_dir'] = {'train': '../data/seeta_omg/omg_seeta_tr',
                         'val': '../data/seeta_omg/omg_seeta_tr',
                         'test': '../data/seeta_omg/omg_seeta_tr'}
    # Name of label file
    param['label_file'] = {'train': 'omg_train.csv', 'val': 'omg_val.csv', 'test': 'omg_test.csv'}
    # Where the embedding vectors live
    param['embd_path'] = '../data/seeta_omg'
    # Whether to output learned features
    param['feat_gate'] = True

    '''-----------------------------------------------------
    Network architecture
    -----------------------------------------------------'''
    # top-n accuracy
    param['rank'] = 3
    # Image size
    param['img_size'] = 227
    # Batch size
    param['batch_size'] = 16
    # The number of output nodes in FNN
    param['fnn_outnode'] = 512
    # Embedding feature dimesion (non-positive integer disables fc layer after embedding layer)
    param['embd_size'] = 0
    # The number of hidden nodes in RNN
    param['hidden_size'] = 256
    # The number of attetion nodes (non-positive integer disables attention layer)
    param['attention_size'] = 32
    # Maximum sequence length (non-positive integer means no limitation)
    param['max_frames'] = 64
    # The number of classes
    param['num_class'] = -7
    # The number of values
    param['num_value'] = 2
    # Acceptable neighbor area
    param['neighbor'] = 0.08
    # Whether use one-hot labels
    param['one_hot'] = False
    # Tyoe of fnn
    param['fnn_type'] = 'InceptionV3'
    # Type of rnn
    param['rnn_type'] = 'lstm'
    # Weight decay
    param['wd'] = 0.00005
    # Probability of dropping neurons
    param['pdrop'] = 0.
    # Type of loss function
    # mse = 0.5 * AVG[(y-y')^2]
    # contrastive = 0.5 * AVG[I*(y-y')^2 + (1-I)*MAX[0, thresh-(y-y')^2]]
    param['loss'] = 'mae'

    '''-----------------------------------------------------
    Data preparation
    -----------------------------------------------------'''
    # Image feature flow switch
    param['enable_img'] = True
    # Video feature flow switch
    param['enable_vid'] = False
    # Text feature flow switch
    param['enable_text'] = False
    # Audio feature flow switch
    param['enable_aud'] = False
    # Whether to augment data sequentially
    param['seq_aug'] = 'sample'
    # Frame sampling frequency
    param['sample_freq'] = 5
    # Whether to shuffle data
    param['if_shuffle'] = True

    '''-----------------------------------------------------
    Training settings
    -----------------------------------------------------'''
    # Maximum allowed step
    param['max_steps'] = 4000
    # Inspect progress every x steps
    param['inspect'] = 2
    # Save model every x steps
    param['interval'] = 50
    # Initial learning rate
    param['lr'] = 0.008
    # How often to decay learning rate
    param['lr_decay_step'] = 200
    # Learning rate decay coefficient: eta *= decay_rate
    param['decay_rate'] = 0.8
    # Type of optimizer
    param['optimizer'] = 'momentum'
    # Standard of performance measurement
    param['metric'] = 'dist'
    # Specify which gpu(s) to occupy
    param['available_gpus'] = '0'
    # Planned gpu memory to occupy
    param['gpu_mem_fraction'] = 0.9

    def __init__(self):
        assert self.param['setting']['devel_mode'] in self.dev_modes, \
            '%s is not an available developing mode.'%self.param['setting']['devel_mode']
        if self.param['setting']['devel_mode'] in ['train', 'comtrain']:
            self.curr_step = 0
        else: # finetune / test / freeze
            if self.param['setting']['rnn_step'] > 0:
                checkpoint = '%s/ckpt-%d'%(self.param['setting']['rnn_ckpt'], self.param['setting']['rnn_step'])
            else:
                checkpoint = tf.train.latest_checkpoint(self.param['setting']['rnn_ckpt'])
            self.curr_step = 0 #int(str(checkpoint).split('-')[-1])

        self.epoch = 0
        self.global_step = tf.Variable(self.curr_step, trainable=False)
        self.nseqs = {}
        self.order = {}
        self.nseqs['train'] = 0
        self.order['train'] = []
        self.text = tf.zeros([self.param['batch_size'],256], dtype=tf.float32)
        self.aud = tf.zeros([self.param['batch_size'], 256], dtype=tf.float32)
        self.nextBatch = self.readData('train')
        self.logits_valid = tf.zeros([self.param['batch_size'], self.param['max_frames']], dtype=tf.float32)
        # build up network
        self.input_img = tf.placeholder(tf.float32, (self.param['batch_size'], self.param['max_frames'],
                                                     self.param['img_size'], self.param['img_size'], 3))
        self.input_len = tf.placeholder(tf.int32, (self.param['batch_size'], None))
        self.input_num = tf.placeholder(tf.float32, (self.param['batch_size'], abs(self.param['num_value'])))
        if self.param['one_hot']:
            self.input_lab = tf.placeholder(tf.int32, (self.param['batch_size'], abs(self.param['num_class'])))
        else:
            self.input_lab = tf.placeholder(tf.int32, (self.param['batch_size']))

        if self.param['enable_img']:
            self.input_feat = tf.placeholder(tf.float32, (self.param['batch_size'],
                                                          self.param['max_frames'],
                                                          self.param['fnn_outnode']))
            self.output_rnn, self.alpha = self.buildRNN(self.input_img, self.input_len,
                                                        self.param['rnn_type'], self.input_feat)
        else:
            self.output_rnn, self.alpha = self.buildRNN(self.input_img, self.input_len, self.param['rnn_type'])
        # _, self.pred = tf.nn.top_k(self.output_rnn)
        self.cost_total = self.loss(self.output_rnn, self.input_lab, self.input_num)
        self.performance = self.measure(self.output_rnn, self.input_lab, self.input_num)
        self.train_op = self.optimizer()
        # validation
        if self.param['data_dir']['val']:
            self.nseqs['val'] = 0
            self.order['val'] = []
            self.iterAll = self.readData('val')
            self.val_img = tf.placeholder(tf.float32, (self.param['batch_size'], self.param['max_frames'],
                                                         self.param['img_size'], self.param['img_size'], 3))
            self.val_len = tf.placeholder(tf.int32, (self.param['batch_size'], None))
            self.val_num = tf.placeholder(tf.float32, (self.param['batch_size'], abs(self.param['num_value'])))
            if self.param['one_hot']:
                self.val_lab = tf.placeholder(tf.int32, (self.param['batch_size'], abs(self.param['num_class'])))
            else:
                self.val_lab = tf.placeholder(tf.int32, (self.param['batch_size']))

            if self.param['enable_img']:
                self.val_feat = tf.placeholder(tf.float32, (self.param['batch_size'],
                                                            self.param['max_frames'],
                                                            self.param['fnn_outnode']))
                self.val_output, self.val_alpha = self.buildRNN(self.val_img, self.val_len,
                                                                self.param['rnn_type'], self.val_feat)
            else:
                self.val_output, self.val_alpha = self.buildRNN(self.val_img, self.val_len, self.param['rnn_type'])
            self.val_cost = self.loss(self.val_output, self.val_lab, self.val_num)
            self.val_performance = self.measure(self.val_output, self.val_lab, self.val_num)
        # predict
        if self.param['data_dir']['test']:
            self.nseqs['test'] = 0
            self.order['test'] = []
            self.fetchEg = self.readData('test')
            self.test_img = tf.placeholder(tf.float32, (self.param['batch_size'], self.param['max_frames'],
                                                       self.param['img_size'], self.param['img_size'], 3))
            self.test_len = tf.placeholder(tf.int32, (self.param['batch_size'], None))
            self.test_num = tf.placeholder(tf.float32, (self.param['batch_size'], abs(self.param['num_value'])))
            if self.param['one_hot']:
                self.test_lab = tf.placeholder(tf.int32, (self.param['batch_size'], abs(self.param['num_class'])))
            else:
                self.test_lab = tf.placeholder(tf.int32, (self.param['batch_size']))

            if self.param['enable_img']:
                self.test_feat = tf.placeholder(tf.float32, (self.param['batch_size'],
                                                            self.param['max_frames'],
                                                            self.param['fnn_outnode']))
                self.test_output, self.test_alpha = self.buildRNN(self.test_img, self.test_len,
                                                                self.param['rnn_type'], self.test_feat)
            else:
                self.test_output, self.test_alpha = self.buildRNN(self.test_img, self.test_len,
                                                                self.param['rnn_type'])
        # start a session
        config = tf.ConfigProto(log_device_placement=False)
        config.gpu_options.per_process_gpu_memory_fraction = self.param['gpu_mem_fraction']

        # aval_gpu = self.getAvailableGpus()
        # if aval_gpu:
        #     print('+' + (19 * '-') + '+')
        #     print('|Executing on /gpu:%s|' % aval_gpu)
        #     print('+' + (19 * '-') + '+')
        #     config.gpu_options.visible_device_list = aval_gpu
        config.gpu_options.visible_device_list = self.param['available_gpus']
        self.sess = tf.Session(config=config)
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=3)
        self.sess.run(tf.global_variables_initializer())

        if self.param['setting']['devel_mode'] == 'finetune':
            print('+' + (60 * '-') + '+')
            print('|Restore from %47s|'%str(checkpoint))
            print('+' + (60 * '-') + '+')
            to_be_restored = tf.global_variables(scope=self.param['rnn_type'])
            # to_be_restored = [var for var in to_be_restored if self.param['rnn_type']+'/cls' not in var.op.name]
            restorer = tf.train.Saver(to_be_restored)
            restorer.restore(self.sess, checkpoint)

        if self.param['setting']['devel_mode'] == 'comtrain':
            ckpt = tf.train.latest_checkpoint(self.param['setting']['fnn_ckpt'])
            variables = tf.get_collection(tf.GraphKeys.VARIABLES, scope=self.param['rnn_type']+'/'+self.param['fnn_type'])
            to_be_restored = {}
            for var in variables:
                var_name = re.sub('%s/%s'%(self.param['rnn_type'],self.param['fnn_type']),
                                  self.param['fnn_type'], var.op.name)
                to_be_restored[var_name] = var
            # to_be_restored = [var for var in variables if self.param['fnn_type'] in var.op.name]
            restorer = tf.train.Saver(to_be_restored)
            restorer.restore(self.sess, ckpt)

    def buildRNN(self, input_img, length, RNN, feature=None):
        '''

        Return:
        '''
        network_err = Exception('It is not in support of %s neural nets.'%RNN)
        if RNN == 'lstm':
            logits_rnn = self.lstm(input_img, length, feature)
        elif RNN == 'c3d':
            logits_rnn = self.c3d(input_img)
        elif RNN == 'mulstm':
            values_rnn = []
            for i in range(self.param['num_value']):
                values_temp, alpha = self.lstm(input_img, length, feature, 'lstm_'+str(i))
                values_rnn += [values_temp[1]]
            logits_rnn = tf.concat(values_rnn, axis=1)
            return [tf.constant(0, dtype=tf.float32), logits_rnn], alpha
        else:
            raise network_err

        return logits_rnn

    def lstm(self, inputs, length, logits_fnn, var_scope='lstm'):
        def _getVarWithDecay(name, shape, initializer='xavier'):
            init_err = Exception('%s initializer is not defined or supported.' % initializer)
            # with tf.device('/cpu:0'):
            if initializer == 'xavier':
                var = tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer())
            elif initializer == 'trunc_norm':
                var = tf.get_variable(name, shape, initializer=tf.truncated_normal_initializer())
            elif initializer == 'rand_norm':
                var = tf.get_variable(name, initializer=tf.random_normal(shape, stddev=0.1))
            elif initializer == 'zero':
                var = tf.get_variable(name, shape, initializer=tf.zeros_initializer())
            elif initializer == 'one':
                var = tf.get_variable(name, shape, initializer=tf.ones_initializer())
            else:
                raise init_err
            weight_decay = self.param['wd'] * tf.nn.l2_loss(var)
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, weight_decay)

            return var

        # a batch_size long vector which indicates frames of each sample
        length = tf.reshape(length,[self.param['batch_size'], self.param['max_frames']])
        seq_len = tf.reduce_sum(length, 1)
        last_frames = tf.range(0, self.param['batch_size']) * self.param['max_frames'] + seq_len - 1
        if var_scope != 'lstm':
            var_scope = self.param['rnn_type']+'/'+var_scope
        with tf.variable_scope(var_scope, reuse=tf.AUTO_REUSE) as scope:
            if self.param['enable_img']:
                pass
            else:
                if self.param['setting']['fnn_ckpt'] and self.param['fnn_type']:
                    model_fn = select_model(self.param['fnn_type'])
                    logits_unroll = model_fn(self.param['num_class'],
                                          tf.reshape(inputs, [-1,self.param['img_size'],self.param['img_size'],3]),
                                          1-self.param['pdrop'], False)
                    logits_fnn = tf.reshape(logits_unroll, [self.param['batch_size'], self.param['max_frames'],
                                                            self.param['fnn_outnode']], name='fnn')
                else:
                    ouput_graph_def = tf.GraphDef()
                    with open(self.param['fnn_path'], 'rb') as model:
                        ouput_graph_def.ParseFromString(model.read())
                        logits_fnn = tf.import_graph_def(ouput_graph_def,
                                                         {'input:0': inputs},
                                                         return_elements=[self.param['output_node']])
                        logits_fnn = tf.reshape(tf.convert_to_tensor(logits_fnn[0]),
                                                [self.param['batch_size'], self.param['max_frames'],
                                                 self.param['fnn_outnode']], name='fnn')
            if self.param['embd_size'] > 0:
                logits_embd = layers.fully_connected(logits_fnn, scope='embedding',
                                                     num_outputs=self.param['embd_size'],
                                                     activation_fn=tf.nn.relu,
                                                     weights_regularizer=tf.contrib.layers.l2_regularizer(
                                                         self.param['wd']))
            else:
                logits_embd = tf.identity(logits_fnn, name='embedding/Relu')

            cell = tf.nn.rnn_cell.BasicLSTMCell(self.param['hidden_size'], state_is_tuple=True)
            #rnn_layers = [tf.nn.rnn_cell.BasicLSTMCell(hidden_size, state_is_tuple=True)
            #        for hidden_size in [self.param['hidden_size'], self.param['hidden_size']]]
            #cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)
            init_state = cell.zero_state(self.param['batch_size'], tf.float32)
            output_rnn, output_state = tf.nn.dynamic_rnn(cell, logits_embd, initial_state=init_state,
                                                         sequence_length=seq_len)

            cell_state_size = int(output_rnn.get_shape()[2])
            if self.param['attention_size'] > 0:
                w = _getVarWithDecay('w_omega', [self.param['hidden_size'], self.param['attention_size']], 'rand_norm')
                b = _getVarWithDecay('b_omega', [self.param['attention_size']], 'rand_norm')
                u = _getVarWithDecay('u_omega', [self.param['attention_size']], 'rand_norm')
                logits_att = tf.tanh(tf.tensordot(output_rnn, w, axes=1) + b)
                # logits_valid = 1e2*tf.cast(length-1, tf.float32) + tf.tensordot(logits_att, u, axes=1)
                logits_valid = tf.tensordot(logits_att, u, axes=1)
                alpha = tf.nn.softmax(logits_valid, name='alpha_weights')
                output_weighted = tf.multiply(output_rnn, tf.expand_dims(alpha, -1))
                output_timestep = tf.reshape(tf.reduce_sum(output_weighted, 1), [-1, cell_state_size])
            else:
                alpha = tf.constant(0, dtype=tf.float32)
                output_timestep = tf.gather(tf.reshape(output_rnn, [-1, cell_state_size]), last_frames)
                # cls_projection = lambda x: layers.fully_connected(x, scope='cls',
                #                                                num_outputs=self.param['num_class'],
                #                                                activation_fn=tf.nn.softmax)
                # logits_rnn = tf.map_fn(cls_projection, output_rnn, dtype=tf.float32)

            # combine extra feature
            if self.param['enable_text']:
                output_timestep = tf.concat([output_timestep, self.text], axis=1)
            if self.param['enable_aud']:
                output_timestep = tf.concat([output_timestep, self.aud], axis=1)

            self.feat = tf.identity(output_timestep, name='feat')

            if var_scope == 'lstm':
                if self.param['num_class'] > 0:
                    logits_rnn = layers.fully_connected(output_timestep, scope='cls',
                                                    num_outputs=self.param['num_class'],
                                                    activation_fn=tf.nn.softmax,
                                                    weights_regularizer=tf.contrib.layers.l2_regularizer(self.param['wd']))
                else:
                    logits_rnn = tf.constant(0, dtype=tf.float32)
                if self.param['num_value'] > 0:
                    values_rnn = layers.fully_connected(output_timestep, scope='reg',
                                                    num_outputs=self.param['num_value'],
                                                    activation_fn=tf.tanh,
                                                    weights_regularizer=tf.contrib.layers.l2_regularizer(self.param['wd']))
                else:
                    values_rnn = tf.constant(0, dtype=tf.float32)
            else:
                values_rnn = layers.fully_connected(output_timestep, scope='reg' + var_scope[-1],
                                                    num_outputs=1,
                                                    activation_fn=tf.tanh,
                                                    weights_regularizer=tf.contrib.layers.l2_regularizer(
                                                        self.param['wd']))
                return [tf.constant(0, dtype=tf.float32), values_rnn], alpha

        return [logits_rnn, values_rnn], alpha

    def c3d(self, inputs, var_scope='c3d'):
        '''
        -----------------------------------------
        ____layer___________|____shape___________
            input           |    96x112x112x3
            conv1           |    96x56 x56 x64
            conv2           |    48x28 x28 x128
            conv3i          |    24x14 x14 x256
            conv3ii         |    24x14 x14 x256
            conv4i          |    12x7  x7  x512
            conv4ii         |    12x7  x7  x512
            conv5i          |    6 x4  x4  x512
            conv5ii         |    6 x4  x4  x512
            fc1             |    1 x4096
            fc2             |    1 x4096
            output          |    1 x nclass
        -----------------------------------------
        '''

        def _getVarWithDecay(name, shape, initializer='xavier'):
            init_err = Exception('%s initializer is not defined or supported.' % initializer)
            with tf.device('/cpu:0'):
                if initializer == 'xavier':
                    var = tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer())
                elif initializer == 'trunc_norm':
                    var = tf.get_variable(name, shape, initializer=tf.truncated_normal_initializer())
                elif initializer == 'rand_norm':
                    var = tf.get_variable(name, initializer=tf.random_normal(shape, stddev=0.1))
                elif initializer == 'zero':
                    var = tf.get_variable(name, shape, initializer=tf.zeros_initializer())
                elif initializer == 'one':
                    var = tf.get_variable(name, shape, initializer=tf.ones_initializer())
                else:
                    raise init_err
            weight_decay = self.param['wd'] * tf.nn.l2_loss(var)
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, weight_decay)

            return var

        def _conv3d(layer, input, w, b, stride=(1, 1), mode='relu'):
            activation_err = Exception('%s function is not defined or supported.' % mode)
            conv = tf.nn.bias_add(tf.nn.conv3d(input, w, [1, stride[0], stride[1], stride[1], 1], 'SAME'),
                                  b, name='conv_' + layer)
            if mode == 'relu':
                return tf.nn.relu(conv, name=mode + '_' + layer)
            else:
                raise activation_err

        def _pooling3d(layer, input, kernel=(2, 2), stride=(2, 2), mode='max'):
            pooling_err = Exception('%s pooling is not defined or supported.' % mode)
            if mode == 'max':
                return tf.nn.max_pool3d(input,
                                        [1, kernel[0], kernel[1], kernel[1], 1],
                                        [1, stride[0], stride[1], stride[1], 1],
                                        'SAME', name='pool_' + layer)
            else:
                raise pooling_err

        def _dense3d(layer, input, batch_size, out_nodes, pdrop=0., initializer='xavier', actv_fn='relu'):
            init_err = Exception('%s initializer is not defined or supported.' % initializer)
            if initializer == 'xavier':
                return tf.nn.dropout(
                            tf.layers.dense(
                                    tf.reshape(input, [batch_size, -1]),
                                                units=out_nodes,
                                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                kernel_regularizer=tf.contrib.layers.l2_regularizer(self.param['wd']),
                                                activation=tf.nn.relu,
                                                name='fc_' + layer),
                            1 - pdrop)
            elif initializer == 'trunc_norm':
                return tf.nn.dropout(
                            tf.layers.dense(
                                    tf.reshape(input, [batch_size, -1]),
                                                units=out_nodes,
                                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                                kernel_regularizer=tf.contrib.layers.l2_regularizer(self.param['wd']),
                                                name='fc_' + layer),
                            1 - pdrop)
            else:
                raise init_err

        with tf.variable_scope(var_scope, reuse=tf.AUTO_REUSE) as scope:
            weights = {'conv1': _getVarWithDecay('w_conv_1', [3, 3, 3, 3, 64]),
                       'conv2': _getVarWithDecay('w_conv_2', [3, 3, 3, 64, 128]),
                       'conv3i': _getVarWithDecay('w_conv_3i', [3, 3, 3, 128, 256]),
                       'conv3ii': _getVarWithDecay('w_conv_3ii', [3, 3, 3, 256, 256]),
                       'conv4i': _getVarWithDecay('w_conv_4i', [3, 3, 3, 256, 512]),
                       'conv4ii': _getVarWithDecay('w_conv_4ii', [3, 3, 3, 512, 512]),
                       'conv5i': _getVarWithDecay('w_conv_5i', [3, 3, 3, 512, 512]),
                       'conv5ii': _getVarWithDecay('w_conv_5ii', [3, 3, 3, 512, 512]),
                       'output': _getVarWithDecay('w_output_8', [4096, self.param['num_class']])}
            # biases are vectors cauz a broadcastable function will be applied
            biases = {'conv1': _getVarWithDecay('b_conv_1', [64]),
                      'conv2': _getVarWithDecay('b_conv_2', [128]),
                      'conv3i': _getVarWithDecay('b_conv_3i', [256]),
                      'conv3ii': _getVarWithDecay('b_conv_3ii', [256]),
                      'conv4i': _getVarWithDecay('b_conv_4i', [512]),
                      'conv4ii': _getVarWithDecay('b_conv_4ii', [512]),
                      'conv5i': _getVarWithDecay('b_conv_5i', [512]),
                      'conv5ii': _getVarWithDecay('b_conv_5ii', [512]),
                      'output': _getVarWithDecay('b_output_8', [self.param['num_class']], 'zero')}

            conv1 = _conv3d('1', inputs, weights['conv1'], biases['conv1'])
            pool1 = _pooling3d('1', conv1, kernel=(1, 2), stride=(1, 2))

            conv2 = _conv3d('2', pool1, weights['conv2'], biases['conv2'])
            pool2 = _pooling3d('2', conv2)

            conv3i = _conv3d('3i', pool2, weights['conv3i'], biases['conv3i'])
            conv3ii = _conv3d('3ii', conv3i, weights['conv3ii'], biases['conv3ii'])
            pool3 = _pooling3d('3', conv3ii)

            conv4i = _conv3d('4i', pool3, weights['conv4i'], biases['conv4i'])
            conv4ii = _conv3d('4ii', conv4i, weights['conv4ii'], biases['conv4ii'])
            pool4 = _pooling3d('4', conv4ii)

            conv5i = _conv3d('5i', pool4, weights['conv5i'], biases['conv5i'])
            conv5ii = _conv3d('5ii', conv5i, weights['conv5ii'], biases['conv5ii'])
            pool5 = _pooling3d('5', conv5ii)

            fc1 = _dense3d('6', pool5, self.param['batch_size'], 4096, pdrop=self.param['pdrop'])
            fc2 = _dense3d('7', fc1, self.param['batch_size'], 4096, pdrop=self.param['pdrop'])

            output = tf.nn.bias_add(tf.matmul(fc2, weights['output']),
                                    biases['output'], name='output_8')
            logits = tf.nn.softmax(output, name='sftm')

        return logits

    def loss(self, outputs, labels, numbers):

        def _cost(logits, values, labels, numbers, type):
            loss_type_err = Exception('%s loss is unsupported.' % type)
            if type == 'xentropy':
                assert self.param['num_class']>0
                cost_per = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=logits,
                    labels=labels,
                    name='xentropy_per_example')
                cost_mean = tf.reduce_mean(cost_per, name='xentropy')
            elif type == 'mse':
                assert self.param['num_value']>0
                cost_per_value = tf.squared_difference(values, numbers)
                weight = tf.constant([[0.5], [0.5]], dtype=tf.float32)
                cost_per = tf.matmul(cost_per_value, weight, name='mse_per_example')
                cost_mean = tf.reduce_mean(cost_per, name='mse')
            elif type == 'mae':
                assert self.param['num_value']>0
                cost_per_value = tf.abs(values - numbers)
                weight = tf.constant([[0.45], [0.55]], dtype=tf.float32)
                cost_per = tf.matmul(cost_per_value, weight, name='mae_per_example')
                cost_mean = tf.reduce_mean(cost_per, name='mae')
            elif type == 'contrastive':
                assert  self.param['num_class']>0 and self.param['num_value']>0
                distance = tf.reduce_sum(tf.squared_difference(values, numbers), axis=1)
                cluster = tf.cast(tf.argmax(logits, tf.rank(logits) - 1), tf.int32)
                indicator = tf.cast(tf.equal(cluster, labels), tf.float32)
                cost_per = indicator * distance + \
                           (1 - indicator) * \
                           tf.maximum(self.param['neighbor'] * self.param['neighbor'] - distance, 0)
                cost_mean = tf.reduce_mean(cost_per, name='contrastive')
            elif type == 'rank':
                assert self.param['num_value'] > 0
                def _sort(batch, epsilon = 1e-6):
                    diff = []
                    for b in range(self.param['batch_size']):
                        expanded_batch = tf.stack(
                            self.param['batch_size'] * [batch[b]], axis=0) \
                                         + epsilon - batch
                        abs_expb = tf.abs(expanded_batch)
                        diff += [tf.div(expanded_batch, abs_expb)]
                    return tf.reduce_mean(tf.stack(diff, axis=0), axis=0)

                def _pearson(p, q):
                    covariance = tf.reduce_mean(p * q) - tf.reduce_mean(p) * tf.reduce_mean(q)

                    std_p = tf.sqrt(tf.reduce_mean(tf.square(p)) - tf.square(tf.reduce_mean(p)))
                    std_q = tf.sqrt(tf.reduce_mean(tf.square(q)) - tf.square(tf.reduce_mean(q)))

                    return tf.div(0.1*covariance, (std_p * std_q), name='rank')

                rank_true = _sort(numbers)
                rank_pred = _sort(values)
                cost_mean = _pearson(rank_true, rank_pred)
            else:
                raise loss_type_err
            return cost_mean

        logits, values = outputs
        cost_total = tf.constant(0, dtype=tf.float32)
        for loss in self.param['loss'].split(','):
            cost_total += _cost(logits, values, labels, numbers, loss)
        cost_total += self.param['wd'] * sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        #print(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        return cost_total

    def optimizer(self):
        optimizer_err = Exception('%s optimizer is unsupported.' % self.param['optimizer'])

        if self.param['optimizer'] == 'adam':
            optz = lambda lr: tf.train.AdamOptimizer(lr, 0.5, 0.95)
        elif self.param['optimizer'] == 'adadelta':
            optz = lambda lr: tf.train.AdadeltaOptimizer(lr)
        elif self.param['optimizer'] == 'momentum':
            optz = lambda lr: tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)
        else:
            raise optimizer_err

        return tf.contrib.layers.optimize_loss(self.cost_total, self.global_step, self.param['lr'], optz,
                                               clip_gradients=4., learning_rate_decay_fn=self.lrDecay)

    def lrDecay(self, lr, global_step):
        print('+' + (33 * '-') + '+')
        print('|Decay [%.3f] every [%5d] steps|' % (self.param['decay_rate'], self.param['lr_decay_step']))
        print('+' + (33 * '-') + '+')
        return tf.train.exponential_decay(lr, global_step, self.param['lr_decay_step'],
                                          self.param['decay_rate'], staircase=True)

    def measure(self, outputs, labels, numbers):
        def _mode(matrix, num):
            nrows = matrix.shape[0]
            ncols = matrix.shape[1]
            cnt = np.zeros((nrows, num), np.int32)
            for i in range(nrows):
                for j in range(ncols):
                    cnt[i, matrix[i, j]] += 1

            return np.argmax(cnt, 1).astype(np.int32)

        logits, values = outputs
        measure_err = Exception('%s as measure is unsupported.' % self.param['metric'])
        if self.param['metric'] == 'accuracy':
            assert self.param['num_class'] > 0
            labels_frame = tf.cast(tf.argmax(logits, tf.rank(logits)-1), tf.int32)
            # labels_frame = tf.py_func(_mode, [labels_frame, self.param['num_class']], tf.int32)
            matching_pred = tf.cast(tf.equal(labels, labels_frame), tf.float32)
            performance = tf.reduce_mean(matching_pred)
        elif self.param['metric'] == 'dist':
            assert self.param['num_value'] > 0
            distance = tf.sqrt(tf.reduce_sum(tf.squared_difference(values, numbers), axis=1))
            performance = tf.shape(tf.where(distance<self.param['neighbor']))[0]/self.param['batch_size']
        else:
            raise measure_err

        return performance

    def ccc(self, y_true, y_pred):

        def _ccc(y_true, y_pred):
            true_mean = np.mean(y_true)
            pred_mean = np.mean(y_pred)

            rho, _ = pearsonr(y_pred, y_true)

            std_predictions = np.std(y_pred)

            std_gt = np.std(y_true)

            ccc = 2 * rho * std_gt * std_predictions / (
                std_predictions ** 2 + std_gt ** 2 +
                (pred_mean - true_mean) ** 2)

            return ccc, rho

        arousal_true = y_true[:,0]
        valence_true = y_true[:,1]
        arousal_pred = y_pred[:, 0]
        valence_pred = y_pred[:, 1]

        arousal_ccc, acor = _ccc(arousal_true, arousal_pred)
        valence_ccc, vcor = _ccc(valence_true, valence_pred)

        return arousal_ccc, valence_ccc

    def readData(self, usage):
        assert not (self.param['enable_img'] and self.param['enable_vid'])
        feat_dir = 'test' if usage=='test' else usage
        if self.param['enable_img']:
            feat_img = h5py.File(os.path.join(self.param['embd_path'],'vgg_feature_%s.hdf5'%feat_dir))
        if self.param['enable_vid']:
            feat_img = h5py.File(os.path.join(self.param['embd_path'],'vid_feature_%s.hdf5'%feat_dir))
        # (temporary code) read text features
        if self.param['enable_text']:
            csv_text = pd.read_csv(
                '/users/seria/caochenjie/OMG_text_aud/text_feature/TextAttention_context0_%s_feat256.csv' % usage,
                header=0, sep=',')
            feat_text = {}
            for i, row in csv_text.iterrows():
                feat_text[row[0] + '_' + row[1].split('.')[0].split('_')[-1]] = row[4:].values
            # feat_text = {}
            # with open(os.path.join(self.param['embd_path'],'txt_feature_%s.txt'%feat_dir), 'r') as text:
            #     textline = text.readline()
            #     while textline:
            #         temp = textline.split('\t')
            #         utt = temp[0]
            #         featstr = temp[-1]
            #         featstr = featstr.split(' ')[0:256]
            #         feat = []
            #         for fs in range(len(featstr)):
            #             feat += [featstr[fs]]
            #         feat_text[utt] = np.array(feat).astype(np.float32)
            #         textline = text.readline()
        if self.param['enable_aud']:
            csv_aud = pd.read_csv(
                '/users/seria/caochenjie/OMG_text_aud/aud_feature/AudCNN_context0_%s_feat.csv' % usage,
                header=0, sep=',')
            feat_aud = {}
            for i, row in csv_aud.iterrows():
                feat_aud[row[0] + '_' + row[1].split('.')[0].split('_')[-1]] = row[4:].values
            # feat_aud = {}
            # with open(os.path.join(self.param['embd_path'],'aud_feature_%s.csv'%feat_dir), 'r') as aud:
            #     audreader = csv.reader(aud)
            #     for a, audline in enumerate(audreader):
            #         if a > 0:
            #             utt = audline[0]
            #             feat = audline[1:]
            #             feat_aud[utt] = np.array(feat).astype(np.float32)


        annotation = []
        max_frames = 0
        with open(os.path.join(self.param['data_dir'][feat_dir], self.param['label_file'][feat_dir]), 'r') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            prev = ''
            seq = {'filelist':[], 'frames':0}
            for l,line in enumerate(csvreader):
                filename = line[0]
                name_parser = filename.split('_')
                curr = name_parser[0]+name_parser[-2]
                # new sequence starts with lower number in filename
                if curr != prev and prev:
                    # if l>5000:
                    #    break
                    if seq['frames'] > max_frames:
                        max_frames = seq['frames']
                    annotation += [seq]
                    seq = {'filelist': [], 'frames': 0}

                if self.param['max_frames'] > 0 and \
                                seq['frames'] > self.param['max_frames']*self.param['sample_freq']-1:
                    continue
                else:
                    seq['filelist'] += [filename]
                    seq['label'] = line[1:]
                    seq['frames'] += 1
                    prev = curr
            annotation += [seq]
        if self.param['max_frames'] < 1:
            self.param['max_frames'] = max_frames
        nseqs = len(annotation)
        order = [i for i in range(nseqs)]
        self.curr_idx = 0
        self.nseqs[usage] = nseqs
        self.order[usage] = order
        # write down validation order
        # if usage == 'train':
        #     ground_truth = []
        #     for annt in annotation:
        #         ground_truth += [annt['label'][0:abs(self.param['num_value'])]]
        #     ground_truth = np.array(ground_truth).astype(np.float32)
        #     print(ground_truth[:,0].std(), ground_truth[:,1].std())
        if usage == 'val':
            ground_truth = []
            for annt in annotation:
                ground_truth += [annt['label'][0:abs(self.param['num_value'])]]
            self.ground_truth = np.array(ground_truth).astype(np.float32)
        if usage == 'test':
            self.eg_order = []
            with open('%s/output_order.csv' % self.param['data_dir'][usage], 'w') as scribe:
                writer = csv.writer(scribe, delimiter=',')
                writer.writerow(['video', 'utterance'])
                for annt in annotation:
                    file_info = annt['filelist'][0].split('_')
                    writer.writerow([file_info[0], 'utterance_%s.mp4' %
                                     (file_info[3] if len(file_info)>4 else file_info[2])])

        def _shuffleData(order):
            '''
            Args:
            num: batch size
            leftover: the rest samples of last epoch
            '''
            for j in range(nseqs):
                idx = int(rand.random() * (nseqs - j)) + j
                temp = order[j]
                order[j] = order[idx]
                order[idx] = temp

        if self.param['if_shuffle'] and usage == 'train':
            _shuffleData(self.order[usage])

        def _preProcess(img_path):
            src_img = ndimg.imread(img_path)
            np_img = transform.resize(src_img, (self.param['img_size'], self.param['img_size'], 3))
            _mean = np.mean(np_img)
            _stddev = np.std(np_img)
            _adj = 1 / np.sqrt(np_img.size)
            _dev = max(_stddev, _adj)
            std_img = (np_img - _mean) / _dev
            magn = np.max(np.abs(std_img))
            img = [std_img / magn]

            return img

        def _dataAugmentation(seq_len, usage=usage):
            if self.param['seq_aug'] == 'sample':
                seq_len = seq_len // self.param['sample_freq']
                indices = []
                for sample in range(seq_len):
                    if usage == 'train':
                        indices.append(int((rand.random() + sample) * self.param['sample_freq']))
                    else:
                        indices.append(sample * self.param['sample_freq'])
                if seq_len==0:
                    indices = [0]
                    seq_len = 1
            elif self.param['seq_aug'] == 'chunk':
                length = min(self.param['max_frames'], seq_len)
                if usage == 'train':
                    onset = int(rand.random() * max((seq_len - self.param['max_frames']), 0))
                    indices = list(range(onset, onset + length))
                else:
                    indices = [idx for idx in range(length)]
            else:
                indices = list(range(seq_len))

            return indices, seq_len

        def _nextBatch(one_hot=False, usage=usage):
            '''
            Args:
            one_hot: whether to use dense label
            from_raw_data: True  -> read from raw images for embedding and subsequent procedure
                           False -> read from saved embedding vectors
            Returns:

            '''
            if usage == 'train':
                self.epoch = -abs(self.epoch)
            curr_batch = []
            self.curr_idx += self.param['batch_size']
            # start a new epoch
            if self.curr_idx > nseqs:
                for idx in self.order[usage][self.curr_idx - self.param['batch_size']:nseqs]:
                    curr_batch += [annotation[idx]]
                for idx in self.order[usage][0:self.curr_idx-nseqs]:
                    curr_batch += [annotation[idx]]

                if usage == 'train':
                    self.epoch = abs(self.epoch)+1
                if self.param['if_shuffle'] and usage == 'train':
                    # reshuffle data to read
                    _shuffleData(self.order[usage])
                else:
                    self.order[usage] = [i for i in range(nseqs)]
                self.curr_idx = 0
            else:
                for idx in self.order[usage][self.curr_idx - self.param['batch_size']:self.curr_idx]:
                    curr_batch += [annotation[idx]]

            images = []
            labels = []
            frames = []
            numbers = []
            npimg = []
            nptext = []
            npaud = []
            featdir = 'test' if usage == 'test' else usage
            if usage == 'test':
                self.eg_order = []
            for b in range(self.param['batch_size']):
                file_list = curr_batch[b]['filelist']
                info = file_list[0].split('_')
                if len(info) > 4:
                    utterance = info[0] + '_' + info[1] + '_' + info[3]
                    utt_temp = info[0] + '_' + info[2] + '_' + info[3]
                else:
                    utterance = info[0] + '_' + info[2]
                    utt_temp = info[0] + '_' + info[1] + '_' + info[2]
                self.eg_order.append(utterance)
                if self.param['enable_img']:
                    if self.param['fnn_outnode']!=512:
                        feature = []
                        indices, _ = _dataAugmentation(len(file_list))
                        for idx in indices:
                            file_path = file_list[idx]
                            feature += [feat_img[os.path.basename(file_path)].value.flatten()]
                        npimg += [np.vstack((np.array(feature),
                                              np.zeros((self.param['max_frames'] - len(indices),
                                                        self.param['fnn_outnode']))))]

                    # ----------------------------------------------------------------
                    else:
                        x_temp = pd.read_csv(
                            '/users/seria/caochenjie/OMG/dataset/new_%sset_vggface/%s.csv'
                            % ('full', utterance)).values
                        indices, ori_length = _dataAugmentation(x_temp.shape[0])
                        if ori_length < self.param['max_frames']:
                            x_temp = np.concatenate((x_temp[indices],
                                                     np.zeros((self.param['max_frames'] - ori_length,
                                                               x_temp.shape[1]))), axis=0)
                        else:
                            x_temp = x_temp[indices[0:self.param['max_frames']], :]
                        npimg.append(x_temp)
                    # ----------------------------------------------------------------

                    # empty images
                    images += [np.zeros((self.param['max_frames'],
                                         self.param['img_size'], self.param['img_size'], 3))]
                else:
                    imgs = []
                    indices, _ = _dataAugmentation(len(file_list))
                    for idx in indices:
                        file_path = file_list[idx]
                        imgs += _preProcess(os.path.join(self.param['data_dir'][featdir], file_path))
                    images += [np.vstack((np.array(imgs),
                                          np.zeros((self.param['max_frames'] - len(indices),
                                                    self.param['img_size'], self.param['img_size'], 3))))]
                if one_hot:
                    curr_labels = np.zeros((self.param['max_frames'], self.param['num_class']))
                    bgm = np.arange(self.param['max_frames']) * self.param['num_class']
                    curr_labels.flat[bgm + int(curr_batch[b]['label'][-1])] = 1
                else:
                    curr_labels = int(curr_batch[b]['label'][-1])
                labels += [curr_labels]
                curr_frames = np.zeros(self.param['max_frames'])
                curr_frames[0:curr_batch[b]['frames']] += 1
                frames += [curr_frames]
                curr_numbers = [float(num) for num in curr_batch[b]['label'][0:abs(self.param['num_value'])]]
                numbers += [curr_numbers]

                if self.param['enable_text']:
                    nptext += [feat_text[utterance]]
                if self.param['enable_aud']:
                    npaud += [feat_aud[utterance]]
            self.text = tf.convert_to_tensor(np.array(nptext).astype(np.float32))
            self.aud = tf.convert_to_tensor(np.array(npaud).astype(np.float32))
            #############
            ###delete###
            ###########
            if self.param['fnn_outnode'] < 600:
                npimg = np.array(npimg)
                npimg = npimg.reshape(self.param['batch_size'], self.param['max_frames'], -1)
            # labels: class->int32 , value->float32
            return np.array(npimg).astype(np.float32), \
                   np.array(images).astype(np.float32), \
                   np.array(labels).astype(np.int32), \
                   np.array(numbers).astype(np.float32), \
                   np.array(frames).astype(np.int32)

        return _nextBatch

    def train(self):
        if self.param['setting']['devel_mode'] == 'freeze':
            pass
        else:
            ckpt_dir = '%s/ckpt-%s-%s' % (self.param['data_dir']['train'],
                                          self.param['rnn_type'],
                                          time.strftime('%m%d%M%S', time.localtime()))
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)
            self.scribe = open('%s/scribe.csv'%ckpt_dir, 'w')
            self.writer = csv.writer(self.scribe, delimiter=',')
            print('Models are saved in %s'%ckpt_dir)
            ckpt_file = ckpt_dir+'/ckpt'
            summ_acc = []
            summ_loss = []
            prev_end = self.curr_step
            max_a = -1
            max_v = -1
            max_c = -1
            for step in range(self.curr_step, self.param['max_steps']):
                if step == self.curr_step:
                    print('+'+78*'-'+'+')

                if self.epoch > 0 or step == self.curr_step:
                    prev_end = step
                    flag_update = False
                    if self.epoch > 0:
                        self.writer.writerow(summ_acc)
                        self.writer.writerow(summ_loss)
                        acc_epoch = np.mean(np.array(summ_acc))
                        loss_epoch = np.mean(np.array(summ_loss))
                        self.writer.writerow([step, acc_epoch])
                    else:
                        acc_epoch = 0
                        loss_epoch = 0
                    summ_acc = []
                    summ_loss = []
                    start_val = time.time()

                    iteration_val = self.nseqs['val']//self.param['batch_size'] + 1
                    perform_avg = 0
                    loss_avg = 0
                    regression = []
                    for it in range(iteration_val):
                        feats, images, labels, numbers, frames = self.iterAll(one_hot=self.param['one_hot'])
                        if self.param['enable_img']:
                            perform_curr, loss_curr, out = self.sess.run(
                                [self.val_performance, self.val_cost, self.val_output],
                                feed_dict={self.val_feat: feats, self.val_lab: labels,
                                           self.val_num: numbers, self.val_len: frames})
                        else:
                            perform_curr, loss_curr, out = self.sess.run(
                                [self.val_performance, self.val_cost, self.val_output],
                                feed_dict={self.val_img: images, self.val_lab: labels,
                                           self.val_num: numbers, self.val_len: frames})
                        if self.param['num_value'] > 0:
                            for eg in range(self.param['batch_size']):
                                regression += [out[1][eg,:]]
                        perform_avg += perform_curr
                        loss_avg += loss_curr
                        duration = time.time() - start_val
                        bracket = int(10*it*self.param['batch_size']/self.nseqs['val'])%10 * '■'
                        print('| Wait for Validating # %-5d/%5d: %4.3fs elapsed [%-9s] acc -> %.4f |  '
                              %(it*self.param['batch_size'], self.nseqs['val'], duration, bracket, perform_curr), end='\r')
                    perform_avg /= iteration_val
                    loss_avg /= iteration_val
                    # -------------- check if this is best epoch for now-------------- #
                    regression = np.array(regression).astype(np.float32)[0:self.nseqs['val']]
                    ccc_a, ccc_v = self.ccc(self.ground_truth, regression)
                    ccc = (ccc_a+ccc_v)/2
                    if ccc_a > max_a:
                        max_a = ccc_a
                        flag_update = True
                    if ccc_v > max_v:
                        max_v = ccc_v
                        flag_update = True
                    if ccc > max_c:
                        max_c = ccc
                        flag_update = True
                    if (flag_update or \
                        int(self.param['max_steps']/
                        (int(self.nseqs['train']/self.param['batch_size'])+1)) == self.epoch):
                        start_test = time.time()
                        self.saver.save(self.sess, ckpt_file, global_step=self.global_step)
                        hdf = h5py.File(os.path.join(ckpt_dir,'vid_feat_%d.hdf5'%self.epoch), 'w')
                        scribe = open(os.path.join(ckpt_dir,'output_tr_%d.csv'%self.epoch), 'w')
                        writer = csv.writer(scribe, delimiter=',')
                        writer.writerow(['arousal', 'valence'])
                        iteration_test = self.nseqs['test'] // self.param['batch_size'] + 1
                        for it in range(iteration_test):
                            feats, images, labels, numbers, frames = self.fetchEg(one_hot=self.param['one_hot'])
                            if self.param['enable_img']:
                                out = self.sess.run([self.test_output, self.feat],
                                    feed_dict={self.test_feat: feats, self.test_lab: labels,
                                               self.test_num: numbers, self.test_len: frames})
                            else:
                                out = self.sess.run([self.test_output, self.feat],
                                    feed_dict={self.test_img: images, self.test_lab: labels,
                                               self.test_num: numbers, self.test_len: frames})
                            for eg in range(self.param['batch_size']):
                                if self.param['num_value'] > 0:
                                    writer.writerow([str(vl) for vl in out[0][1][eg, :]])
                                if self.param['feat_gate'] and len(hdf.keys())<self.nseqs['test']:
                                    hdf[self.eg_order[eg]] = out[1][eg, :]
                            duration = time.time() - start_test
                            bracket = int(10 * it * self.param['batch_size'] / self.nseqs['test']) % 10 * '■'
                            print('| Wait for Testing # %-5d/%5d: %4.3fs elapsed [%-9s] recording output |  '
                                  % (it * self.param['batch_size'], self.nseqs['test'], duration, bracket), end='\r')
                        scribe.close()
                        hdf.close()
                    duration = time.time() - start_val
                    print(' ', end='\r')
                    # print('| Epoch #%-2d: %3.3fs/test - Acc: t/v->%.4f|%.4f - Loss: t/v->%.4f|%.4f%s'
                    #       %(self.epoch, duration, acc_epoch, perform_avg, loss_epoch, loss_avg, ' |\n+'+78*'-'+'+'))
                    print('| Epoch #%-2d: %3.3fs/test - ccc: a/v->%.4f|%.4f - Loss: t/v->%.4f|%.4f%s'
                          %(self.epoch, duration, ccc_a, ccc_v, loss_epoch, loss_avg, ' |\n+' + 78 * '-' + '+'))
                    self.writer.writerow([self.epoch, ccc_a, ccc_v, loss_epoch, loss_avg])


                start = time.time()
                feats, images, labels, numbers, frames = self.nextBatch(one_hot=self.param['one_hot'])
                if self.param['enable_img']:
                    _, loss_total, accuracy, pr, lrn = self.sess.run([self.train_op, self.cost_total, self.performance,
                                                                      self.alpha, self.output_rnn],
                                                                     feed_dict={self.input_feat: feats,
                                                                                self.input_lab: labels,
                                                                                self.input_num: numbers,
                                                                                self.input_len: frames})
                else:
                    _, loss_total, accuracy, pr, lrn = self.sess.run([self.train_op, self.cost_total, self.performance,
                                                                      self.alpha, self.output_rnn],
                                                                     feed_dict={self.input_img: images,
                                                                                self.input_lab: labels,
                                                                                self.input_num: numbers,
                                                                                self.input_len: frames})
                # print(frames.shape)
                # print(30 * '-')
                # print(lrn[0])
                # print(30 * '-')
                # if step>0:
                #     print(pr[0])
                #     import pdb
                #     pdb.set_trace()

                if step % self.param['inspect'] == 0:
                    summ_acc += [accuracy]
                    summ_loss += [loss_total]
                    duration = time.time()-start
                    bracket = int(10*(step-prev_end)*self.param['batch_size']/self.nseqs['train'])%10 * '■'
                    print('| Step #%5dx%-3d@%4d: %2.3fs/batch [%-9s] acc-> %.3f, loss-> %.5f  |  '
                          %(step, self.param['batch_size'], self.nseqs['train'], duration, bracket,
                            accuracy, loss_total), end='\r')
                # if step % self.param['interval'] == 0:
                #     self.saver.save(self.sess, ckpt_file, global_step=self.global_step)
            print('See u next script'+70*' ')
            self.scribe.close()

    def getAvailableGpus(self):
        local_device_protos = device_lib.list_local_devices()
        max_limit = 0
        device_name = ''
        for d, x in enumerate(local_device_protos):
            if x.device_type == "GPU":
                print('device: %s -> memory: %.3f' % (x.name, x.memory_limit))
            if x.device_type == "GPU" and x.memory_limit > max_limit:
                device_name = x.name
                max_limit = x.memory_limit
        if device_name:
            return device_name.split(":")[-1]
        else:
            # no available gpu
            return ''

if __name__ == '__main__':
    rnn = RNN_Seq2Sng()
    rnn.train()