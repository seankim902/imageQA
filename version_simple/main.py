# -*- coding: utf-8 -*-


import os
import pandas as pd
import numpy as np
from models import RNN
from keras.utils import np_utils


def prepare_data(seqs_x, maxlen=None):
    lengths_x = [len(s) for s in seqs_x]
    
    n_samples = len(seqs_x)
    if maxlen is None:
        maxlen = np.max(lengths_x) + 1
    
    x = np.zeros((maxlen, n_samples)).astype('int32')
    x_mask = np.zeros((maxlen, n_samples)).astype('float32')
    
    for idx, s_x in enumerate(seqs_x):
        x[:lengths_x[idx],idx] = s_x
        x_mask[:lengths_x[idx],idx] = 1.
        #x_mask[:lengths_x[idx]+1,idx] = 1.

    return x, x_mask
    
    
    
def main():
    os.chdir('/home/seonhoon/Desktop/workspace/ImageQA/data/')

    n_vocab = 12047
    y_vocab = 430
    dim_word = 1024
    dim = 1024
    maxlen = 60
    
    train=pd.read_pickle('train_vgg.pkl')



    train_x=[ q for q in train['q'] ]
    train_y=[ a[0] for a in train['a'] ]
    train_y=np.array(train_y)[:,None]
    train_y = np_utils.to_categorical(train_y, y_vocab).astype('int32')
    train_x , train_x_mask = prepare_data(train_x, maxlen)
    train_x_img = np.array([ img.tolist() for img in train['cnn_feature'] ]).astype('float32')

    
    print 'x :', train_x.shape
    print 'x_mask:', train_x_mask.shape
    print 'x_img:', train_x_img.shape
    print 'y : ', train_y.shape
    model = RNN(n_vocab, y_vocab, dim_word, dim)

    model.train(train_x, train_x_mask, train_x_img, train_y, batch_size=512, epoch=50, save=15)
    
if __name__ == '__main__':
    main() 
    