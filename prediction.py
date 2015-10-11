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
    
    x = np.zeros((maxlen, n_samples)).astype('int64')
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
    
    train=pd.read_pickle('test.pkl')

    train_x=[ q for q in train['q'] ]
    train_y=[ a[0] for a in train['a'] ]
    train_y_original=train_y
    train_y=np.array(train_y)[:,None]
    train_y = np_utils.to_categorical(train_y, y_vocab)
    train_x , train_x_mask = prepare_data(train_x, maxlen)
  #  train_img = [ img.tolist() for img in train['cnn_feature'] ]

    
    print 'x :', train_x.shape
    print 'x_mask:', train_x_mask.shape
    print 'y : ', train_y.shape
    model = RNN(n_vocab, y_vocab, dim_word, dim)

    pred_y = model.prediction(train_x, train_x_mask, train_y, batch_size=2048)
    
    print pred_y[:10], len(pred_y)
    print train_y_original[:10], len(train_y_original)
    
    correct = 0 
    for i in range(len(pred_y)):
        if pred_y[i]==train_y_original[i] : 
            correct += 1
    print 'accuracy : ', float(correct) / len(pred_y)
if __name__ == '__main__':
    main() 
    
    