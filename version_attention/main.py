# -*- coding: utf-8 -*-


import os
import pandas as pd
import numpy as np
from models import RNN_GRU, RNN_LSTM, BIRNN_GRU
from keras.utils import np_utils

from sklearn.cross_validation import train_test_split


def prepare_data(seqs_x, maxlen=None):
    lengths_x = [len(s) for s in seqs_x]
    
    n_samples = len(seqs_x)
    if maxlen is None:
        maxlen = np.max(lengths_x) + 1
    
    x = np.zeros((maxlen, n_samples)).astype('int32')
    x_mask = np.zeros((maxlen, n_samples)).astype('float32')
    
    for idx, s_x in enumerate(seqs_x):
        x[:lengths_x[idx],idx] = s_x
        #x_mask[:lengths_x[idx],idx] = 1.
        x_mask[:lengths_x[idx],idx] = 1.

    return x, x_mask
    
    
    
def main():
    
    



    os.chdir('/home/seonhoon/Desktop/workspace/ImageQA/data/')

    n_vocab = 12047
    y_vocab = 430
    dim_word = 1024
    dim = 1024
    dim_ctx = 512
    maxlen = 60
    
    train=pd.read_pickle('train.pkl')
 #   train, valid = train_test_split(train, test_size=0.2)
    valid=pd.read_pickle('test.pkl')  
    
    train_x=[ q for q in train['q'] ]
    train_y=[ a[0] for a in train['a'] ]
    train_y=np.array(train_y)[:,None]
    train_y = np_utils.to_categorical(train_y, y_vocab).astype('int32')
    train_x , train_x_mask = prepare_data(train_x, maxlen)


    img_folder='/home/seonhoon/Desktop/workspace/ImageQA/data/images/'
    train_imgs = train['img_id'].apply(lambda x : img_folder+'train/'+x+'.jpg')
    train_imgs = train_imgs.tolist()
    
    valid_x=[ q for q in valid['q'] ]
    valid_y=[ a[0] for a in valid['a'] ]
    valid_x , valid_x_mask = prepare_data(valid_x, maxlen)

    valid_imgs = valid['img_id'].apply(lambda x : img_folder+'test/'+x+'.jpg')
    valid_imgs = valid_imgs.tolist()
    
    print 'train x :', train_x.shape
    print 'train x_mask:', train_x_mask.shape
    print 'train y : ', train_y.shape
    print 'train imgs :', len(train_imgs)
    
    print 'valid x :', valid_x.shape
    print 'valid x_mask:', valid_x_mask.shape
    #print 'valid y : ', valid_y.shape
    print 'valid imgs :', len(valid_imgs)

    model = BIRNN_GRU(n_vocab, y_vocab, dim_word, dim, dim_ctx)

    model.train(train_x, train_x_mask, train_imgs, train_y, 
                valid_x, valid_x_mask, valid_imgs, valid_y, valid=2,
                lr=0.008, dropout=0.4, batch_size=512, epoch=60, save=4)
    
if __name__ == '__main__':
    main() 
    
    