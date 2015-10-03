# -*- coding: utf-8 -*-


import os
import pandas as pd
import numpy as np
from models import RNN
from sklearn.preprocessing import OneHotEncoder

def prepare_data(seqs_x, maxlen=None):
    lengths_x = [len(s) for s in seqs_x]
    
    n_samples = len(seqs_x)
    maxlen_x = np.max(lengths_x) # + 1
    
    x = np.zeros((maxlen_x, n_samples)).astype('int64')
    x_mask = np.zeros((maxlen_x, n_samples)).astype('float32')
    
    for idx, s_x in enumerate(seqs_x):
        x[:lengths_x[idx],idx] = s_x
        x_mask[:lengths_x[idx],idx] = 1.
        #x_mask[:lengths_x[idx]+1,idx] = 1.

    return x, x_mask
    
    
    
def main():
    os.chdir('/Users/seonhoon/Desktop/workspace_python/ImageQA/data/')
    
    train=pd.read_pickle('train.pkl')
    test=pd.read_pickle('test.pkl')
    

    train_x=[ q for q in train['q'] ]
    train_y=[ a[0] for a in train['a'] ]
    train_y=np.array(train_y)[:,None]
    enc = OneHotEncoder()
    enc.fit(train_y)
    train_y=enc.transform(train_y).toarray()
    train_x , train_x_mask = prepare_data(train_x)
    
    test_x=[ q for q in test['q'] ]
    test_y=[ a[0] for a in test['a'] ]
    test_y=np.array(test_y)[:,None]
    enc = OneHotEncoder()
    enc.fit(test_y)
    test_y=enc.transform(test_y).toarray()
    test_x , test_x_mask = prepare_data(test_x)
    
    
    
    n_vocab = 12047
    y_vocab = 430
    dim_word = 1024
    dim = 512
    
    
    model = RNN(n_vocab, y_vocab, dim_word, dim)
    model.train(train_x, train_x_mask, train_y,batch_size=6000)
    
if __name__ == '__main__':
    main() 
    
    