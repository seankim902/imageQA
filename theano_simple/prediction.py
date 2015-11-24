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
    
    test=pd.read_pickle('test_vgg.pkl')

    test_x=[ q for q in test['q'] ]
    test_y=[ a[0] for a in test['a'] ]
    test_y_original=test_y
    test_y=np.array(test_y)[:,None]
    test_y = np_utils.to_categorical(test_y, y_vocab)
    test_x , test_x_mask = prepare_data(test_x, maxlen)
    test_x_img = [ img.tolist() for img in test['cnn_feature'] ]

    
    print 'x :', test_x.shape
    print 'x_mask:', test_x_mask.shape
    print 'y : ', test_y.shape
    model = RNN(n_vocab, y_vocab, dim_word, dim)

    pred_y = model.prediction(test_x, test_x_mask, test_x_img, test_y, batch_size=2048)
    
    print pred_y[:10], len(pred_y)
    print test_y_original[:10], len(test_y_original)
    
    correct = 0 
    for i in range(len(pred_y)):
        if pred_y[i]==test_y_original[i] : 
            correct += 1
    print 'accuracy : ', float(correct) / len(pred_y)
if __name__ == '__main__':
    main() 