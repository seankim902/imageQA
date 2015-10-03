# -*- coding: utf-8 -*-

import pandas as pd
import os
import cPickle

os.chdir('/Users/seonhoon/Desktop/workspace_python/ImageQA/data/')


with open('dict.pkl', 'rb') as f:
    dict=cPickle.load(f)
idx2word=dict[0]
word2idx=dict[1]
idx2answer=dict[2]
answer2idx=dict[3]

train=pd.read_pickle('train.pkl')
test=pd.read_pickle('test.pkl')

def getIndex(x, type='question'):
    xs=x.split()
    idx=[]
    getIdx=word2idx
    if type=='answer':
        getIdx=answer2idx
    for x in xs:
        idx.append(getIdx[x])
    return idx
    
train['q']=train['question'].apply(lambda x : getIndex(x))
train['a']=train['answer'].apply(lambda x : getIndex(x,type='answer'))
test['q']=test['question'].apply(lambda x : getIndex(x))
test['a']=test['answer'].apply(lambda x : getIndex(x,type='answer'))

train.to_pickle('train.pkl')
test.to_pickle('test.pkl')
