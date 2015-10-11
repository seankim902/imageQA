# -*- coding: utf-8 -*-
import pandas as pd
import os
import cPickle

os.chdir('/Users/seonhoon/Desktop/workspace_python/ImageQA/data/')

train=pd.read_pickle('train.pkl')
test=pd.read_pickle('test.pkl')

questions = train['question'].tolist()+test['question'].tolist()
questions = [question.split() for question in questions]

tokens=[]

for question in questions:
    for word in question:
            tokens.append(word)

tokens = list(set(tokens))
tokens.sort()

idx2word = dict([(i,k) for i,k in enumerate(tokens)])
word2idx = dict([(k,i) for i,k in enumerate(tokens)])




answers = list(set(train['answer'].tolist()+test['answer'].tolist()))
answers.sort()

idx2answer = dict([(i,k) for i,k in enumerate(answers)])
answer2idx = dict([(k,i) for i,k in enumerate(answers)])

with open('dict.pkl', 'wb') as f:
    cPickle.dump([idx2word, word2idx, idx2answer, answer2idx], f)