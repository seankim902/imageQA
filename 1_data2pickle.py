# -*- coding: utf-8 -*-

import pandas as pd
import os

### test data to pickle
os.chdir('/Users/seonhoon/Desktop/workspace_python/ImageQA/data/test')
test=pd.DataFrame()

data=[]
with open('img_ids.txt') as f:
    for line in f:
        data.append(line.split('\n')[0])
test['img_id']=data

data=[]
with open('questions.txt') as f:
    for line in f:
        data.append(line.split('\n')[0])
test['question']=data

data=[]
with open('answers.txt') as f:
    for line in f:
        data.append(line.split('\n')[0])
test['answer']=data

data=[]
with open('types.txt') as f:
    for line in f:
        data.append(line.split('\n')[0])
test['type']=data

test.to_pickle('../test.pkl')


### train data to pickle
os.chdir('/Users/seonhoon/Desktop/workspace_python/ImageQA/data/train')
train=pd.DataFrame()

data=[]
with open('img_ids.txt') as f:
    for line in f:
        data.append(line.split('\n')[0])
train['img_id']=data

data=[]
with open('questions.txt') as f:
    for line in f:
        data.append(line.split('\n')[0])
train['question']=data

data=[]
with open('answers.txt') as f:
    for line in f:
        data.append(line.split('\n')[0])
train['answer']=data

data=[]
with open('types.txt') as f:
    for line in f:
        data.append(line.split('\n')[0])
train['type']=data

train.to_pickle('../train.pkl')
