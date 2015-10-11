# -*- coding: utf-8 -*-


from cnn import *
import pandas as pd
import os


os.chdir('/home/seonhoon/Desktop/workspace/ImageQA/data/')
train=pd.read_pickle('train.pkl')
test=pd.read_pickle('test.pkl')


os.chdir('/home/seonhoon/Desktop/workspace/ImageQA/data/caffemodel/vgg16')

#/home/seonhoon/Desktop/workspace/ImageQA/data/images/train/xxx.jpg
img_folder='/home/seonhoon/Desktop/workspace/ImageQA/data/images/'

# train
print 'train .. '
imglist = train['img_id'].apply(lambda x : img_folder+'train/'+x+'.jpg')
imglist = imglist.tolist()
cnn = CNN()
featurelist = cnn.get_features(imglist)
train['cnn_feature']=0
train['cnn_feature']=train['cnn_feature'].astype(object)
for i in range(len(train)):
    train.loc[i,'cnn_feature']=featurelist[i]



#test
print 'test .. '
imglist = test['img_id'].apply(lambda x : img_folder+'test/'+x+'.jpg')
imglist = imglist.tolist()
cnn = CNN()
featurelist = cnn.get_features(imglist)
test['cnn_feature']=0
test['cnn_feature']=test['cnn_feature'].astype(object)
for i in range(len(test)):
    test.loc[i,'cnn_feature']=featurelist[i]

#modify pickle
os.chdir('/home/seonhoon/Desktop/workspace/ImageQA/data/')
train.to_pickle('train_vgg.pkl')
test.to_pickle('test_vgg.pkl')
