# -*- coding: utf-8 -*-

import urllib
import pandas as pd
import os
import shutil

os.chdir("/home/seonhoon/Desktop/workspace/ImageQA/data/images/")



i=0

def downloadImage(id, location):
    global i
    i=i+1
    if i%1000==0 :
        print 'iter : ', i
    fn=id.zfill(12)
    if os.path.isfile('/home/seonhoon/Downloads/test2014/COCO_test2014_'+fn+'.jpg'):
        print 'shutil.copy2 test ', id
        shutil.copy2('/home/seonhoon/Downloads/test2014/COCO_test2014_'+fn+'.jpg', location+'/'+id+".jpg")
    elif os.path.isfile('/home/seonhoon/Downloads/train2014/COCO_train2014_'+fn+'.jpg'):
        print 'shutil.copy2 train ', id        
        shutil.copy2('/home/seonhoon/Downloads/train2014/COCO_train2014_'+fn+'.jpg', location+'/'+id+".jpg")
    elif os.path.isfile(location+'/'+id+".jpg"):
        print 'location ', id
    else:
        print 'download ', id
        urllib.urlretrieve("http://mscoco.org/images/"+id, location+'/'+id+".jpg")

#train=pd.read_pickle('/home/seonhoon/Desktop/workspace/ImageQA/data/train.pkl')

#train['img_id'].apply(lambda x : downloadImage(x, 'train'))

test=pd.read_pickle('/home/seonhoon/Desktop/workspace/ImageQA/data/test.pkl')

test['img_id'].apply(lambda x : downloadImage(x, 'test'))