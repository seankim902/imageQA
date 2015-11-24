# -*- coding: utf-8 -*-

from cnn import CNN
import pandas as pd


cnn = CNN()
data=pd.DataFrame()

data['cnn_feature']=0
data['cnn_feature']=data['cnn_feature'].astype(object)

featurelist = cnn.get_features(['/home/seonhoon/Desktop/workspace/ImageQA/version_tensorflow/web/images/moodo.jpg'], layer='fc7')
data.loc[0,'cnn_feature']=featurelist[0].flatten()


data.to_pickle('/home/seonhoon/Desktop/workspace/ImageQA/version_tensorflow/web/cnn.pkl')
