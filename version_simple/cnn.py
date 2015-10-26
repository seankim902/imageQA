# -*- coding: utf-8 -*-


import caffe
import numpy as np
from scipy.misc import imread, imresize

class CNN(object):

    def __init__(self, deploy='VGG_ILSVRC_16_layers_deploy.prototxt', model='VGG_ILSVRC_16_layers.caffemodel'):
        caffe.set_mode_gpu()
        self.net = caffe.Net(deploy, model, caffe.TEST)
       
        #if model.startswith('VGG'):
        self.mean = np.array([103.939, 116.779, 123.68])

            
    def get_batch_features(self, in_data, net, layer):
        out = net.forward(blobs=[layer], **{net.inputs[0]: in_data})
        features = out[layer]#.squeeze(axis=(2,3))
        return features
    
    def get_features(self, filelist, layer='fc7'):
        
        N, channel, height, width = self.net.blobs[self.net.inputs[0]].data.shape
        n_files = len(filelist)

        
        if  str(layer).startswith('fc'):
            feature = self.net.blobs[layer].data.shape[1]
            all_features = np.zeros((n_files, feature))
        else :
            feature1, feature2, feature3 = self.net.blobs[layer].data.shape[1], self.net.blobs[layer].data.shape[2], self.net.blobs[layer].data.shape[3]
            all_features = np.zeros((n_files, feature1, feature2, feature3))
        
        
        for i in range(0, n_files, N):
            in_data = np.zeros((N, channel, height, width), dtype=np.float32)
    
            batch_range = range(i, min(i+N, n_files))
            batch_filelist = [filelist[j] for j in batch_range]
                
            batch_images = np.zeros((len(batch_range), 3, height, width))
            for j,file in enumerate(batch_filelist):
                im = imread(file)
                if len(im.shape) == 2:
                    im = np.tile(im[:,:,np.newaxis], (1,1,3))
                im = im[:,:,(2,1,0)] # RGB -> BGR
                im = im - self.mean # mean subtraction
                im = imresize(im, (height, width), 'bicubic') # resize
                im = np.transpose(im, (2, 0, 1)) # get channel in correct dimension
                batch_images[j,:,:,:] = im
                
            in_data[0:len(batch_range), :, :, :] = batch_images
    
            features = self.get_batch_features(in_data, self.net, layer)
    
            for j in range(len(batch_range)):
                all_features[i+j,:] = features[j,:]
    
            #print 'Done %d/%d files' % (i+len(batch_range), len(filelist))
    
        return all_features
