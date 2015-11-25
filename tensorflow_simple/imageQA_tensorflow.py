# -*- coding: utf-8 -*-

import os
import time
import pandas as pd
import numpy as np

from keras.utils import np_utils

#import tensorflow.python.platform
import tensorflow as tf
from tensorflow.models.rnn import rnn_cell


# To do
'''
마지막 배치 사이즈 이슈


'''

def prepare_data(seqs_x, maxlen=None):
    lengths_x = [len(s) for s in seqs_x]
    
    n_samples = len(seqs_x)
    if maxlen is None:
        maxlen = np.max(lengths_x) + 1
    
    x = np.zeros((n_samples, maxlen)).astype('int32')
    x_mask = np.zeros((n_samples, maxlen)).astype('float32')
    
    for idx, s_x in enumerate(seqs_x):
        x[idx, :lengths_x[idx]] = s_x
        x_mask[idx, :lengths_x[idx]+1] = 1. # Adding 1, for image

    return x, x_mask
    

def get_minibatch_indices(n, batch_size, shuffle=False):

    idx_list = np.arange(n, dtype="int32")

    if shuffle:
        np.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // batch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + batch_size])
        minibatch_start += batch_size
#    if (minibatch_start != n):   # last mini-batch issue !!! 
#        minibatches.append(idx_list[minibatch_start:])
    return minibatches
    
    

class ImageQA(object):

    def __init__(self, config):
        
        self.config = config

        self.vocab_size = vocab_size = config.vocab_size
        self.y_size = y_size = config.y_size

        self.batch_size = batch_size = config.batch_size
        self.steps = config.steps
        
        self.layers = layers = config.layers
              
        self.dim_ictx = dim_ictx = config.dim_ictx      
        self.dim_iemb = dim_iemb = config.dim_iemb
        self.dim_wemb = dim_wemb = config.dim_wemb
        self.dim_hidden = dim_hidden = config.dim_hidden
        
        self.lr = tf.Variable(config.lr, trainable=False)
                
        rnn_type = config.rnn_type
        if rnn_type == 'gru':
            rnn_ = rnn_cell.GRUCell(dim_hidden)
        elif rnn_type == 'lstm':
            rnn_ = rnn_cell.BasicLSTMCell(dim_hidden)
            
        if layers is not None:
            self.my_rnn = my_rnn = rnn_cell.MultiRNNCell([rnn_] * layers)
            self.init_state = my_rnn.zero_state(batch_size, tf.float32)
        else:
            self.my_rnn = my_rnn = rnn_
            self.init_state = tf.zeros([batch_size, my_rnn.state_size])

        self.W_iemb = tf.get_variable("W_iemb", [dim_ictx, dim_iemb])
        self.b_iemb = tf.get_variable("b_iemb", [dim_iemb])
        with tf.device("/cpu:0"):
            self.W_wemb = tf.get_variable("W_wemb", [vocab_size, dim_wemb])
        
        if config.is_birnn : # add 보다 concat이 더 잘나오는듯..
            self.W_pred = tf.get_variable("W_pred", [dim_hidden * 2, y_size])
        else :
            self.W_pred = tf.get_variable("W_pred", [dim_hidden, y_size])

        self.b_pred = tf.get_variable("b_pred", [y_size])
        
        
    def build_model(self):
        
        x = tf.placeholder(tf.int32, [self.batch_size, self.steps])
        x_mask = tf.placeholder(tf.float32, [self.batch_size, self.steps])
        y = tf.placeholder(tf.float32, [self.batch_size, self.y_size])
        img = tf.placeholder(tf.float32, [self.batch_size, self.dim_ictx])

        

        with tf.device("/cpu:0"):
            inputs = tf.split(1, self.steps, tf.nn.embedding_lookup(self.W_wemb, x))
            # sample * steps * dim -> split -> sample * 1 * dim
        inputs = [tf.squeeze(input_, [1]) for input_ in inputs]
        # [sample * dim, sample * dim, sample * dim, ... ]
        img_emb = tf.nn.xw_plus_b(img, self.W_iemb, self.b_iemb)
        inputs = [img_emb]+inputs[:-1] # -1 is for img
        
      

        hiddens = []
        states = []
        
        
        state = self.init_state
        with tf.variable_scope("RNN", reuse=None):
            for i in range(len(inputs)):           
                if i == 0:
                    (hidden, state) = self.my_rnn(inputs[i], state)
                else: 
                    m = x_mask[:, i]
                    
                    tf.get_variable_scope().reuse_variables()
                    (prev_hidden, prev_state) = (hidden, state) # for masking
                    (hidden, state) = self.my_rnn(inputs[i], state)

                    m_1 = tf.expand_dims(m,1)
                    m_1 = tf.tile(m_1, [1, self.dim_hidden]) 
                    m_0 = tf.expand_dims(1. - m,1)
                    m_0 = tf.tile(m_0, [1, self.dim_hidden])
                    hidden = tf.add(tf.mul(m_1, hidden), tf.mul(m_0, prev_hidden))
                    state = tf.add(tf.mul(m_1, state), tf.mul(m_0, prev_state))
                hiddens.append(hidden)
                states.append(state)

        if self.config.is_birnn :
            rhiddens = []
            rstates = []
            rx_mask = tf.reverse(x_mask,[False, True])            
            rinputs = inputs[::-1]
            state = self.init_state
            with tf.variable_scope("rRNN", reuse=None):
                for i in range(len(rinputs)):
                    if i == 0:
                        (hidden, state) = self.my_rnn(inputs[i], state)
                    else: 
                        m = rx_mask[:, i]
                        
                        tf.get_variable_scope().reuse_variables()
                        (prev_hidden, prev_state) = (hidden, state) # for masking
                        (hidden, state) = self.my_rnn(rinputs[i], state)
                        m_1 = tf.expand_dims(m,1)
                        m_1 = tf.tile(m_1, [1, self.dim_hidden]) 
                        m_0 = tf.expand_dims(1. - m,1)
                        m_0 = tf.tile(m_0, [1, self.dim_hidden])
                        hidden = tf.add(tf.mul(m_1, hidden), tf.mul(m_0, prev_hidden))
                        state = tf.add(tf.mul(m_1, state), tf.mul(m_0, prev_state))
                    rhiddens.append(hidden)
                    rstates.append(state)           
            
        
            hiddens = tf.concat(2, [tf.pack(hiddens), tf.pack(rhiddens[::-1])])
            #hiddens = tf.add(tf.pack(hiddens), tf.pack(rhiddens[::-1]))
            hiddens = tf.unpack(hiddens)

        '''
        mean or last hidden -> logit_hidden : sample * dim
        '''
        # 1. last hidden
        #logit_hidden = hiddens[-1] #tf.reduce_mean(hiddens, 0)    
        # 2. mean of hiddens
        x_mask_t = tf.transpose(x_mask)
        x_mask_t_denom = tf.expand_dims(tf.reduce_sum(x_mask_t, 0), 1)
        x_mask_t_denom = tf.tile(x_mask_t_denom, [1, self.W_pred.get_shape().dims[0].value])
        x_mask_t = tf.expand_dims(x_mask_t, 2)
        x_mask_t = tf.tile(x_mask_t, [1, 1, self.W_pred.get_shape().dims[0].value])
        hiddens = tf.pack(hiddens)
        logit_hidden = tf.reduce_sum(tf.mul(hiddens, x_mask_t), 0)
        logit_hidden = tf.div(logit_hidden, x_mask_t_denom)
        
        
        if self.config.dropout is not None:
            logit_hidden = tf.nn.dropout(logit_hidden, self.config.dropout)
  
          
        logits = tf.nn.xw_plus_b(logit_hidden, self.W_pred, self.b_pred)

        probs = tf.nn.softmax(logits)
        prediction = tf.argmax(probs, 1)
        correct_prediction = tf.equal(prediction, tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, y)
        loss = tf.reduce_mean(cross_entropy)
        train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)
        return x, x_mask, y, img, loss, train_op, accuracy, prediction


def train():


    config = get_config()

    os.chdir('/home/seonhoon/Desktop/workspace/ImageQA/data/')
    train = pd.read_pickle('train_vgg.pkl')
    train_x = [ q for q in train['q'] ]
    train_y = [ a[0] for a in train['a'] ]
    train_y = np.array(train_y)[:,None]
    train_y = np_utils.to_categorical(train_y, config.y_size).astype('float32')
    train_x , train_x_mask = prepare_data(train_x, config.steps)
    train_x_img = np.array([ img.tolist() for img in train['cnn_feature'] ]).astype('float32')

    n_train = len(train_x)
    
    print 'train_x :', train_x.shape
    print 'train_x_mask :', train_x_mask.shape
    print 'train_x_img :', train_x_img.shape
    print 'train_y :',train_y.shape
    
    if config.valid_epoch is not None:
        valid=pd.read_pickle('test_vgg.pkl')    
        valid_x=[ q for q in valid['q'] ]
        valid_y=[ a[0] for a in valid['a'] ]
        valid_y=np.array(valid_y)[:,None]
        valid_y = np_utils.to_categorical(valid_y, config.y_size).astype('float32')
        valid_x , valid_x_mask = prepare_data(valid_x, config.steps)
        valid_x_img = np.array([ img.tolist() for img in valid['cnn_feature'] ]).astype('float32')
        n_valid = len(valid_x)
        valid_batch_indices=get_minibatch_indices(n_valid, config.batch_size, shuffle=False)
    
        print 'valid_x :', valid_x.shape
        print 'valid_x_mask :', valid_x_mask.shape
        print 'valid_x_img :', valid_x_img.shape
        print 'valid_y :',valid_y.shape
        

    
    with tf.Session() as sess:
        
        
        initializer = tf.random_normal_initializer(0, 0.1)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            model = ImageQA(config = config)
            
        x, x_mask, y, img, loss, train_op, accuracy, prediction = model.build_model()
        
        saver = tf.train.Saver()        
        sess.run(tf.initialize_all_variables())
        
        for i in range(config.epoch):
            start = time.time()
            lr_decay = config.lr_decay ** max(i - config.decay_epoch, 0.0)
            sess.run(tf.assign(model.lr, config.lr * lr_decay))


            batch_indices=get_minibatch_indices(n_train, config.batch_size, shuffle=True)

            preds = []
            for j, indices in enumerate(batch_indices):

                x_ = np.array([ train_x[k,:] for k in indices])
                x_mask_ = np.array([ train_x_mask[k,:] for k in indices])
                y_ = np.array([ train_y[k,:] for k in indices])
                img_ = np.array([ train_x_img[k,:] for k in indices])
                
                  
                
                cost, _, acc, pred = sess.run([loss, train_op, accuracy, prediction],
                                              {x: x_,
                                               x_mask: x_mask_,
                                               y: y_,
                                               img : img_})
                preds = preds + pred.tolist()
                if j % 99 == 0 :
                    print 'cost : ', cost, ', accuracy : ', acc, ', iter : ', j+1, ' in epoch : ',i+1
            print 'cost : ', cost, ', accuracy : ', acc, ', iter : ', j+1, ' in epoch : ',i+1,' elapsed time : ', int(time.time()-start)
            if config.valid_epoch is not None:  # for validation
                best_accuracy = 0.
                
                if (i+1) % config.valid_epoch == 0:
                    val_preds = []
                    for j, indices in enumerate(valid_batch_indices):
                        x_ = np.array([ valid_x[k,:] for k in indices])
                        x_mask_ = np.array([ valid_x_mask[k,:] for k in indices])
                        y_ = np.array([ valid_y[k,:] for k in indices])
                        img_ = np.array([ valid_x_img[k,:] for k in indices])
                        
                        pred = sess.run(prediction,
                                        {x: x_,
                                         x_mask: x_mask_,
                                         y: y_,
                                         img : img_})
               
                        val_preds = val_preds + pred.tolist()
                    valid_acc = np.mean(np.equal(val_preds, np.argmax(valid_y,1)))
                    print '##### valid accuracy : ', valid_acc, ' after epoch ', i+1
                    if valid_acc > best_accuracy and i >= 10:
                        best_accuracy = valid_acc
                        saver.save(sess, config.model_ckpt_path, global_step=int(best_accuracy*100))

                    

def test():                    

    config = get_config()

    os.chdir('/home/seonhoon/Desktop/workspace/ImageQA/data/')

    test=pd.read_pickle('test_vgg.pkl')    
    test_x=[ q for q in test['q'] ]
    test_y=[ a[0] for a in test['a'] ]
    test_y=np.array(test_y)[:,None]
    test_y = np_utils.to_categorical(test_y, config.y_size).astype('float32')
    test_x , test_x_mask = prepare_data(test_x, config.steps)
    test_x_img = np.array([ img.tolist() for img in test['cnn_feature'] ]).astype('float32')
    n_test = len(test_x)
    test_batch_indices=get_minibatch_indices(n_test, config.batch_size, shuffle=False)
    
    
    
    with tf.Session() as sess:
        
        
        with tf.variable_scope("model", reuse=None):
            model = ImageQA(config = config) 
            
        x, x_mask, y, img, _, _, _, prediction = model.build_model()
        saver = tf.train.Saver()        
        ckpt = tf.train.get_checkpoint_state(os.path.dirname(config.model_ckpt_path))
        sess.run(tf.initialize_all_variables())
        saver.restore(sess, ckpt.model_checkpoint_path)
        test_preds = []
        for j, indices in enumerate(test_batch_indices):
            x_ = np.array([test_x[k,:] for k in indices])
            x_mask_ = np.array([ test_x_mask[k,:] for k in indices])
            y_ = np.array([ test_y[k,:] for k in indices])
            img_ = np.array([ test_x_img[k,:] for k in indices])
            
            pred = sess.run(prediction,
                            {x: x_,
                             x_mask: x_mask_,
                             y: y_,
                             img : img_})
            test_preds = test_preds + pred.tolist()


        test_acc = np.mean(np.equal(test_preds, np.argmax(test_y,1)))
        print 'test accuracy :', test_acc


def test_sample(test_x, test_x_maks, test_x_img):                    

    config = get_config()
    config.batch_size = 1

    with tf.Session() as sess:
        with tf.variable_scope("model", reuse=None):
            model = ImageQA(config = config) 
            
        x, x_mask, _, img, _, _, _, prediction = model.build_model()
        saver = tf.train.Saver()        
        ckpt = tf.train.get_checkpoint_state(os.path.dirname(config.model_ckpt_path))
        saver.restore(sess, ckpt.model_checkpoint_path)

        x_ = test_x
        x_mask_ = test_x_maks
        img_ = test_x_img
        
        pred = sess.run(prediction,
                        {x: x_,
                         x_mask: x_mask_,
                         img : img_})
        
        return pred




def get_config():
    class Config1(object):
        vocab_size = 12047
        y_size = 430
        batch_size = 364 #703 #s28
        steps = 60

        dim_ictx = 4096
        dim_iemb = 1024 # image embedding
        dim_wemb = 1024 # word embedding
        dim_hidden = 1024
        epoch = 100
        
        lr = 0.001
        lr_decay = 0.9
        decay_epoch = 333. # epoch 보다 크면 decay 하지않음.
        
        dropout = 0.4
        
        rnn_type = 'gru'
        layers = None #it doesn’t work yet
        is_birnn = True #False
        valid_epoch = 1 # or None
        model_ckpt_path = '/home/seonhoon/Desktop/workspace/ImageQA/version_tensorflow/model/model.ckpt'
    return Config1()

def main(_):


    is_train = False  # if False then test
    
    if is_train :
        train()
              
    else:
        test()
        

if __name__ == "__main__":
  tf.app.run()
  
  
  


