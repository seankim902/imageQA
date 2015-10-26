# -*- coding: utf-8 -*-
import theano
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import theano.tensor as T
from keras import initializations

from cnn import *

import optimizer
import cPickle
import time
import numpy as np





def save_tparams(tparams, path):
    with open(path,'wb') as f:
        for params in tparams:
            cPickle.dump(params.get_value(), f, protocol=cPickle.HIGHEST_PROTOCOL)

def load_tparams(tparams, path):
    with open(path,'rb') as f:
          for i in range(len(tparams)):
              tparams[i].set_value(cPickle.load(f))
    return tparams

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
    if (minibatch_start != n):
        minibatches.append(idx_list[minibatch_start:])
    return minibatches

def get_minibatch_cnn_features(cnn, filelist) :
    minibatch_size = len(filelist)
    cnn = cnn
    featurelist = cnn.get_features(filelist, layer='conv5_4') # sample * 512 * 14 * 14
    featurelist = featurelist.reshape(minibatch_size, 512, -1).swapaxes(1, 2) # sample * 512 * 196 -> sample * 196 * 512
    featurelist = np.array(featurelist, dtype='float32')                
    return featurelist
    
    
def dropout_layer(state_before, use_noise, trng, p):
    ratio = 1. - p
    proj = T.switch(use_noise, 
            state_before * trng.binomial(state_before.shape, p=ratio, n=1, dtype=state_before.dtype), # for training..
            state_before * ratio)
    return proj

    
class RNN_GRU:
    def __init__(self, n_vocab, y_vocab, dim_word, dim, dim_ctx):

        self.n_vocab = n_vocab  # 12047
        self.y_vocab = y_vocab  # 430
        self.dim_word = dim_word # 1024
        self.dim = dim  # 1024
        self.dim_ctx = dim_ctx  # 512
        
        ### initial context
        self.W_ctx_init = initializations.uniform((self.dim_ctx, self.dim))     
        self.b_ctx_init = initializations.zero((self.dim))

        
        ### forward : img_dim to context
        self.W_ctx_att = initializations.uniform((self.dim_ctx, self.dim_ctx)) 
        self.b_ctx_att = initializations.zero((self.dim_ctx)) 
   
        ### forward : hidden_dim to context
        self.W_dim_att = initializations.uniform((self.dim, self.dim_ctx)) 
    
        ### context energy
        self.U_att = initializations.uniform((self.dim_ctx, 1)) 
        self.c_att = initializations.zero((1)) 
   
   
        
        ### Word Embedding ###        
        self.W_emb = initializations.uniform((self.n_vocab, self.dim_word))
        
        ### enc forward GRU ###
        self.W_gru_ctx = initializations.uniform((self.dim_word, self.dim_ctx))
        self.b_gru_ctx = initializations.zero((self.dim_ctx))

        
        self.W_gru = initializations.uniform((self.dim_word, self.dim * 2))
        self.U_gru = initializations.uniform((self.dim, self.dim * 2))
        self.b_gru = initializations.zero((self.dim * 2))
        self.U_gru_ctx = initializations.uniform((self.dim_ctx, self.dim * 2))
        
        self.W_gru_cdd = initializations.uniform((self.dim_word, self.dim)) # cdd : candidate
        self.U_gru_cdd = initializations.uniform((self.dim, self.dim))
        self.b_gru_cdd = initializations.zero((self.dim)) 
        self.U_gru_cdd_ctx = initializations.uniform((self.dim_ctx, self.dim))

        ### prediction ###
        self.W_pred = initializations.uniform((self.dim, self.y_vocab))
        self.b_pred = initializations.zero((self.y_vocab))


        self.params = [self.W_ctx_init, self.b_ctx_init,
                       self.W_ctx_att, self.b_ctx_att,
                       self.W_dim_att,
                       self.U_att, self.c_att,
                       self.W_emb,
                       self.W_gru_ctx, self.b_gru_ctx,
                       self.W_gru, self.U_gru, self.b_gru, self.U_gru_ctx,
                       self.W_gru_cdd, self.U_gru_cdd, self.b_gru_cdd, self.U_gru_cdd_ctx,
                       self.W_pred, self.b_pred]

    def gru_layer(self, state_below, mask=None, context=None, init_state=None):
        #state_below : step * sample * dim
        nsteps = state_below.shape[0]
        n_samples = state_below.shape[1]
        
        dim = self.dim

        def _slice(_x, n, dim):
            if _x.ndim == 3:
                print '_x.ndim : ' , _x.ndim
                return _x[:, :, n*dim:(n+1)*dim]
            return _x[:, n*dim:(n+1)*dim]
    
        if init_state is None :
            init_state = T.alloc(0., n_samples, dim)
         
        # context forward
        proj_ctx = T.dot(context, self.W_ctx_att) + self.b_ctx_att   
        #context : sample * annotation * dim_ctx
        
        # step * samples * dim
        state_below_ = T.dot(state_below, self.W_gru) + self.b_gru
        state_belowx = T.dot(state_below, self.W_gru_cdd) + self.b_gru_cdd
        state_belowc = T.dot(state_below, self.W_gru_ctx) + self.b_gru_ctx
        
        def _step(m_, xc, x_, xx_, h_, a_, as_, proj_ctx):
            '''
            m_ : (samples,)
            x_, h_ : samples * dimensions   
            '''
            
            pstate_ = T.dot(h_, self.W_dim_att) # sample * dim_ctx
            proj_ctx_ = proj_ctx + pstate_[:, None, :]  # sample * annotation * dim_ctx
            proj_ctx_ = proj_ctx_ + xc[:, None, :]
            #추후 x to ctx 추가해볼 것. -- state_belowc            


            proj_ctx_ = T.tanh(proj_ctx_)
            
            alpha = T.dot(proj_ctx_, self.U_att)+self.c_att
            alpha = T.nnet.softmax(alpha.reshape([alpha.shape[0], alpha.shape[1]])) # softmax
            # alpha = sample * annotation(prob)
            ctx_ = (context * alpha[:,:,None]).sum(1) # sample * expected_dim_ctx
            alpha_sample = alpha # you can return something else reasonable here to debug

           
            preact = T.dot(h_, self.U_gru)
            preact += x_ # samples * 1024
            preact += T.dot(ctx_, self.U_gru_ctx)

            r = T.nnet.sigmoid(_slice(preact, 0, dim))
            u = T.nnet.sigmoid(_slice(preact, 1, dim))
    
            preactx = T.dot(h_, self.U_gru_cdd)
            preactx *= r
            preactx += xx_ # samples * 512
            preactx += T.dot(ctx_, self.U_gru_cdd_ctx)             
            
            h = T.tanh(preactx)
    
            h = u * h_ + (1. - u) * h
            h = m_[:,None] * h + (1. - m_)[:,None] * h_  # m_[:,None] : samples * 1
    
            return [h, alpha, alpha_sample, ctx_]#, r, u, preact, preactx
            
        seqs = [mask, state_belowc, state_below_, state_belowx]
        outputs_info = [init_state,
                        T.alloc(0., n_samples, proj_ctx.shape[1]),
                        T.alloc(0., n_samples, proj_ctx.shape[1]),
                        None]
        rval, updates = theano.scan(_step, 
                                    sequences=seqs,
                                    outputs_info = outputs_info,
                                    non_sequences = [proj_ctx],
                                    name='gru_layer',
                                    n_steps=nsteps)
        return rval
        
                            
    def build_model(self, lr=0.001, dropout=None):
    
        trng = RandomStreams(1234)
        use_noise = theano.shared(np.float32(0.))
    
        # description string: #words x #samples


        x = T.matrix('x', dtype = 'int32')  # step * samples
        x_mask = T.matrix('x_mask', dtype='float32')  # step * samples
        y = T.matrix('y', dtype = 'int32')  # sample * emb
        ctx = T.tensor3('ctx', dtype = 'float32')  # sample * annotation * dim
        
        n_timesteps = x.shape[0]
        n_samples = x.shape[1]


        emb = self.W_emb[x.flatten()]
        
        emb = emb.reshape([n_timesteps, n_samples, self.dim_word])
        
        ctx0 = ctx
        ctx_mean = ctx0.mean(1)
        
        init_state = T.dot(ctx_mean, self.W_ctx_init) + self.b_ctx_init
                
             
        # proj : gru hidden 들의 리스트   
        proj = self.gru_layer(emb, mask=x_mask, context=ctx, init_state=init_state)
        proj_h = proj[0]
        
        # hidden 들의 평균
        proj_h = (proj_h * x_mask[:, :, None]).sum(axis=0)
        proj_h = proj_h / x_mask.sum(axis=0)[:, None]  # sample * dim
        
        # 마지막 hidden
        #proj = proj[-1]  # sample * dim
        
        if dropout is not None :
            proj_h = dropout_layer(proj_h, use_noise, trng, dropout)


        
        output = T.dot(proj_h, self.W_pred) + self.b_pred
        
        probs = T.nnet.softmax(output)
        prediction = probs.argmax(axis=1)
        
        ## avoid NaN
        epsilon = 1.0e-9
        probs = T.clip(probs, epsilon, 1.0 - epsilon)
        probs /= probs.sum(axis=-1, keepdims=True)
        ## avoid NaN
    
    
        cost = T.nnet.categorical_crossentropy(probs, y)
        cost = T.mean(cost)
        
        updates = optimizer.adam(cost=cost, params=self.params, lr=lr)

        return trng, use_noise, x, x_mask, ctx, y, cost, updates, prediction
        
        
        
    def train(self, train_x, train_mask_x, train_imgs, train_y,
              valid_x=None, valid_mask_x=None, valid_imgs=None, valid_y=None,
              valid=None,
              lr=0.001,
              dropout=None,
              batch_size=16,
              epoch=100,
              save=None):

        cnn = CNN(deploy='/home/seonhoon/Desktop/caffemodel/vgg19/VGG_ILSVRC_19_layers_deploy.prototxt', model='/home/seonhoon/Desktop/caffemodel/vgg19/VGG_ILSVRC_19_layers.caffemodel')

        n_train = train_x.shape[1]

        trng, use_noise, x, x_mask, ctx, y, cost, updates, prediction = self.build_model(lr, dropout)
        # x : step * sample * dim
        # x_mask : step * sample
        # ctx : sample * annotation * dim_ctx
        # y : sample * emb
        

        train_model = theano.function(inputs=[x, x_mask, ctx, y],
                                      outputs=cost,
                                      updates=updates)
        
        if valid is not None:
            valid_model = theano.function(inputs=[x, x_mask, ctx],
                                          outputs=prediction)   
            valid_batch_indices = get_minibatch_indices(valid_x.shape[1], batch_size)
            
            
        best_cost = np.inf
        best_cost_epoch = 0
        best_params = self.params
        for i in xrange(epoch):
            start_time = time.time()
            use_noise.set_value(1.)
            batch_indices=get_minibatch_indices(n_train, batch_size, shuffle=True)
            

            
            for j, indices in enumerate(batch_indices):
                minibatch_avg_cost = 0 
                
                x = [ train_x[:,t] for t in indices]
                x = np.transpose(x)
                x_mask = [ train_mask_x[:,t] for t in indices]
                x_mask = np.transpose(x_mask)
                y = [ train_y[t,:] for t in indices]
                y = np.array(y)
                batch_imgs = [ train_imgs[t] for t in indices] 
                ctx = get_minibatch_cnn_features(cnn, batch_imgs)
                
                minibatch_avg_cost = train_model(x, x_mask, ctx, y)
                print '        minibatch cost : ' , minibatch_avg_cost, ' , minibatch :', (j+1), ' in epoch \'', (i+1) 

                if minibatch_avg_cost < best_cost :
                    best_cost = minibatch_avg_cost
                    best_cost_epoch = i+1
                    best_params = self.params
            end_time = time.time()       
            print 'cost : ' , minibatch_avg_cost, ' in epoch \'', (i+1) ,'\' ', '[ ', int(end_time-start_time), 'sec ]   [ best_cost : ', best_cost, ' in epoch \'', best_cost_epoch, '\' ]'



            # validation  
            if valid is not None:
                if (i+1) % valid == 0:
                    use_noise.set_value(0.)

                    valid_prediction = []
                    for k, valid_indices in enumerate(valid_batch_indices):
                        
                        val_x = [ valid_x[:,t] for t in valid_indices]
                        val_x = np.transpose(val_x)
                        val_x_mask = [ valid_mask_x[:,t] for t in valid_indices]
                        val_x_mask = np.transpose(val_x_mask)
                        val_batch_imgs = [ valid_imgs[t] for t in valid_indices]
                        val_ctx = get_minibatch_cnn_features(cnn, val_batch_imgs)
                        
                        valid_prediction += valid_model(val_x, val_x_mask, val_ctx).tolist()
                    correct = 0 
                    for l in range(len(valid_prediction)):
                        if valid_prediction[l]==valid_y[l] : 
                            correct += 1
                    print '## valid accuracy : ', float(correct) / len(valid_prediction)
                    

            # save model       
            if save is not None:
                if (i+1) % save == 0:
                    print 'save best param..',
                    save_tparams(best_params, 'model.pkl')
                    print 'Done'
            
         
    def prediction(self, test_x, test_mask_x, test_img, test_y,
             # valid_x=None, valid_mask_x=None, 
              #valid_y=None, valid_mask_y=None,
             # optimizer=None,
              lr=0.001,
              batch_size=16,
              epoch=100,
              save=None):
        
        load_tparams(self.params, 'model.pkl')
        test_shared_x = theano.shared(np.asarray(test_x, dtype='int32'), borrow=True)
        #test_shared_y = theano.shared(np.asarray(test_y, dtype='int32'), borrow=True)
        test_shared_mask_x = theano.shared(np.asarray(test_mask_x, dtype='float32'), borrow=True)
        test_shared_img = theano.shared(np.asarray(test_img, dtype='float32'), borrow=True)

        n_test = test_x.shape[1]
        n_test_batches = int(np.ceil(1.0 * n_test / batch_size))
        
        index = T.lscalar('index')    # index to a case
        final_index = T.lscalar('final_index')
        
        
        trng, use_noise, x, x_mask, img, y, _, _, prediction = self.build_model(lr)
       
        batch_start = index * batch_size
        batch_stop = T.minimum(final_index, (index + 1) * batch_size)     

        test_model = theano.function(inputs=[index, final_index],
                                      outputs=prediction,
                                      givens={
                                         x: test_shared_x[:,batch_start:batch_stop],
                                         x_mask: test_shared_mask_x[:,batch_start:batch_stop],
                                         img: test_shared_img[batch_start:batch_stop,:]})
        
        
        prediction = []
        for minibatch_idx in xrange(n_test_batches):
            print minibatch_idx, ' / ' , n_test_batches
            prediction += test_model(minibatch_idx, n_test).tolist()
        return prediction 
        




class RNN_LSTM:
    def __init__(self, n_vocab, y_vocab, dim_word, dim, dim_ctx):

        self.n_vocab = n_vocab  # 12047
        self.y_vocab = y_vocab  # 430
        self.dim_word = dim_word # 1024
        self.dim = dim  # 1024
        self.dim_ctx = dim_ctx  # 512
        
        ### initial context
        self.W_hidden_init = initializations.uniform((self.dim_ctx, self.dim))     
        self.b_hidden_init = initializations.zero((self.dim))
        self.W_memory_init = initializations.uniform((self.dim_ctx, self.dim))     
        self.b_memory_init = initializations.zero((self.dim))

        
        ### forward : img_dim to context
        self.W_ctx_att = initializations.uniform((self.dim_ctx, self.dim_ctx)) 
        self.b_ctx_att = initializations.zero((self.dim_ctx)) 
   
        ### forward : hidden_dim to context
        self.W_dim_att = initializations.uniform((self.dim, self.dim_ctx)) 
    
        ### context energy
        self.U_att = initializations.uniform((self.dim_ctx, 1)) 
        self.c_att = initializations.zero((1)) 
   
   
        
        ### Word Embedding ###        
        self.W_emb = initializations.uniform((self.n_vocab, self.dim_word))
        
        ### enc forward GRU ###
        self.W_lstm_ctx = initializations.uniform((self.dim_word, self.dim_ctx))
        self.b_lstm_ctx = initializations.zero((self.dim_ctx))

        
        self.W_lstm = initializations.uniform((self.dim_word, self.dim * 4))
        self.U_lstm = initializations.uniform((self.dim, self.dim * 4))
        self.b_lstm = initializations.zero((self.dim * 4))
        self.U_lstm_ctx = initializations.uniform((self.dim_ctx, self.dim * 4))
        


        ### prediction ###
        self.W_pred = initializations.uniform((self.dim, self.y_vocab))
        self.b_pred = initializations.zero((self.y_vocab))


        self.params = [self.W_hidden_init, self.b_hidden_init,self.W_memory_init, self.b_memory_init,
                       self.W_ctx_att, self.b_ctx_att,
                       self.W_dim_att,
                       self.U_att, self.c_att,
                       self.W_emb,
                      # self.W_lstm_ctx, self.b_lstm_ctx,
                       self.W_lstm, self.U_lstm, self.b_lstm, self.U_lstm_ctx,
                       self.W_pred, self.b_pred]

    def lstm_layer(self, state_below, mask=None, context=None, init_state=None, init_memory=None):
        #state_below : step * sample * dim
        nsteps = state_below.shape[0]
        n_samples = state_below.shape[1]
        
        dim = self.dim

        def _slice(_x, n, dim):
            if _x.ndim == 3:
                print '_x.ndim : ' , _x.ndim
                return _x[:, :, n*dim:(n+1)*dim]
            return _x[:, n*dim:(n+1)*dim]
    
        if init_state is None :
            init_state = T.alloc(0., n_samples, dim)
        if init_memory is None:
            init_memory = T.alloc(0., n_samples, dim)         
        # context forward
        proj_ctx = T.dot(context, self.W_ctx_att) + self.b_ctx_att   
        #context : sample * annotation * dim_ctx
        
        # step * samples * dim
        state_below_ = T.dot(state_below, self.W_lstm) + self.b_lstm
        #state_belowc = T.dot(state_below, self.W_lstm_ctx) + self.b_lstm_ctx
        
        def _step(m_, x_, h_, c_, a_, as_, proj_ctx):
            '''
            m_ : (samples,)
            x_, h_ : samples * dimensions   
            '''
            
            pstate_ = T.dot(h_, self.W_dim_att) # sample * dim_ctx
            proj_ctx_ = proj_ctx + pstate_[:, None, :]  # sample * annotation * dim_ctx
            #proj_ctx_ = proj_ctx_ + xc[:, None, :]
            #추후 x to ctx 추가해볼 것. -- state_belowc            


            proj_ctx_ = T.tanh(proj_ctx_)
            
            alpha = T.dot(proj_ctx_, self.U_att)+self.c_att
            alpha = T.nnet.softmax(alpha.reshape([alpha.shape[0], alpha.shape[1]])) # softmax
            # alpha = sample * annotation(prob)
            ctx_ = (context * alpha[:,:,None]).sum(1) # sample * expected_dim_ctx
            alpha_sample = alpha # you can return something else reasonable here to debug

           
            preact = T.dot(h_, self.U_lstm)
            preact += x_ # samples * 1024
            preact += T.dot(ctx_, self.U_lstm_ctx)
    
            i = _slice(preact, 0, dim)
            f = _slice(preact, 1, dim)
            o = _slice(preact, 2, dim)
    
            i = T.nnet.sigmoid(i)
            f = T.nnet.sigmoid(f)
            o = T.nnet.sigmoid(o)
            c = T.tanh(_slice(preact, 3, dim))
    
            # compute the new memory/hidden state
            # if the mask is 0, just copy the previous state
            c = f * c_ + i * c
            c = m_[:,None] * c + (1. - m_)[:,None] * c_ 
    
            h = o * T.tanh(c)
            h = m_[:,None] * h + (1. - m_)[:,None] * h_
  
    
            return [h, c, alpha, alpha_sample, ctx_]#, r, u, preact, preactx
            
        seqs = [mask, state_below_]
        outputs_info = [init_state,
                        init_memory,
                        T.alloc(0., n_samples, proj_ctx.shape[1]),
                        T.alloc(0., n_samples, proj_ctx.shape[1]),
                        None]
        rval, updates = theano.scan(_step, 
                                    sequences=seqs,
                                    outputs_info = outputs_info,
                                    non_sequences = [proj_ctx],
                                    name='lstm_layer',
                                    n_steps=nsteps)
        return rval
        
                            
    def build_model(self, lr=0.001, dropout=None):
    
        trng = RandomStreams(1234)
        use_noise = theano.shared(np.float32(0.))
    
        # description string: #words x #samples


        x = T.matrix('x', dtype = 'int32')  # step * samples
        x_mask = T.matrix('x_mask', dtype='float32')  # step * samples
        y = T.matrix('y', dtype = 'int32')  # sample * emb
        ctx = T.tensor3('ctx', dtype = 'float32')  # sample * annotation * dim
        
        n_timesteps = x.shape[0]
        n_samples = x.shape[1]


        emb = self.W_emb[x.flatten()]
        
        emb = emb.reshape([n_timesteps, n_samples, self.dim_word])
        
        ctx0 = ctx
        ctx_mean = ctx0.mean(1)
        
        init_state = T.dot(ctx_mean, self.W_hidden_init) + self.b_hidden_init
        init_memory = T.dot(ctx_mean, self.W_memory_init) + self.b_memory_init
           
             
        # proj : lstm hidden 들의 리스트   
        proj = self.lstm_layer(emb, mask=x_mask, context=ctx, init_state=init_state, init_memory=init_memory)
        proj_h = proj[0]
        
        # hidden 들의 평균
        proj_h = (proj_h * x_mask[:, :, None]).sum(axis=0)
        proj_h = proj_h / x_mask.sum(axis=0)[:, None]  # sample * dim
        
        # 마지막 hidden
        #proj_h = proj_h[-1]  # sample * dim
        

        if dropout is not None :
            proj_h = dropout_layer(proj_h, use_noise, trng, dropout)

        
        output = T.dot(proj_h, self.W_pred) + self.b_pred
        
        probs = T.nnet.softmax(output)
        prediction = probs.argmax(axis=1)
        
        ## avoid NaN
        epsilon = 1.0e-9
        probs = T.clip(probs, epsilon, 1.0 - epsilon)
        probs /= probs.sum(axis=-1, keepdims=True)
        ## avoid NaN
    
    
        cost = T.nnet.categorical_crossentropy(probs, y)
        cost = T.mean(cost)
        
        updates = optimizer.adam(cost=cost, params=self.params, lr=lr)

        return trng, use_noise, x, x_mask, ctx, y, cost, updates, prediction
        
        
        
    def train(self, train_x, train_mask_x, train_imgs, train_y,
              valid_x=None, valid_mask_x=None, valid_imgs=None, valid_y=None,
              valid=None,
              lr=0.001,
              dropout=None,
              batch_size=16,
              epoch=100,
              save=None):

        cnn = CNN(deploy='/home/seonhoon/Desktop/caffemodel/vgg19/VGG_ILSVRC_19_layers_deploy.prototxt', model='/home/seonhoon/Desktop/caffemodel/vgg19/VGG_ILSVRC_19_layers.caffemodel')

        n_train = train_x.shape[1]

        trng, use_noise, x, x_mask, ctx, y, cost, updates, prediction = self.build_model(lr, dropout)
        # x : step * sample * dim
        # x_mask : step * sample
        # ctx : sample * annotation * dim_ctx
        # y : sample * emb
        

        train_model = theano.function(inputs=[x, x_mask, ctx, y],
                                      outputs=cost,
                                      updates=updates)
        
        if valid is not None:
            valid_model = theano.function(inputs=[x, x_mask, ctx],
                                          outputs=prediction)   
            valid_batch_indices = get_minibatch_indices(valid_x.shape[1], batch_size)
            
            
        best_cost = np.inf
        best_cost_epoch = 0
        best_params = self.params
        for i in xrange(epoch):
            start_time = time.time()
            use_noise.set_value(1.)
            batch_indices=get_minibatch_indices(n_train, batch_size, shuffle=True)



            
            
            for j, indices in enumerate(batch_indices):
                minibatch_avg_cost = 0 
                
                x = [ train_x[:,t] for t in indices]
                x = np.transpose(x)
                x_mask = [ train_mask_x[:,t] for t in indices]
                x_mask = np.transpose(x_mask)
                y = [ train_y[t,:] for t in indices]
                y = np.array(y)
                batch_imgs = [ train_imgs[t] for t in indices] 
                ctx = get_minibatch_cnn_features(cnn, batch_imgs)
                
                minibatch_avg_cost = train_model(x, x_mask, ctx, y)
                if j % 15 == 0 :
                    print '        minibatch cost : ' , minibatch_avg_cost, ' , minibatch :', (j+1), ' in epoch \'', (i+1) 

                if minibatch_avg_cost < best_cost :
                    best_cost = minibatch_avg_cost
                    best_cost_epoch = i+1
                    best_params = self.params
            end_time = time.time()       
            print 'cost : ' , minibatch_avg_cost, ' in epoch \'', (i+1) ,'\' ', '[ ', int(end_time-start_time), 'sec ]   [ best_cost : ', best_cost, ' in epoch \'', best_cost_epoch, '\' ]'
            
           
            # validation  
            if valid is not None:
                if (i+1) % valid == 0:
                    use_noise.set_value(0.)

                    valid_prediction = []
                    for k, valid_indices in enumerate(valid_batch_indices):
                        
                        val_x = [ valid_x[:,t] for t in valid_indices]
                        val_x = np.transpose(val_x)
                        val_x_mask = [ valid_mask_x[:,t] for t in valid_indices]
                        val_x_mask = np.transpose(val_x_mask)
                        val_batch_imgs = [ valid_imgs[t] for t in valid_indices]
                        val_ctx = get_minibatch_cnn_features(cnn, val_batch_imgs)
                        
                        valid_prediction += valid_model(val_x, val_x_mask, val_ctx).tolist()
                    correct = 0 
                    for l in range(len(valid_prediction)):
                        if valid_prediction[l]==valid_y[l] : 
                            correct += 1
                    print '## valid accuracy : ', float(correct) / len(valid_prediction)
            

            # save model       
            if save is not None:
                if (i+1) % save == 0:
                    print 'save best param..',
                    save_tparams(best_params, 'model.pkl')
                    print 'Done'
            
         
 



    
class BIRNN_GRU:
    def __init__(self, n_vocab, y_vocab, dim_word, dim, dim_ctx):

        self.n_vocab = n_vocab  # 12047
        self.y_vocab = y_vocab  # 430
        self.dim_word = dim_word # 1024
        self.dim = dim  # 1024
        self.dim_ctx = dim_ctx  # 512
        
        ### initial context
        self.W_ctx_init = initializations.uniform((self.dim_ctx, self.dim))     
        self.b_ctx_init = initializations.zero((self.dim))

        
        ### forward : img_dim to context
        self.W_ctx_att = initializations.uniform((self.dim_ctx, self.dim_ctx)) 
        self.b_ctx_att = initializations.zero((self.dim_ctx)) 
   
        ### forward : hidden_dim to context
        self.W_dim_att = initializations.uniform((self.dim, self.dim_ctx)) 
    
        ### context energy
        self.U_att = initializations.uniform((self.dim_ctx, 1)) 
        self.c_att = initializations.zero((1)) 
   
   
        
        ### Word Embedding ###        
        self.W_emb = initializations.uniform((self.n_vocab, self.dim_word))
        
        ### enc forward GRU ###
        self.W_gru_ctx = initializations.uniform((self.dim_word, self.dim_ctx))
        self.b_gru_ctx = initializations.zero((self.dim_ctx))

        
        self.W_gru = initializations.uniform((self.dim_word, self.dim * 2))
        self.U_gru = initializations.uniform((self.dim, self.dim * 2))
        self.b_gru = initializations.zero((self.dim * 2))
        self.U_gru_ctx = initializations.uniform((self.dim_ctx, self.dim * 2))
        
        self.W_gru_cdd = initializations.uniform((self.dim_word, self.dim)) # cdd : candidate
        self.U_gru_cdd = initializations.uniform((self.dim, self.dim))
        self.b_gru_cdd = initializations.zero((self.dim)) 
        self.U_gru_cdd_ctx = initializations.uniform((self.dim_ctx, self.dim))

        ### prediction ###
        self.W_pred = initializations.uniform((self.dim * 2, self.y_vocab))
        self.b_pred = initializations.zero((self.y_vocab))


        self.params = [self.W_ctx_init, self.b_ctx_init,
                       self.W_ctx_att, self.b_ctx_att,
                       self.W_dim_att,
                       self.U_att, self.c_att,
                       self.W_emb,
                       self.W_gru_ctx, self.b_gru_ctx,
                       self.W_gru, self.U_gru, self.b_gru, self.U_gru_ctx,
                       self.W_gru_cdd, self.U_gru_cdd, self.b_gru_cdd, self.U_gru_cdd_ctx,
                       self.W_pred, self.b_pred]

    def gru_layer(self, state_below, mask=None, context=None, init_state=None):
        #state_below : step * sample * dim
        nsteps = state_below.shape[0]
        n_samples = state_below.shape[1]
        
        dim = self.dim

        def _slice(_x, n, dim):
            if _x.ndim == 3:
                print '_x.ndim : ' , _x.ndim
                return _x[:, :, n*dim:(n+1)*dim]
            return _x[:, n*dim:(n+1)*dim]
    
        if init_state is None :
            init_state = T.alloc(0., n_samples, dim)
         
        # context forward
        proj_ctx = T.dot(context, self.W_ctx_att) + self.b_ctx_att   
        #context : sample * annotation * dim_ctx
        
        # step * samples * dim
        state_below_ = T.dot(state_below, self.W_gru) + self.b_gru
        state_belowx = T.dot(state_below, self.W_gru_cdd) + self.b_gru_cdd
        state_belowc = T.dot(state_below, self.W_gru_ctx) + self.b_gru_ctx
        
        def _step(m_, xc, x_, xx_, h_, a_, as_, proj_ctx):
            '''
            m_ : (samples,)
            x_, h_ : samples * dimensions   
            '''
            
            pstate_ = T.dot(h_, self.W_dim_att) # sample * dim_ctx
            proj_ctx_ = proj_ctx + pstate_[:, None, :]  # sample * annotation * dim_ctx
            proj_ctx_ = proj_ctx_ + xc[:, None, :]
            #추후 x to ctx 추가해볼 것. -- state_belowc            


            proj_ctx_ = T.tanh(proj_ctx_)
            
            alpha = T.dot(proj_ctx_, self.U_att)+self.c_att
            alpha = T.nnet.softmax(alpha.reshape([alpha.shape[0], alpha.shape[1]])) # softmax
            # alpha = sample * annotation(prob)
            ctx_ = (context * alpha[:,:,None]).sum(1) # sample * expected_dim_ctx
            alpha_sample = alpha # you can return something else reasonable here to debug

           
            preact = T.dot(h_, self.U_gru)
            preact += x_ # samples * 1024
            preact += T.dot(ctx_, self.U_gru_ctx)

            r = T.nnet.sigmoid(_slice(preact, 0, dim))
            u = T.nnet.sigmoid(_slice(preact, 1, dim))
    
            preactx = T.dot(h_, self.U_gru_cdd)
            preactx *= r
            preactx += xx_ # samples * 512
            preactx += T.dot(ctx_, self.U_gru_cdd_ctx)             
            
            h = T.tanh(preactx)
    
            h = u * h_ + (1. - u) * h
            h = m_[:,None] * h + (1. - m_)[:,None] * h_  # m_[:,None] : samples * 1
    
            return [h, alpha, alpha_sample, ctx_]#, r, u, preact, preactx
            
        seqs = [mask, state_belowc, state_below_, state_belowx]
        outputs_info = [init_state,
                        T.alloc(0., n_samples, proj_ctx.shape[1]),
                        T.alloc(0., n_samples, proj_ctx.shape[1]),
                        None]
        rval, updates = theano.scan(_step, 
                                    sequences=seqs,
                                    outputs_info = outputs_info,
                                    non_sequences = [proj_ctx],
                                    name='gru_layer',
                                    n_steps=nsteps)
        return rval
        
                            
    def build_model(self, lr=0.001, dropout=None):
    
        trng = RandomStreams(1234)
        use_noise = theano.shared(np.float32(0.))
    
        # description string: #words x #samples


        x = T.matrix('x', dtype = 'int32')  # step * samples
        x_mask = T.matrix('x_mask', dtype='float32')  # step * samples
        y = T.matrix('y', dtype = 'int32')  # sample * emb
        ctx = T.tensor3('ctx', dtype = 'float32')  # sample * annotation * dim
        
        n_timesteps = x.shape[0]
        n_samples = x.shape[1]

        xr = x[::-1]
        xr_mask = x_mask[::-1]
        
        emb = self.W_emb[x.flatten()]
        emb = emb.reshape([n_timesteps, n_samples, self.dim_word])

        embr = self.W_emb[xr.flatten()]
        embr = embr.reshape([n_timesteps, n_samples, self.dim_word])
        
        ctx0 = ctx
        ctx_mean = ctx0.mean(1)
        
        init_state = T.dot(ctx_mean, self.W_ctx_init) + self.b_ctx_init
                
             
        # proj : gru hidden 들의 리스트   
        proj = self.gru_layer(emb, mask=x_mask, context=ctx, init_state=init_state)
        proj_h = proj[0]

        projr = self.gru_layer(embr, mask=xr_mask, context=ctx, init_state=init_state)
        projr_h = projr[0]


        concat_proj_h = concatenate([proj_h, projr_h[::-1]], axis=proj_h.ndim-1)
        # step_ctx : step * samples * (dim*2)
        concat_proj_h = (concat_proj_h * x_mask[:,:,None]).sum(0) / x_mask.sum(0)[:,None]
        # step_ctx_mean : samples * (dim*2)


        if dropout is not None :
            concat_proj_h = dropout_layer(concat_proj_h, use_noise, trng, dropout)


        
        output = T.dot(concat_proj_h, self.W_pred) + self.b_pred
        
        probs = T.nnet.softmax(output)
        prediction = probs.argmax(axis=1)
        
        ## avoid NaN
        epsilon = 1.0e-9
        probs = T.clip(probs, epsilon, 1.0 - epsilon)
        probs /= probs.sum(axis=-1, keepdims=True)
        ## avoid NaN
    
    
        cost = T.nnet.categorical_crossentropy(probs, y)
        cost = T.mean(cost)
        
        updates = optimizer.adam(cost=cost, params=self.params, lr=lr)

        return trng, use_noise, x, x_mask, ctx, y, cost, updates, prediction
        
        
        
    def train(self, train_x, train_mask_x, train_imgs, train_y,
              valid_x=None, valid_mask_x=None, valid_imgs=None, valid_y=None,
              valid=None,
              lr=0.001,
              dropout=None,
              batch_size=16,
              epoch=100,
              save=None):

        cnn = CNN(deploy='/home/seonhoon/Desktop/caffemodel/vgg19/VGG_ILSVRC_19_layers_deploy.prototxt', model='/home/seonhoon/Desktop/caffemodel/vgg19/VGG_ILSVRC_19_layers.caffemodel')

        n_train = train_x.shape[1]

        trng, use_noise, x, x_mask, ctx, y, cost, updates, prediction = self.build_model(lr, dropout)
        # x : step * sample * dim
        # x_mask : step * sample
        # ctx : sample * annotation * dim_ctx
        # y : sample * emb
        

        train_model = theano.function(inputs=[x, x_mask, ctx, y],
                                      outputs=cost,
                                      updates=updates)
        
        if valid is not None:
            valid_model = theano.function(inputs=[x, x_mask, ctx],
                                          outputs=prediction)   
            valid_batch_indices = get_minibatch_indices(valid_x.shape[1], batch_size)
            
            
        best_cost = np.inf
        best_cost_epoch = 0
        best_params = self.params
        for i in xrange(epoch):
            start_time = time.time()
            use_noise.set_value(1.)
            batch_indices=get_minibatch_indices(n_train, batch_size, shuffle=True)
            

            
            for j, indices in enumerate(batch_indices):
                minibatch_avg_cost = 0 
                
                x = [ train_x[:,t] for t in indices]
                x = np.transpose(x)
                x_mask = [ train_mask_x[:,t] for t in indices]
                x_mask = np.transpose(x_mask)
                y = [ train_y[t,:] for t in indices]
                y = np.array(y)
                batch_imgs = [ train_imgs[t] for t in indices] 
                ctx = get_minibatch_cnn_features(cnn, batch_imgs)
                
                minibatch_avg_cost = train_model(x, x_mask, ctx, y)
                print '        minibatch cost : ' , minibatch_avg_cost, ' , minibatch :', (j+1), ' in epoch \'', (i+1) 

                if minibatch_avg_cost < best_cost :
                    best_cost = minibatch_avg_cost
                    best_cost_epoch = i+1
                    best_params = self.params
            end_time = time.time()       
            print 'cost : ' , minibatch_avg_cost, ' in epoch \'', (i+1) ,'\' ', '[ ', int(end_time-start_time), 'sec ]   [ best_cost : ', best_cost, ' in epoch \'', best_cost_epoch, '\' ]'



            # validation  
            if valid is not None:
                if (i+1) % valid == 0:
                    use_noise.set_value(0.)

                    valid_prediction = []
                    for k, valid_indices in enumerate(valid_batch_indices):
                        
                        val_x = [ valid_x[:,t] for t in valid_indices]
                        val_x = np.transpose(val_x)
                        val_x_mask = [ valid_mask_x[:,t] for t in valid_indices]
                        val_x_mask = np.transpose(val_x_mask)
                        val_batch_imgs = [ valid_imgs[t] for t in valid_indices]
                        val_ctx = get_minibatch_cnn_features(cnn, val_batch_imgs)
                        
                        valid_prediction += valid_model(val_x, val_x_mask, val_ctx).tolist()
                    correct = 0 
                    for l in range(len(valid_prediction)):
                        if valid_prediction[l]==valid_y[l] : 
                            correct += 1
                    print '## valid accuracy : ', float(correct) / len(valid_prediction)
                    

            # save model       
            if save is not None:
                if (i+1) % save == 0:
                    print 'save best param..',
                    save_tparams(best_params, 'model.pkl')
                    print 'Done'
            
         
 
        
   

            
            
            
            

                    



        