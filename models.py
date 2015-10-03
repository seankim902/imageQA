# -*- coding: utf-8 -*-
import theano
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import theano.tensor as T
from keras import activations, initializations

import numpy as np

class RNN:
    def __init__(self, n_vocab, y_vocab, dim_word, dim):
        self.n_vocab = n_vocab  # 12047
        self.y_vocab = y_vocab  # 430
        self.dim_word = dim_word # 1024
        self.dim = dim  # 512
        
        ### Word Embedding ###        
        self.W_emb = initializations.uniform((self.n_vocab, self.dim_word))
        
        ### enc forward GRU ###
        self.W_gru = initializations.uniform((self.dim_word, self.dim * 2))
        self.U_gru = initializations.uniform((self.dim, self.dim * 2))
        self.b_gru = initializations.zero((self.dim * 2))
        self.W_gru_cdd = initializations.uniform((self.dim_word, self.dim)) # cdd : candidate
        self.U_gru_cdd = initializations.uniform((self.dim, self.dim))
        self.b_gru_cdd = initializations.zero((self.dim))       
        ### prediction ###
        self.W_pred = initializations.uniform((self.dim, self.y_vocab))
        self.b_pred = initializations.zero((self.y_vocab))


        self.params = [self.W_emb,
                       self.W_gru, self.U_gru, self.b_gru,
                       self.W_gru_cdd, self.U_gru_cdd, self.b_gru_cdd,
                       self.W_pred, self.b_pred]

    def gru_layer(self, state_below, mask=None, **kwargs):
        #state_below : step * sample * dim
        nsteps = state_below.shape[0]
        n_samples = state_below.shape[1]
        
        dim = self.dim

        def _slice(_x, n, dim):
            if _x.ndim == 3:
                print '_x.ndim : ' , _x.ndim
                return _x[:, :, n*dim:(n+1)*dim]
            return _x[:, n*dim:(n+1)*dim]
    
        # step * samples * dim
        state_below_ = T.dot(state_below, self.W_gru) + self.b_gru
        state_belowx = T.dot(state_below, self.W_gru_cdd) + self.b_gru_cdd
        
        def _step_slice(m_, x_, xx_, h_, U, Ux):
            '''
            m_ : (samples,)
            x_, h_ : samples * dimensions   
            '''
            preact = T.dot(h_, U)
            preact += x_ # samples * 1024
    
            r = T.nnet.sigmoid(_slice(preact, 0, dim))
            u = T.nnet.sigmoid(_slice(preact, 1, dim))
    
            preactx = T.dot(h_, Ux)
            preactx = preactx * r
            preactx = preactx + xx_ # samples * 512
    
            h = T.tanh(preactx)
    
            h = u * h_ + (1. - u) * h
            h = m_[:,None] * h + (1. - m_)[:,None] * h_  # m_[:,None] : samples * 1
    
            return h#, r, u, preact, preactx
        seqs = [mask, state_below_, state_belowx]
        _step = _step_slice
    
        rval, updates = theano.scan(_step, 
                                    sequences=seqs,
                                    outputs_info = [T.alloc(0., n_samples, dim)],
                                                    #None, None, None, None],
                                    non_sequences = [self.U_gru, self.U_gru_cdd],
                                    name='gru_layer', 
                                    n_steps=nsteps)
        rval = rval
        return rval

                            
    def build_model(self, lr=0.001):
    
        trng = RandomStreams(1234)
        use_noise = theano.shared(np.float32(0.))
    
        # description string: #words x #samples


        x = T.matrix('x', dtype = 'int32')
        x_mask = T.matrix('x_mask', dtype='float32')
        y = T.matrix('y', dtype = 'int32')
        
        n_timesteps = x.shape[0]
        n_samples = x.shape[1]

           
        emb = self.W_emb[x.flatten()]
        emb = emb.reshape([n_timesteps, n_samples, self.dim_word])
        # proj : gru hidden 들의 리스트   
        proj = self.gru_layer(emb, mask=x_mask)
    
    
        #print 'len(proj) : ', len(proj) # steps
        
        proj = T.mean(proj, axis=0) # hidden 들의 평균
        #proj = proj[-1]   # 마지막 hidden
        
        output = T.dot(proj, self.W_pred) + self.b_pred
        print 'output shape : ', output.shape  # samples * output
        probs = T.nnet.softmax(output)
        cost = T.nnet.categorical_crossentropy(probs, y)
        cost = T.mean(cost)
        
        print 'probs : ',probs
        print 'y : ', y
        updates = self.adam(cost=cost, params=self.params, lr=lr)
        #####updates = self.adam(cost=cost, params=self.params, lr=lr)

        return x, x_mask, y, cost, updates
        
        
    
    def sgd(self, cost, params, lr ):
        grads = T.grad(cost, params)
        updates = []
        for param, grad in zip(params, grads):
            updates.append((param, param - lr*grad))
    
        return updates
 
    def adam(self, cost, params, lr=0.0002, b1=0.1, b2=0.001, e=1e-8):
        updates = []
        grads = T.grad(cost, params)
        i = theano.shared(np.float32(0.))
        i_t = i + 1.
        fix1 = 1. - (1. - b1)**i_t
        fix2 = 1. - (1. - b2)**i_t
        lr_t = lr * (T.sqrt(fix2) / fix1)
        for p, g in zip(params, grads):
            m = theano.shared(p.get_value() * 0.)
            v = theano.shared(p.get_value() * 0.)
            m_t = (b1 * g) + ((1. - b1) * m)
            v_t = (b2 * T.sqr(g)) + ((1. - b2) * v)
            g_t = m_t / (T.sqrt(v_t) + e)
            p_t = p - (lr_t * g_t)
            updates.append((m, m_t))
            updates.append((v, v_t))
            updates.append((p, p_t))
        updates.append((i, i_t))
        return updates
        
    def train(self, train_x, train_mask_x, 
              train_y,
             # valid_x=None, valid_mask_x=None, 
              #valid_y=None, valid_mask_y=None,
             # optimizer=None,
              lr=0.001,
              batch_size=16,
              epoch=100):
        
        
        train_shared_x = theano.shared(np.asarray(train_x, dtype='int32'), borrow=True)
        train_shared_y = theano.shared(np.asarray(train_y, dtype='int32'), borrow=True)
        train_shared_mask_x = theano.shared(np.asarray(train_mask_x, dtype='float32'), borrow=True)
        
        n_train = train_shared_x.get_value(borrow=True).shape[1]
        n_train_batches = int(np.ceil(1.0 * n_train / batch_size))
        
        index = T.lscalar('index')    # index to a case
        final_index = T.lscalar('final_index')
        
        
        x, x_mask, y, cost, updates = self.build_model(lr)
        # x : step * sample * dim
        # x_mask : step * sample
        # y : sample * emb
       
        batch_start = index * batch_size
        batch_stop = T.minimum(final_index, (index + 1) * batch_size)
     
     
                    
        train_model = theano.function(inputs=[index, final_index],
                                      outputs=cost,
                                      updates=updates,
                                      givens={
                                         x: train_shared_x[:,batch_start:batch_stop],
                                         x_mask: train_shared_mask_x[:,batch_start:batch_stop],
                                         y: train_shared_y[batch_start:batch_stop]})
       
        
        i = 0
        while (i < epoch):
            i = i + 1
            print 'epoch : ', i
            for minibatch_idx in xrange(n_train_batches):
                minibatch_avg_cost = train_model(minibatch_idx, n_train)
                print 'cost : ' , minibatch_avg_cost, ' [ mini batch \'', minibatch_idx+1, '\' in epoch \'', i ,'\' ]'
       
