# -*- coding: utf-8 -*-
"""
Created on  三月 30 16:16 2017

@author: aeloyq
"""

import theano
import theano.tensor as t
import numpy as np
import timeit

idim=100
bdim=10
udim=50

x1=t.tensor3('x1')
x2=t.vector('x2')

d1=np.random.randn(idim,bdim,udim).astype('float32')
d2=np.random.randn(bdim,udim).astype('float32')

w1=theano.shared(np.random.randn(udim,udim).astype('float32'),name='w1',borrow=True)
w2=theano.shared(np.random.randn(udim,udim).astype('float32'),name='w2',borrow=True)
b=theano.shared(np.random.randn(udim).astype('float32'),name='b',borrow=True)
s_0=theano.shared(np.zeros(udim).astype('float32'),name='s_0')



sb4=t.dot(x2,w2)+b
def step4(x_,h_,s_):
    return t.dot(x_,w1)+s_+t.dot(h_,w2)


'''
o2,u2=theano.scan(step4,sequences=[x1],outputs_info=[s_0],non_sequences=[sb4],n_steps=x1.shape[0])

g2=t.grad(t.sum(o2),[w1,w2,b,s_0])
f1=theano.function([x1,x2],[o2]+g2)
def ff1():
    st=timeit.default_timer()
    for i in range(50):
        a=f1(d1,d2)
    print timeit.default_timer()-st
    '''

class RNNSLU(object):
    ''' elman neural net model '''
    def __init__(self, nh, nc, ne, de, cs):
        '''
        nh :: dimension of the hidden layer
        nc :: number of classes
        ne :: number of word embeddings in the vocabulary
        de :: dimension of the word embeddings
        cs :: word window context size
        '''
        # parameters of the model
        self.emb = theano.shared(name='embeddings',
                                 value=0.2 * np.random.uniform(-1.0, 1.0,
                                 (ne+1, de))
                                 # add one for padding at the end
                                 .astype(theano.config.floatX))
        self.wx = theano.shared(name='wx',
                                value=0.2 * np.random.uniform(-1.0, 1.0,
                                (de * cs, nh))
                                .astype(theano.config.floatX))
        self.wh = theano.shared(name='wh',
                                value=0.2 * np.random.uniform(-1.0, 1.0,
                                (nh, nh))
                                .astype(theano.config.floatX))
        self.w = theano.shared(name='w',
                               value=0.2 * np.random.uniform(-1.0, 1.0,
                               (nh, nc))
                               .astype(theano.config.floatX))
        self.bh = theano.shared(name='bh',
                                value=np.zeros(nh,
                                dtype=theano.config.floatX))
        self.b = theano.shared(name='b',
                               value=np.zeros(nc,
                               dtype=theano.config.floatX))
        self.h0 = theano.shared(name='h0',
                                value=np.zeros(nh,
                                dtype=theano.config.floatX))
        self.h0=theano.shared(value=np.zeros(nh,
                                dtype=theano.config.floatX), name='h_0', borrow=True)

        # bundle
        self.params = [self.emb, self.wx, self.wh, self.w,
                       self.bh, self.b, self.h0]
        # end-snippet-2
        # as many columns as context window size
        # as many lines as words in the sentence
        # start-snippet-3
        idxs = t.imatrix()
        x = self.emb[idxs].reshape((idxs.shape[0], de*cs))
        y_sentence = t.ivector('y_sentence')  # labels
        # end-snippet-3 start-snippet-4
        x=t.matrix('x')
        def recurrence(x_t, h_tm1):
            h_t = t.dot(x_t, self.wx)+ t.dot(h_tm1, self.wh) + self.bh
            return h_t

        def step4(x_, h_):
            return t.dot(x_, self.wx) +  t.dot(h_,self.wh)

        h, _ = theano.scan(fn=step4,
                                sequences=x,
                                outputs_info=[self.h0],
                                n_steps=x.shape[0])



        # end-snippet-6 start-snippet-7
        self.normalize = theano.function(inputs=[x],outputs=h[-1])

a=RNNSLU(200,5,6,10,7)
print 'ok'