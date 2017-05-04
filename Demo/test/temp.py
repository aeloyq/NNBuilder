import theano
import theano.tensor as t
import numpy as np
n=50

rnd1=lambda :np.random.uniform(-1,1,[n]).astype('float32')
rnd2=lambda :np.random.uniform(-1,1,[n,n]).astype('float32')

x1=theano.shared(value=rnd2())
w1=theano.shared(value=rnd2())
w2=theano.shared(value=rnd2())
x2=theano.shared(value=rnd2())
w3=theano.shared(value=rnd2())
w4=theano.shared(value=rnd2())
def step1(x1,w1,w2):
    h=x1 * w1  * w2
    return h
def step2(x2,o1,w3,w4):
    h=x2 * w3 + o1 * w4
    return h
def step(x1,x2,w1,w2,w3,w4):
    h=x1 * w1 * w2
    h_=x2 * w3 +h*w4
    return h_
o1,u1=theano.scan(step1,[x1],[],[w1,w2],n_steps=n)
o2,u2=theano.scan(step2, [x2,o1],[],[w3,w4],n_steps=n)
o,u=theano.scan(step, [x1,x2],[None],[w1,w2,w3,w4],n_steps=n)
g_=t.grad(o2.sum(),[w1,w2,w3,w4])
g=t.grad(o[1].sum(),[w1,w2,w3,w4])
fn_=theano.function([],g_)
fn=theano.function([],g)
fn()
fn_()


import timeit
timeit.timeit('fn()', 'from __main__ import fn')
timeit.timeit('fn_()', 'from __main__ import fn_')