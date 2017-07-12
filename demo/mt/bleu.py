# -*- coding: utf-8 -*-
"""
Created on  三月 18 19:58 2017

@author: aeloyq
"""

from model import  *
import dictionary
import argparse
import progressbar
import types
import os

parser = argparse.ArgumentParser()
parser.add_argument("-beam_size",nargs='?',default=1,type=int,
                    help="set the size of window in the beam search algrithm")
parser.add_argument("-no_unk",nargs='*',
                    help="don't take unk words into account")
parser.add_argument("-cache",nargs='*',
                    help="reserve the cache into ./bleutmp")
parser.add_argument("-saves_path",nargs='?',type=types.StringType,default='./{}/save/epoch'.format(config.name),
                    help="provide where you saved a list of savings")
args = parser.parse_args()

n_size=args.beam_size
unk=True
cache=False
if args.no_unk is not None:
    unk=False
if args.cache is not None:
    cache=True
saves_path=args.saves_path

def prepare_model():
    model.build()

def set_beam_size(n):
    model.output.gen_sample(n)

def load_data(path,unk):
    if unk:
        return np.load(path)
    else:
        return exclude_unk(np.load(path))

def exclude_unk(datas):
    nd=len(datas[2])
    new_list=[]
    for idx,ds,dt in zip(range(nd),datas[2], datas[5]):
        if not(1 in ds or 1 in dt):
            new_list.append(idx)
    newx=[datas[2][i] for i in new_list]
    newy=[datas[5][i] for i in new_list]
    return [datas[0],datas[1],newx,datas[3],datas[4],newy]

def compile_theano_function():
    print 'compiling...'
    fn = theano.function(model.inputs, model.output.sample, on_unused_input='ignore',
                         updates=model.output.sample_updates)
    print 'compile ok'
    return fn

def bleutest(save):
    s_text = ''
    saveload.config.load_npz(model, save)

    print 'bleuing...'
    bar = progressbar.ProgressBar()
    for idx, index in bar(mbs):
        data = prepare_data(datas[2], datas[5], index)
        ss = fn(*data)
        for s_ in ss:
            st = dictionary.mt_t(s_, unk)
            s_text += st + '\r\n'
    print 'bleu ok'

    print 'dumping...'
    f = open('./bleutmp/{}.txt'.format(str(n_size)), 'wb')
    f.write(s_text + "\r\n")
    f.close()
    print 'dump ok'

    print'bleu testing...'
    os.system('perl ./scripts/multi-bleu.perl ./bleutmp/t.txt < ./bleutmp/{}.txt >> ./bleutmp/scoretmp.txt'.format(str(n_size)))
    f = open('bleutmp', 'rb')
    s = f.read()
    print s
    f.close()
    f = open('score.txt', 'wa')
    f.write(s.split(',')[0][7:])
    f.close()

def build_truth():
    t_text = ''
    print "building truth..."
    bar = progressbar.ProgressBar()
    for s in bar(datas[-1]):
        st = dictionary.mt_t(s)
        t_text += st + '\r\n'
    f = open('./bleutmp/t.txt', 'wb')
    f.write(t_text + "\r\n")
    f.close()
    print "build ok"

if __name__ == "__main__":
    prepare_model()
    set_beam_size(n_size)
    datas=load_data(config.data_path,unk)
    fn=compile_theano_function()

    mbs=get_minibatches_idx(datas)
    mbs=mbs[2]

    savelist = [name for name in os.listdir(saves_path) if name.endswith('.npz')]
    def cp(x,y):
        xt=os.stat('./{}/save/epoch/'.format(config.name)+x)
        yt=os.stat('./{}/save/epoch/'.format(config.name)+y)
        if xt.st_mtime>yt.st_mtime :return 1
        else:return -1
    savelist.sort(cp)
    if not os.path.exists('bleutmp'): os.mkdir('bleutmp')
    if not os.path.exists('bleutmp/t.txt'): build_truth()
    for sv in savelist:
        bleutest('epoch/' + sv.replace('.npz', ''))


