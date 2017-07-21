# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 15:06:02 2016

@author: aeloyq
"""

def mnist_sample(picture,result,y):
    sample_str = ""
    picture=picture[0]
    for i in range(28):
        for j in range(28):
                if picture[i*28+j]==0:
                    sample_str+='  '
                else:
                    sample_str+='* '
        sample_str+='\r\n'
    sample_str+="Recognized:%d      Truth:%d"%(int(result[0]),int(y[0]))
    return sample_str

def xor_sample(inputs,result,y):
    sample_str =""
    a=int(inputs[0])
    b=int(inputs[1])
    c=int(result)
    sample_str+="%d xor %d = %d (%d)"%(a,b,c,int(y[0]))
    return sample_str

def add_sample(inputs,result,y):
    sample_str = "Sample:\r\n"
    a=b=c=y_new=''
    for i in inputs[0][0]:
        a+='%d'%i
    for i in inputs[1][0]:
        b+='%d'%i
    for i in result[0]:
        c+='%d'%i
    for i in y[0]:
        y_new+='%d'%i
    a=int(a,2)
    b=int(b,2)
    c=int(c,2)
    y_new=int(y_new,2)
    sample_str+="%d + %d = %d (%d)"%(a,b,c,y_new)
    return sample_str


