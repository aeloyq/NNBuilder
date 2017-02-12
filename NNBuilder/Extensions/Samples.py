# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 15:06:02 2016

@author: aeloyq
"""

def mnist_sample(picture,result):
    print "Sample:"
    index=0
    for pic in picture:
        sample_str=''
        for i in range(28):
            for j in range(28):
                    if pic[i*28+j]==0:
                        sample_str+='□ '
                    else:
                        sample_str+='■ '
            sample_str+='\r\n'
        sample_str+="This picture is recognized as number:   %d"%(int(result[index]))
        print(sample_str)
        index+=1
def xor_sample(inputs,result):
    index=0
    for inp in inputs:
        print "Sample:"
        a=int(inp[0])
        b=int(inp[1])
        c=int(result[index])
        print a,"xor",b,"=",c
        index+=1