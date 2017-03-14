# -*- coding: utf-8 -*-
"""
Created on  三月 12 1:40 2017

@author: aeloyq
"""
import config
import time

log=""
path='./%s/log/'%config.name
name=time.asctime()+'.log'
is_save_log=False

try:
    path= config.path
except:
    pass
try:
    is_save_log= config.savelog
except:
    pass


def logger(log,level,enter=0):
    if log.find("\r\n")!=-1:
        lines=log.split('\r\n')
        et=enter
        while (et > 0):
            et-=1
            logger("", level, 0)
        for line in lines:
            logger(line, level, 0)
        et = enter
        while (et > 0):
            et -= 1
            logger("", level, 0)
        return
    prefix=""
    for i in range(level):
        prefix+="            "
    log=prefix+log
    log2save=time.asctime()+log
    while(enter>0):
        enter-=1
        log="\r\n"+log+"\r\n"
        log2save=time.asctime()+"\r\n"+log+time.asctime()+"\r\n"
    print log
    if is_save_log:
        f=open(path+name,'wb')
        f.writelines(log2save)
        f.close()