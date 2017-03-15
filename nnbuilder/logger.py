# -*- coding: utf-8 -*-
"""
Created on  三月 12 1:40 2017

@author: aeloyq
"""
import config
import time

import os



log_time=time.asctime()
def logger(log,level,enter=0):

    if not os.path.exists('./%s' % config.name):
        os.mkdir('./%s' % config.name)
    if not os.path.exists('./%s/log' % config.name):
        os.mkdir('./%s/log' % config.name)
    if not os.path.exists('./%s/save' % config.name):
        os.mkdir('./%s/save' % config.name)
    path = './%s/log/' % config.name
    name = log_time.replace(' ', '-').replace(':','_') + '.log'
    is_save_log = False
    try:
        path = config.path
    except:
        pass
    try:
        is_save_log = config.savelog
    except:
        pass

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
    if log.find("\r")!=-1:
        lines=log.split('\r')
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
    if log.find("\n")!=-1:
        lines=log.split('\n')
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
    while(enter>0):
        enter-=1
        log="\r\n"+log+"\r\n"
    print log
    if is_save_log:
        f=open(path+name,'a')
        f.write(log+"\r\n")
        f.close()