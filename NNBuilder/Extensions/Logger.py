# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 13:45:07 2016

@author: aeloyq
"""

import subprocess
import Extension
import time

base=Extension.extension
class ex(base):
    def __init__(self,**kwargs):
        base.__init__(self,**kwargs)
    def before_train(self,**kwargs):
        self.logger()
    def after_epoch(self,**kwargs):
        self.logger()
    def after_train(self,**kwargs):
        self.logger()
    def logger(self):
        r = subprocess.Popen(['ls', '-l'], stdout=subprocess.PIPE).communicate()[0]
        f = open('f:/%s-%s-%s_%s:%s:%s-testlog.log' %
                 (time.localtime().tm_year, time.localtime().tm_mon,
                  time.localtime().tm_mday, time.localtime().tm_hour,
                  time.localtime().tm_min, time.localtime().tm_sec), 'wb')
        f.write(r)
        f.close()