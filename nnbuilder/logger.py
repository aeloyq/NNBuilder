# -*- coding: utf-8 -*-
"""
Created on  三月 12 1:40 2017

@author: aeloyq
"""

import time
import main
import os
import copy

config = main.config

log_time = time.asctime()


def logger(log, level=0, enter=0):
    path = './%s/log/' % config.name
    name = log_time.replace(' ', '-').replace(':', '_') + '.log'
    is_save_log = config.savelog

    def parse_level(log, level):
        l = copy.copy(log)
        l = l.replace(' ', '')
        if l == "":
            return 4
        else:
            return level

    def parse_spaceline(keyword):
        if log.find(keyword) != -1:
            lines = log.split('\r\n')
            add_enter(enter)
            for line in lines:
                logger(line, level, 0)
            add_enter(enter)
            return False
        else:
            return True

    def add_prefix(log, level):
        prefix = ""
        for i in range(level):
            prefix += "            "
        log = prefix + log
        return log

    def add_enter(enter):
        et = enter
        while (et > 0):
            et -= 1
            logger('', level, 0)

    level = parse_level(log, level)
    if parse_spaceline('\r\n'):
        add_enter(enter)
        if config.is_log_detail():
            log = add_prefix(log, level)
        if config.verbose > level or config.verbose < 0:
            print(log)
        add_enter(enter)
    if is_save_log:
        f = open(path + name, 'a')
        f.write(log + "\r\n")
        f.close()
