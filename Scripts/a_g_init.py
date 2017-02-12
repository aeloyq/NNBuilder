# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 00:35:33 2017

@author: aeloyq
"""

import os
path=raw_input('Please enter the module name:\r\n')
path='./'+path
print 'Doing...'
def dfs_initfile_generator(path,deepth):
    #if os.path.isdir(path):
    namelist=[name for name in os.listdir(path) if not name.startswith(('.','_'))]
    currentfilelist=[]
    for name in namelist:
        loginfo=''
        current_path=path+'./'+name
        for i in range(deepth):
            loginfo+='- '
        print loginfo,name
        if os.path.isdir(current_path):
            currentfilelist.append(name)
            dfs_initfile_generator(current_path,deepth+1)
        else:
            currentfilelist.append(name.replace('.py',''))
    initfile=''
    if currentfilelist==[]:
        return
    for currentfile in currentfilelist:
        initfile+='import '+currentfile+'\r\n'
    initfile+='__all__=['
    for currentfile in currentfilelist:
        initfile+='\''+currentfile+'\','
    initfile=initfile[:-1]+']'
    w=open(path+'/__init__.py','wb')
    w.write(initfile)
    w.close()
dfs_initfile_generator(path,1)