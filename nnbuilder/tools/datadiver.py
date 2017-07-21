# -*- coding: utf-8 -*-
"""
Created on  七月 17 6:45 2017

@author: aeloyq
"""
import numpy as np
import progressbar
import os


def ExecuteDiv(DataPath, StdinFunction, NumOfDivParts, OutPath=None, SaveFunction=np.save, BatchSize=1, TrueList=False):
    print 'Divide Data Script V1.0:'
    print 'Loading ...'
    data = StdinFunction(DataPath)
    print 'Dividing ...'
    if OutPath is None:
        OutPath = './data/DivData'
    if not os.path.exists(OutPath): os.mkdir(OutPath)
    if not TrueList:
        vttmp = [data[1], data[2], data[4], data[5]]
    else:
        vttmp = [data[1], data[2], [data[4]], [data[5]]]
    SaveFunction(OutPath + '/ValidTest', vttmp)
    n_data = len(data[0])
    BucketSize = np.round(n_data // NumOfDivParts)
    if BucketSize % BatchSize != 0:
        BucketSize = BucketSize + (BatchSize - BucketSize % BatchSize)
    bar = progressbar.ProgressBar()
    for i in bar(range(NumOfDivParts - 1)):
        if not TrueList:
            SaveFunction(OutPath + '/TrainData{}'.format(str(i + 1).zfill(3)),
                         [data[0][i * BucketSize:(i + 1) * BucketSize], data[3][i * BucketSize:(i + 1) * BucketSize]],
                         )
        else:
            tmp=[data[0][i * BucketSize:(i + 1) * BucketSize],
                          [data[3][i * BucketSize:(i + 1) * BucketSize]]]
            SaveFunction(OutPath + '/TrainData{}'.format(str(i + 1).zfill(3)),
                         tmp
                         )

    if not TrueList:
        SaveFunction(OutPath + '/TrainData{}'.format(str(NumOfDivParts).zfill(3)),
                     [data[0][(NumOfDivParts - 1) * BucketSize:NumOfDivParts * BucketSize],
                      data[3][(NumOfDivParts - 1) * BucketSize:NumOfDivParts * BucketSize]])
    else:
        SaveFunction(OutPath + '/TrainData{}'.format(str(NumOfDivParts).zfill(3)),
                     [data[0][(NumOfDivParts - 1) * BucketSize:NumOfDivParts * BucketSize],
                     [data[3][(NumOfDivParts - 1) * BucketSize:NumOfDivParts * BucketSize]]])

