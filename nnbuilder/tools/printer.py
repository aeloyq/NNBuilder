# -*- coding: utf-8 -*-
"""
Created on  七月 19 2:33 2017

@author: aeloyq
"""


def lineformatter(StringList, Length=None, LengthList=None, Align='left',FirstColumnLeft=True):
    string = ""
    if Length is not None:
        lengthlist = []
        for _ in StringList:
            lengthlist.append(Length)
    else:
        lengthlist = LengthList
    first_column = True
    for s, l in zip(StringList, lengthlist):
        str_ = s[:l]
        if Align == 'right' and (first_column != True or not FirstColumnLeft):
            str_ = str_.rjust(l)
        elif Align == 'center' and (first_column != True or not FirstColumnLeft):
            ll = max((l - len(str_)) // 2,0)
            lr = l - ll - len(str_)
            str_ = str_.ljust(ll + len(str_))
            str_ = str_.rjust(lr + len(str_))
        else:
            str_ = str_.ljust(l)
        string += (str_)
        first_column = False

    return string


def paragraphformatter(ParaList, Length=None, LengthList=None, Align='left',FirstColumnLeft=True):
    para = ''
    for StringList in ParaList:
        string = ""
        if Length is not None:
            lengthlist = []
            for _ in StringList:
                lengthlist.append(Length)
        else:
            lengthlist = LengthList
        first_column = True
        for s, l in zip(StringList, lengthlist):
            str_ = s[:l]
            if Align == 'right' and (first_column != True and FirstColumnLeft):
                str_ = str_.rjust(l)
            elif Align == 'center' and (first_column != True and FirstColumnLeft):
                ll = max((l - len(str_)) // 2,0)
                lr = l - ll - len(str_)
                str_ = str_.ljust(ll + len(str_))
                str_ = str_.rjust(lr + len(str_))
            else:
                str_ = str_.ljust(l)
            string += (str_)
            first_column = False
        para += string + '\r\n'

    return para[:-4]


def timeformatter(Time):
    if Time < 60:
        return '%.2fs' % Time
    elif Time < 3600:
        return '%dm%ds' % (Time // 60, Time % 60)
    elif Time < (3600 * 24):
        return '%dh%dm' % (Time // 3600, (Time % 3600) // 60)
    elif Time < (3600 * 24 * 7):
        return '%dd%dh' % (Time // (3600 * 24), (Time % (3600 * 24)) // 3600)
    elif Time < (3600 * 24 * 30):
        return '%dw%dd' % (Time // (3600 * 24 * 7), (Time % (3600 * 24 * 7)) // (3600 * 24))
    elif Time < (3600 * 24 * 365):
        return '%dm%dw' % (Time // (3600 * 24 * 30), (Time % (3600 * 24 * 30)) // (3600 * 24 * 7))
    else:
        return '%dy%dm' % (Time // (3600 * 24 * 365), (Time % (3600 * 24 * 365)) // (3600 * 24 * 30))
