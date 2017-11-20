# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 15:06:02 2016

@author: aeloyq
"""


def mnist_sample(x, y, result, results):
    sample_str = ""
    picture = x
    for i in range(28):
        for j in range(28):
            if picture[i * 28 + j] == 0:
                sample_str += '  '
            else:
                sample_str += '* '
        sample_str += '\r\n'
    sample_str += "Recognized:%d      Truth:%d" % (int(result), int(y))
    return sample_str


def image_sample(x, y, result, results):
    sample_str = ""
    image = x[0]
    for i in range(28):
        for j in range(28):
            if image[i, j] == 0:
                sample_str += '  '
            else:
                sample_str += '* '
        sample_str += '\r\n'
    sample_str += "Recognized:%d      Truth:%d" % (int(result), int(y))
    return sample_str


def xor_sample(x, y, result, results):
    sample_str = ""
    a = int(x[0])
    b = int(x[1])
    c = int(result)
    sample_str += "%d xor %d = %d (%d)" % (a, b, c, int(y))
    return sample_str


def add_sample(x, y, result, results):
    sample_str = "Sample:\r\n"
    a = b = c = y_new = ''
    for i in x[0]:
        a += '%d' % i
    for i in x[1]:
        b += '%d' % i
    for i in result:
        c += '%d' % i
    for i in y:
        y_new += '%d' % i
    a = int(a, 2)
    b = int(b, 2)
    c = int(c, 2)
    y_new = int(y_new, 2)
    sample_str += "%d + %d = %d (%d)" % (a, b, c, y_new)
    return sample_str
