# -*- coding: utf-8 -*-
"""
Computer Vision Utilities

@author: Adam Raymond Kampff (ARK)
"""

import numpy as np


# Return a list of all rectanglular ROIs in an imageJ zip file (left, top, right, bottom) 
def read_imagej_roi(fileobj):
    '''
    points = read_roi(fileobj)
 
    Read ImageJ's ROI format
    '''
 
    pos = [4]
    def get8():
        pos[0] += 1
        s = fileobj.read(1)
        if not s:
            raise IOError('readroi: Unexpected EOF')
        return ord(s)
 
    def get16():
        b0 = get8()
        b1 = get8()
        return (b0 << 8) | b1
 
    def get32():
        s0 = get16()
        s1 = get16()
        return (s0 << 16) | s1
 
    def getfloat():
        v = np.int32(get32())
        return v.view(np.float32)
 
    magic = fileobj.read(4)
    if magic != 'Iout':
        raise IOError('Magic number not found')
    version = get16()
 
    # It seems that the roi type field occupies 2 Bytes, but only one is used
    roi_type = get8()
    # Discard second Byte:
    get8()
 
    if roi_type != 1:
        raise ValueError('roireader: ROI type %s not supported (!= 7)' % roi_type)
 
    top = get16()
    left = get16()
    bottom = get16()
    right = get16()

    points = np.zeros(4)
    points[0] = left
    points[1] = top
    points[2] = right
    points[3] = bottom
    return points
 
def read_roi_zip(fname):
    import zipfile
    with zipfile.ZipFile(fname) as zf:
        return [read_imagej_roi(zf.open(n))
                    for n in zf.namelist()]
