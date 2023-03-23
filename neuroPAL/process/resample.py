#import matlab.engine
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import os
from scipy.interpolate import RegularGridInterpolator
import tifffile
from scipy.ndimage import zoom


def zoom_interpolate(new_res, old_res, data):

    '''
    Inputs:

    new_res - the new resolution that you want to upsample/downsample the image too.
    Should be length 3 in order X resolution, Y resolution, Z resolution.

    old_res - the resolution of the original image. Should be length 3 in order 
    X resolution, Y resolution, Z resolution.

    data - the 4D volume that you want to rescale. Order of dimensions should be 
    X, Y, Z, C

    Outputs:

    newim - interpolated image. By default uses spline interpolation of order 3. 

    TODO: currently using ndimage zoom function. Takes ~1 min to run with standard image.
    Optimally will make this more efficient but works for now.
    '''

    sz = data.shape

    xrescale = old_res[0]/new_res[0]
    yrescale = old_res[1]/new_res[1]
    zrescale = old_res[2]/new_res[2]

    newim = zoom(data, (xrescale, yrescale, zrescale,1))

    return newim