#!/usr/bin/env python3

import numpy as np
import cv2
import scipy.ndimage
import scipy.spatial
import sys

def reg_peaks(im, peaks, thresh = 36, anisotropy = (6,1,1)):
    """
    function that drops peaks that are found too close to each other (Not used by Spool or Thread)
    """
    peaks = np.array(peaks)
    anisotropy = np.array(anisotropy,dtype=int)
    for i in range(len(anisotropy)):
        peaks[:,i]*=anisotropy[i]

    diff = scipy.spatial.distance.cdist(peaks,peaks,metric='euclidean')
    complete = False
    while not complete:
        try:
            x,y = np.where(diff == np.min(diff[diff!=0]))
            x,y = x[0],y[0]
            if diff[x,y] >= thresh:
                complete = True
            else:
                peak1 = peaks[x] // anisotropy
                peak2 = peaks[y] // anisotropy

                if im[tuple(peak1)] > im[tuple(peak2)]:
                    diff = np.delete(diff, y, axis = 0)
                    diff = np.delete(diff, y, axis = 1)
                    peaks = np.delete(peaks, y, axis = 0)
                else:
                    diff = np.delete(diff, x, axis = 0)
                    diff = np.delete(diff, x, axis = 1)
                    peaks = np.delete(peaks, x, axis = 0)
        except:
            print("\nUnexpected error: ", sys.exc_info())
            complete = True

    for i in range(len(anisotropy)):
        peaks[:,i]= peaks[:,i]/anisotropy[i]
    return peaks.astype(int)


def convAxis(im, axis, kernel):
    '''
    1d convolution along arbitrary axis of an image. 


    Method is to reshape numpy array into a 2d image, then apply cv2.filter2D to perform a 2d filtering operation utilizing your 1d kernel, and then reshaping the entire array back 

    Parameters
    ----------
    im : np.array
        numpy array for image data. can be an arbitrary number of dimensions
    axis : int
        integer axis designating along which axis to do the convolution
    kernel : np.array
        1d numpy array designating the kernel. 

    Outputs
    -------
    im : np.array
        numpy array for filtered image data

    '''
    if len(kernel.shape) == 1:
        kernel.reshape((len(kernel),1))
    axis = int(axis)
    shape = np.array(im.shape)
    shape[0],shape[axis] = shape[axis],shape[0]
    return np.swapaxes(cv2.filter2D(
        np.swapaxes(im,axis,0).reshape((im.shape[axis], int(im.size/im.shape[axis]))), 
        -1, 
        kernel).reshape((shape)),0,axis)

def gaussian2d(im, *args):
    """
    performs 2d convolution along last 2 dimensions of image data. currently, only handles 3d numpy arrays

    Parameters
    ----------
    im : np.array
        3d numpy array. the last 2 axes will be filtered through a 2d gaussian convolution
    gaussian : tuple (no keyword)
        tuple containing the size of the gaussian to use. can be passed anywhere from 2 to 4 arguments. if 2 arguments, the interpretation will be (width, sigma) with the window-width and sigma applied to both the x and y axes. if 3 arguments, the first two will be the width in x and y respectively, with the last being the sigma. if 4 arguments, it will be interpreted as (widthx, widthy, sigmax, sigmay)

    
    """
    width_x = width_y = 19
    sigma_x = sigma_y = 6        
    if args:
        #case: tuples
        if isinstance(args[0],tuple):
            if len(args[0]) == 2:
                width_x = args[0][0]
                width_y = args[0][0]
                sigma_x = args[0][1]
                sigma_y = args[0][1]
            if len(args[0]) == 3:
                width_x = args[0][0]
                width_y = args[0][1]
                sigma_x = args[0][2]
                sigma_y = args[0][2]    
            if len(args[0]) == 4:
                width_x = args[0][0]
                width_y = args[0][1]
                sigma_x = args[0][2]
                sigma_y = args[0][3]
        for i in range(im.shape[0]):
            im[i] = cv2.GaussianBlur(im[i], (width_x,width_y), sigma_x, sigmaY=sigma_y)
    else:
        for i in range(im.shape[0]):
            im[i] = cv2.GaussianBlur(im[i], (19,19), 6)

    return im

def gaussian3d(im, *args):
    """
    applies 3d gaussian convolution over the image as a separable 2d-1d convolution. should be able to handle 3d and 4d images, if the 4d image is indexed (z,c,x,y). 

    Parameters
    ----------
    im : np.array
        3d or 4d numpy array with indexing (z,x,y) or (z,c,x,y)
    gaussian : tuple (no keyword)
        specifies aspects of gaussian kernel. can handle 4 or 6 arguments. if 4 arguments, will be read as 
            first arg: width x and width y
            second arg : width z
            third : sigma x and y
            fourth : sigma z
        if 6 arguments, will be read as (widthx,y,z, sigma x,y,z)



    """
    #print("Applying 3D Gaussian filter...")
    width_x = width_y = 19
    width_z = 5
    sigma_x = sigma_y = 6
    sigma_z = 1
    if args:
        if isinstance(args[0],tuple):
            if len(args[0]) == 4:
                width_x = width_y = args[0][0]
                width_z = args[0][2]
                sigma_x = sigma_y = args[0][1]
                sigma_z = args[0][3]
            elif len(args[0]) == 6: 
                width_x = args[0][0]
                width_y = args[0][1]
                width_z = args[0][4]
                sigma_x = args[0][2]
                sigma_y = args[0][3]
                sigma_z = args[0][5]
    s = im.shape


    im = im.reshape(-1, im.shape[-2], im.shape[-1])
    im = gaussian2d(im, (width_x,width_y, sigma_x,sigma_y))
    im = im.reshape(s)
    #print('Starting 3D convolution')
    
    if len(s) == 3:
        im = convAxis(im, 0, cv2.getGaussianKernel(width_z,sigma_z))
    
    return im 


def medFilter2d(im,*size):
    """
    Median filter image, applied to all z's and channels

    Parameters
    ----------
    im : np.array
        3 or 4d numpy array for your image data
    size : int (optional)
        width of median filter window. default is 3
    """
    if size:
        size = size[0]
    else:
        size = 3
    if len(im.shape) == 3:
        for i in range(im.shape[0]):
            im[i] = cv2.medianBlur(im[i],size)
    elif len(im) == 4:
        for i in range(im.shape[0]):
            for j in range(im.shape[1]):
                im[i,j] = cv2.medianBlur(im[i,j],size)

    return im

def medFilter3d(im,*size):
    """
    Median filter image in 3d, applied to all z's. only handles 3d images at the moment, and runs pretty damn slow. 

    Parameters
    ----------
    im : np.array
        3 or 4d numpy array for your image data
    size : int (optional)
        width of median filter window. default is 3
    """
    if size:
        size = size[0]
    else:
        size = 3
    return scipy.ndimage.filters.median_filter(im, size = size)



def findpeaks2d(im):
    """
    finds peaks in 2d on each z step using np.roll and comparisons. currently only handles 3d implementations

    Parameters
    ----------
    im : np.array
        3d numpy array within which peaks will be found. 

    Outputs:
    --------
    peaks : np.array
        np.array containing the indices for where peaks were found
    """
    #print('Finding peaks...')
    centers_newway = np.array(np.where(\
              (np.roll(im,1, axis = 1) < im) \
            * (np.roll(im,-1, axis = 1) < im) \
            * (np.roll(im,1, axis = 2) < im) \
            * (np.roll(im,-1, axis = 2) < im) \
            * (im!=0))).T

    return centers_newway

def findpeaks3d(im):
    """ 
    find peaks in the 3d image and return as a list of tuples. handles 3d and 4d images. with 4d images, images need to be indexed (z,c,x,y)

    Parameters
    ----------
    im : np.array
        3d or 4d numpy array within which peaks will be found. 

    Outputs:
    --------
    peaks : np.array
        np.array containing the indices for where peaks were found
    """

    if len(im.shape) == 3:
        centers_newway = np.array(np.where(
              (np.roll(im,1, axis = 0) < im) \
            * (np.roll(im,-1, axis = 0) < im) \
            * (np.roll(im,1, axis = 1) < im) \
            * (np.roll(im,-1, axis = 1) < im) \
            * (np.roll(im,1, axis = 2) < im) \
            * (np.roll(im,-1, axis = 2) < im) \
            * (im!=0))).T

    elif len(im.shape)==4:
        for i in range(im.shape[1]):
            if i == 0:
                centers_newway = np.array(np.where(np.roll(np.roll(im[:,i],1, axis = 0) >im[:,i], -1, axis = 0) * np.roll(np.roll(im[:,i],-1, axis = 0) > im[:,i], 1, axis = 0) * np.roll(np.roll(im[:,i],1, axis = 1) > im[:,i], -1, axis = 1) * np.roll(np.roll(im[:,i],-1, axis = 1) > im[:,i], 1, axis = 1) * (im[:,i]!=0))).T
            else:
                #print(centers_newway.shape)
                #print(np.array(np.where(np.roll(np.roll(im[:,i],1, axis = 0) >im[:,i], -1, axis = 0) * np.roll(np.roll(im[:,i],-1, axis = 0) > im[:,i], 1, axis = 0) * np.roll(np.roll(im[:,i],1, axis = 1) > im[:,i], -1, axis = 1) * np.roll(np.roll(im[:,i],-1, axis = 1) > im[:,i], 1, axis = 1) * (im[:,i]!=0))).T.shape)
                centers_newway = np.concatenate((centers_newway, np.array(np.where(np.roll(np.roll(im[:,i],1, axis = 0) >im[:,i], -1, axis = 0) * np.roll(np.roll(im[:,i],-1, axis = 0) > im[:,i], 1, axis = 0) * np.roll(np.roll(im[:,i],1, axis = 1) > im[:,i], -1, axis = 1) * np.roll(np.roll(im[:,i],-1, axis = 1) > im[:,i], 1, axis = 1) * (im[:,i]!=0))).T ), axis = 0)

    return centers_newway


