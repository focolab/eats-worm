#import matlab.engine
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import os

def generate_histograms(image, scale):
    
    image = np.asarray(image)

    im_flat = image.reshape(-1, image.shape[-1])

    fig, axs = plt.subplots(2,2)

    for i, ax1 in enumerate(axs):
        for j, ax in enumerate(ax1):

            hist, bins = np.histogram(im_flat[:,i*2+j], 256, [0, 256*scale] )
            cdf = hist.cumsum()
            cdf_normalized = cdf * hist.max()/cdf.max()
            ax.axvline(np.max(im_flat[:,i*2+j]),linestyle='--') 
            ax.plot(cdf_normalized, color = 'b')
            ax.hist(im_flat[:,i*2+j], bins= np.arange(256)*scale, color= 'red')
            ax.set_xlabel('color channel gray count')
            ax.set_ylabel('pixel count')
            ax.set_xlim([0,256*scale])
            ax.legend(('max value', 'cdf', 'hist'), loc = 'upper right')
    
    axs[0,0].set_title('red histogram')
    axs[0,1].set_title('green histogram')
    axs[1,0].set_title('blue histogram')
    axs[1,1].set_title('white histogram')

    plt.show()


def equalize_hist(RGBW, threshs):
    '''
    thresh defines value above which to perform the histogram equalization
    loop through each pixel in image and transform based on histogram equalization
    '''

    size = RGBW.shape

    RGBW_new = np.zeros(size)

    flat = RGBW.reshape(-1, RGBW.shape[-1])

    for l in range(size[3]):
        channel = flat[:,l]

        thresh = threshs[l]
        
        hist_to_eq = channel[np.where(channel>=thresh)]
        N = len(hist_to_eq)
        num_bins = 4096-thresh
        hist, bins = np.histogram(hist_to_eq, num_bins, [thresh, 4096])
        cdf = hist.cumsum()
        
        for i in range(size[0]):
            for j in range(size[1]):
                for k in range(size[2]):
                        val = RGBW[i,j,k,l]

                        if val >= thresh:
                            val_index = np.where(bins==val)
                            cum_prob = cdf[val_index]/N
                            new_val = np.round(cum_prob*(num_bins-1))+thresh

                            RGBW_new[i,j,k,l] = new_val
                        
                        else:
                            RGBW_new[i,j,k,l] = val

    return RGBW_new

def match_histogram(A, ref, A_max, ref_max): 
    image = np.asarray(A)
    ref_im = np.asarray(ref)

    #A_max = np.ma.minimum_fill_value(image)
    #ref_max = np.ma.minimum_fill_value(ref_im)

    im_flat = image.reshape(-1, image.shape[-1]) #flatten images 
    ref_flat = ref_im.reshape(-1, ref_im.shape[-1])

    newim = np.zeros(A.shape)

    for l in range(image.shape[3]):
        chan_flat = im_flat[:,l]
        chan_ref_flat = ref_flat[:,l]

        hist, bins = np.histogram(chan_flat, A_max, [0, A_max]) #generate histograms
        refhist, refbins = np.histogram(chan_ref_flat, ref_max, [0, ref_max])

        cdf = hist.cumsum()/ chan_flat.size # generate cdf of histograms
        cdf_ref = refhist.cumsum()/ chan_ref_flat.size

        M = np.zeros(A_max) 

        for idx in range(A_max):
            ind = np.argmin(np.abs(cdf[idx]-cdf_ref)) # store pixel values with matching cdf from reference image
            M[idx] = ind

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                for k in range(image.shape[2]):
                    newim[i,j,k,l] = M[A[i,j,k,l]]

    return newim

    
