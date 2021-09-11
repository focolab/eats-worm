#!/usr/bin/env python3

import numpy as np
import cv2
import json
import scipy.ndimage
import scipy.optimize
import scipy.spatial
from skimage._shared.coord import ensure_spacing
import skimage.feature
import skimage.morphology
import sys
from scipy.spatial.distance import pdist, squareform

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
    return np.rint(peaks).astype(int)


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

def peak_filter_2(data=None, params=None):
    """peakfilter 2: template matching filter scheme"""
    # handle params
    if params is None:
        params = {}
    p = {'threshold':0.5, 'template':None}
    p.update(params)

    template = p.get('template', None)
    if template is None:
        raise Exception('template required')

    # the template match result needs padding to match the original data dimensions
    res = skimage.feature.match_template(data, template)
    pad = [int((x-1)/2) for x in template.shape]
    res = np.pad(res, tuple(zip(pad, pad)))
    filtered = res*np.array(res>p['threshold'])
    footprint = np.zeros((3,3,3))
    footprint[1,:,1] = 1
    footprint[1,1,:] = 1
    for i in range(3):
        filtered = skimage.morphology.erosion(filtered, selem=footprint)
    labeled_features, num_features = scipy.ndimage.label(filtered)
    centers = []
    for feature in range(num_features):
        center = scipy.ndimage.center_of_mass(filtered, labels=labeled_features, index=feature)
        centers.append(list(center))

    centers = np.array(centers)
    centers = np.rint(centers[~np.isnan(centers).any(axis=1)]).astype(int)
    intensities = filtered[tuple(centers.T)]
    # Highest peak first
    idx_maxsort = np.argsort(-intensities)
    centers = centers[idx_maxsort]

    centers = ensure_spacing(centers, spacing=9)
    return np.rint(centers).astype(int)

def peakfinder(data=None, peaks=None, params=None, pad=None, legacy=False):
    """Do filtering and peakfinding, then some extra useful things
    Parameters
    ----------
    filt : function
        A function of the form ```filtered=pf(data, params)``` that takes
        a 3D array and parameter dict and returns a filtered array (same size).
    chunk : DataChunk
    params : dict
        These get passed to pf
    pad : list/array
        Defines bounding box around a peak by padding voxels to both sides
    legacy : bool
        If true, use legacy 3D peakfinding (6NN), otherwise 26NN
    Returns
    -------
    out : dict
        May be bloated, but holds extra info useful for diagnostics
    """
    
    # cull out peaks for which the bounding box goes beyond data bounds
    peaks_inbounds, blobs, bboxes = [], [], []
    for i, x in enumerate(peaks):
        blob = SegmentedBlob(index=i, pos=x, dims=["z","x","y"], pad=pad)
        try:
            chreq = blob.chreq()
            bbox = data[chreq["z"][0]:chreq["z"][1], chreq["x"][0]:chreq["x"][1], chreq["y"][0]:chreq["y"][1]]
            peaks_inbounds.append(x)
            blobs.append(blob)
            bboxes.append(bbox)
        except:
            pass

    avg3D_chunk = np.mean([x.data for x in bboxes], axis=0)

    return np.array(bboxes), blobs

class BlobTemplate(object):
    """3D blob template
    Class for a "learned" 3D blob template, should be approximately Gaussian
    and isotropic.
    - holds the chunk of data
    - handles anisotropic voxel size/spacing
    - computes intensity profiles and Gaussian fits
    - plots a bunch of visualizations
    """
    def __init__(self, data=None, scale=None, blobs=None):
        """
        Parameters
        ----------
        data: (ndarray) ndarray containing the template data
        scale: (dict) voxel size/spacing for each dimension (in micron)
        blobs: (list) List of SegmentedBlobs (used to derive the template)
        """

        if len(data.shape) > 4:
            raise NotImplementedError('Color templates not yet implemented')

        self.blobs = blobs
        
        self.data = data
        if scale is None:
            scale = {d:1 for d in data.shape}
        self.scale = scale
        
        # compute some helpful things
        self.chunk_center = {}
        for dim in range(len(data.shape)):
            self.chunk_center[dim] = np.floor(data.shape[dim]/2).astype(int)
    
        self.profiles_1D = self.get_1D_profiles()
                
    def get_1D_center_profile(self, dim=None):
        """1D intensity profile that skewers the chunk center"""
        if dim is None:
            raise Exception('dim required')

        slc = [slice(None)] * len(self.data.shape)
        slc[0] = slice(self.chunk_center[0],self.chunk_center[0] + 1)
        slc[1] = slice(self.chunk_center[1],self.chunk_center[1] + 1)
        slc[2] = slice(self.chunk_center[2],self.chunk_center[2] + 1)
        slc[dim] = slice(0, self.data.shape[dim])
        return self.data[tuple(slc)].squeeze()
        
    def get_1D_profiles(self):
        """Get 1D intensity profiles through center and Gaussian fits
        """
        chunk = self.data
        scale = self.scale

        profiles = {}
        fits = []
        for dim in self.chunk_center:
            x_px = np.arange(chunk.shape[dim])-self.chunk_center[dim]
            x_um = x_px*scale[dim]
            val = self.get_1D_center_profile(dim=dim)
            profiles[dim] = [x_px, x_um, val]

            # gaussian fit (in pixel and micron units)
            # special case zero padding
            if len(x_px) == 1:
                x_px = np.asarray([x_px[0]]*3)
                val = np.asarray([val]*3)
                            
            popt = self._gauss_fitter_1D(x_px, val)
            cols = ['amplitude', 'mean_px', 'std_px']
            fd = dict(zip(cols, popt))
            fd['mean_um'] = fd['mean_px']*scale[dim] 
            fd['std_um'] = fd['std_px']*scale[dim]
            fits.append(fd)
        
        out = dict(
            profiles=profiles
        )
                
        return out

    def _gaussian(self, x, amplitude, mean, stddev):
        return amplitude*np.exp(-0.5*((x-mean)/stddev)**2)

    def _gauss_fitter_1D(self, x, data):
        """returns: amplitude, mean, std"""
        popt, _ = scipy.optimize.curve_fit(self._gaussian, x, data)
        popt[2] = np.abs(popt[2])
        return popt

class SegmentedBlob(object):
    """holds the info for a blob that has been segmented from an image(vol)
    Right now, this is only intended to describe a blob in a static, volumetric
    image (CZYX) and is not intended for a time varying blob (termed a thread
    in gcamp_extractor). A more general blob/thread spec, where this is just a
    special case, would be ideal.
    The provenence attribute ```prov``` can 'detected', 'curated', or 'imputed'
    The attributes ```status``` and ```index``` are not really intrinsic to a
    blob, but are tacked on for convenience.
    ```status``` in [-1, 0, 1] is a quick and easy way to classify a blob that
    is (-1) marked for deletion, (0) needs checking, or (1) has passed human
    curation. If a list of ```SegmentedBlob```s is curated in multiple
    sessions, this status is important to preserve.
    ```index``` is just a numerical index for sorting and, like ```status``` is
    bound to the blob.
    ```stash``` is not serialized or saved with a SegmentedBlob, but could be
    used to hold temporary associated information, e.g. a pre-computed
    bounding box DataChunk or voxel mask, during an interactive session. Such
    a mask or masks could/should also be their own attributes.
    """
    def __init__(self, index=-1, pos=None, status=0, pad=[10, 10, 4], dims=['X','Y','Z'], ID='', prov='x'):
        self.pos = pos          # position
        self.dims = dims        # let's be explicit and not lose track of these
        self.ID = ID            # string for the cell name
        self.pad = pad
        self.prov = prov        # provenence
        self.status = status    # not well defined yet
        self.index = index      # just an int for tracking
        self.stash = {}         # could stash the bbox datachunk here for speed

    @property
    def posd(self):
        """position, but in dict form so dimensions don't get twisted"""
        return dict(zip(self.dims, self.pos))

    def lower_corner(self):
        """lower corner of the bounding box"""
        return {d:r-p for r,p,d in zip(self.pos, self.pad, self.dims)}

    def clone(self):
        """This is mainly used to create a dummy/provisional blob in a GUI.
        A hard copy is done so as not to be referring to another blob's
        attributes
        """
        po = [x for x in self.pos]
        pa = [x for x in self.pad]
        di = [x for x in self.dims]
        return SegmentedBlob(pos=po, pad=pa, dims=di, status=0, prov=self.prov)

    def to_jdict(self):
        """serialize to a json-izable dictionary"""
        dd = dict(
            pos=[int(x) for x in self.pos],
            pad=[int(x) for x in self.pad],
            dims=self.dims,
            ID=self.ID,
            status=int(self.status),
            index=int(self.index),
            prov=self.prov
        )
        return dd

    def chreq(self, offset=None, rounding=None, pad=None):
        """chunk request for the blob's bounding box
        Parameters
        ----------
        offset : dict
            A dictionary of index offsets for the parent DataChunk,
            needed to correctly extract a bounding chunk from a dataset that
            has been cropped down (i.e. a DataChunk with an offset).
            For example, {'X': 10, 'Y':0, 'Z':4}
        rounding : str
            The method to round a non-integer blob center to have
            integer coordinates (i.e. associate it with a voxel).
            TODO: implement 'closest'
        pad : dict
            (optional to override the attribute)
            Same format as offset.
            The chunk is defined by the central voxel plus padding. i.e. the
            interval [X-pad['X'], X+pad['X']] and the equivalent for 'Y' and
            'Z'.
        Returns
        -------
        req: dict
            A DataChunk.subchunk request
        Assuming that the raw data is in a DataChunk, this is a helper that
        generates the DataChunk.subchunk() request, for a blob's bounding box.
        TODO: make sure it does not run outside of parent box dimensions!
        """
        if rounding == None:
            pos_int = self.pos
        elif rounding == 'floor':
            pos_int = np.floor(self.pos).astype(np.int)
        elif rounding == 'closest':
            raise NotImplementedError('rounding (%s) not YET implemented', 'closest')
        else:
            raise NotImplementedError('rounding (%s) not implemented', rounding)

        if offset is None:
            offset = {}

        if pad is None:
            pad_arr = self.pad
        else:
            pad_arr = [pad[k] for k in self.dims]

        req = {}
        for r,p,d in zip(pos_int, pad_arr, self.dims):
            req[d] = (r-p-offset.get(d, 0), r+p-offset.get(d, 0)+1)
        return req


    def load_blobs(j=None):
        """load blobs from json"""
        with open(j) as jfopen:
            data = json.load(jfopen)
        return [SegmentedBlob(**x) for x in data]