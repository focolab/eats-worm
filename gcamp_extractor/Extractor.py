import numpy as np
import pdb
import time
import scipy.spatial
import copy
from .Threads import *
import pickle
import os
from .multifiletiff import *
from improc.segfunctions import *
def mkdir(path):
    try: os.mkdir(path)
    except: pass
import glob
import json
from scipy.ndimage import generate_binary_structure, binary_dilation, rotate
from scipy.spatial.distance import pdist, squareform
from skimage.feature import peak_local_max
from skimage.registration import phase_cross_correlation
from fastcluster import linkage
import scipy.cluster

default_arguments = {
    'root':'/Users/stevenban/Documents/Data/20190917/binned',
    'numz':20,
    'numc':1,
    'frames':[0,1,2,3,4,5,6,7,8,9,10,11],
    'offset':0,
    't':999,
    'gaussian':(25,4,3,1),
    'quantile':0.99,
    'reg_peak_dist':10,
    'anisotropy':(7,1,1),
    'blob_merge_dist_thresh':6.8,
    'remove_blobs_dist':20,
    'infill':True,
    'suppress_output':False,
    'regen':False,
    'algorithm': 'template',
    'algorithm_params': {},
}

def default_quant_function(im, positions, frames):
    """
    Default quantification function, to be used in conjunction with the Extractor class. Takes the mean of the 20 brightest pixels in a 3x7x7 square around the specified position 

    Parameters
    ----------
    im : numpy array
        numpy array representing image volume data at a particular time point
    positions : list
        a list of positions/pixel coordinates to be quantified
    
    Returns
    -------
    activity: list
        list of quantifications corresponding to the positions specified
    """
    timeseries = []
    for i in range(len(positions)):
        center = positions[i]
        z,y,x = [],[],[]
        for i in range(-3,4):
            for j in range(-3,4):
                for k in range(-1,2):
                    if 0 <= round(center[0] + k) < len(frames): 
                        z.append(round(center[0] + k))
                        y.append(round(center[1] + j))
                        x.append(round(center[2] + i))
        masked = im[z,y,x]
        masked.sort()
        timeseries.append(np.mean(masked[-20:]))
    return timeseries

def background_subtraction_quant_function(im, positions, frames):
    """
    Takes the mean of the 20 brightest pixels in a 3x7x7 square around the specified position minus the mean of pixels within a bounding region which are not too close to the roi or other rois

    Parameters
    ----------
    im : numpy array
        numpy array representing image volume data at a particular time point
    positions : list
        a list of positions/pixel coordinates to be quantified
    
    Returns
    -------
    activity: list
        list of quantifications corresponding to the positions specified
    """
    positions = np.rint(np.copy(positions)).astype(int)
    bounding_pad = 20
    timeseries = []
    max_z = len(frames) - 1
    max_x = im.shape[1] - 1
    max_y = im.shape[2] - 1
    pos_z, pos_x, pos_y = positions.T

    pos_masks = np.zeros((len(positions),) + im.shape, dtype=bool)
    background_masks = np.zeros((len(positions),) + im.shape, dtype=bool)
    other_pos_masks = np.zeros((len(positions),) + im.shape, dtype=bool)

    for i in range(len(positions)):
        z_slice = slice(max(0, pos_z[i] - 1), min(max_z + 1, pos_z[i] + 2))
        x_slice = slice(max(0, pos_x[i] - 3), min(max_x + 1, pos_x[i] + 4))
        y_slice = slice(max(0, pos_y[i] - 3), min(max_y + 1, pos_y[i] + 4))
        pos_masks[i, z_slice, x_slice, y_slice] = True
        z_slice = slice(max(0, pos_z[i] - 1), min(max_z + 1, pos_z[i] + 2))
        x_slice = slice(max(0, pos_x[i] - 30), min(max_x + 1, pos_x[i] + 31))
        y_slice = slice(max(0, pos_y[i] - 30), min(max_y + 1, pos_y[i] + 31))
        background_masks[i, z_slice, x_slice, y_slice] = True
        x_slice = slice(max(0, pos_x[i] - 10), min(max_x + 1, pos_x[i] + 11))
        y_slice = slice(max(0, pos_y[i] - 10), min(max_y + 1, pos_y[i] + 11))
        other_pos_masks[:i, z_slice, x_slice, y_slice] = True
        if i + 1< len(positions):
            other_pos_masks[i + 1:, z_slice, x_slice, y_slice] = True
    pos_masks *= ~other_pos_masks
    background_masks *= ~pos_masks
    background_masks *= ~other_pos_masks

    for i in range(len(positions)):
        roi_intensities = im[pos_masks[i]]
        roi_intensities.sort()
        background_intensities = im[background_masks[i]]
        timeseries.append((np.mean(roi_intensities[-20:]) - np.mean(background_intensities)) / np.mean(background_intensities))
    return timeseries

def quantify(mft=None, spool=None, quant_function=default_quant_function, suppress_output=False):
    """
    generates timeseries based on calculated threads. 

    Parameters
    ----------
    quant_function : function
        function that takes in a list of positions/pixel indices, and returns a
        list of floats that describe neuronal activity. A default
        quantification function is included. It takes the 10 brightest pixels
        in a 6x6 square around the position and averages them. 

        Parameters
        ----------
        im : numpy array
            an N (=3) dimensional numpy array of an image volume taken at some time point
        positions : list
            a list of positions/pixel indices of found centers

        Returns
        -------
        activity : list
            list of floats representing neuronal activity. should be returned in the same order as positions, i.e. activity[0] corresponds to positions[0]

    """
    mft.t = 0
    num_threads = len(spool.threads)
    num_t = spool.t

    timeseries = np.zeros((num_t,len(spool.threads)))
    for i in range(num_t):
        timeseries[i] = quant_function(mft.get_t(),[spool.threads[j].get_position_t(i) for j in range(len(spool.threads))], mft.frames)

        if not suppress_output:
            print('\r' + 'Frames Processed (Quantification): ' + str(i+1)+'/'+str(num_t), sep='', end='', flush=True)

    print('\r')

    return timeseries


def load_extractor(path):
    """
    Function for loading an existing extractor object

    Parameters
    ----------
    path : str
        path to the folder containing extractor-objects folder. hypothetically, any containing folder will do, but if it contains multiple extractor-objects, then might break
    
    Returns
    -------
    extractor : extractor
        loaded extractor 
    """
    folders = []
    if 'extractor-objects' not in path:

        for r, d, f in os.walk(path):
            for folder in d:
                if 'extractor-objects' in folder:
                    folders.append(os.path.join(r, folder))
        if len(folders) == 0:
            print('no extractor objects found')
        elif len(folders) != 1:
            print('multiple extractor-objects found')
            return 1
        for folder in folders:
            if folder[-1]!='/':
                folder += '/'
    else:
        if path[-1] != '/':
            path += '/'
        folders.append(path)
        
        
    paramsf = folders[0]+'/params.json'
    mftf = folders[0]+'/mft.obj'
    threadf = folders[0]+'/threads.obj'
    timef = folders[0]+'/timeseries.txt'

    with open(paramsf) as f:
        params = json.load(f)
    params['regen_mft'] = False
    e = Extractor(**params)

    try:
        with open(threadf,'rb') as f:
            thread = pickle.load(f)
        e.spool = thread
    except:
        pass

    try:
        time = np.genfromtxt(timef)
        e.timeseries = time
    except:
        pass

    return e



class Extractor:
    """
    Timeseries extractor for GCaMP neuroimaging of paralyzed C. elegans. 
    
    Arguments
    ---------
    root:str
        a string containing path to directory containing images to be processed
    output_dir:str
        a string containing path to directory to use for output. Default is root
    numz:int
        an integer of the number of frames taken per time point. Default is 10. TODO:self.get numz from image metadata processing
    frames:list
        a list of integers that contains frames mod numz to include. i.e. if 10 frames per time point, and for some reason you want to throw away the 8th and 10th frame, pass the list [0,1,2,3,4,5,6,8]. Note that this uses Python indexing, which starts at 0. default is to include all frames, i.e. list(range(numz))
    offset:int
        an integer that gives the number of frames to throw away at the beginning of the recording. default is 0
    t:int
        number of time points in the recording. default is floor of the total number of frames in the recording, divded by the number of frames per time point
    3d:bool
        boolean for whether to process in 2d or 3d. If 3d is true, gaussian filtering and peak finding will be performed in 3d. otherwise, it will apply to each z slice independently.
    gaussian:list or tuple
        list or tuple of the width and standard deviation of the gaussian filtering step. note that the gaussian filter must have odd width. 
    median:int
        size for median filter. note that unless using image with dtype uint8, filter sizes are limited to 1, 3, and 5. 
    quantile:float
        float between 0 and 1 inclusive for the thresholding step, where all pixels below the quantile specified will be thresholded to 0.
    reg_peak_dist:self.float
        float that 
    

    regen : bool
        whether or not the object was initialized as a new object or a regenerated object from saved data

    Attributes
    ---------
    im:MultiFileTiff
        a MultiFileTiff object for the time series data to be analyzed. See multifiletiff documentation for arguments, attributes, and methods
    spool: Spool
        a Spool object that contains methods for registering incoming neuron centers to existing centers, and for infilling the positions of neurons not found with the first pass. Seself.spool documentation for arguments, attributes, and methods

    Methods
    -------
    

    calc_blob_threads(infill=True/False)
        calculates blob threads. rois are foudn via gaussian filtering, thresholding, max peak finding. 

    """

    def __init__(self, *args, **kwargs):

        ### Specifying all parameters, and filling in initial values if they weren't passed
        if len(args)!=0:
            self.root = args[0]
            kwargs['root'] = args[0]
        elif kwargs.get('root'):
            self.root = kwargs['root']
        else:
            print('did not pass root directory')
            return 0

        self.output_dir = os.path.join(kwargs.get('output_dir', self.root), 'extractor-objects')
        os.makedirs(self.output_dir, exist_ok=True)
        self.numz = kwargs.get('numz', 10)
        self.numc = kwargs.get('numc', 1)
        self.frames= kwargs.get('frames', list(range(self.numz)))
        self.offset = kwargs.get('offset', 0)
        self.anisotropy = kwargs.get('anisotropy', (6,1,1))
        self.mip_movie = kwargs.get('mip_movie', True)
        self.marker_movie = kwargs.get('marker_movie', True)
        _regen_mft = kwargs.get('regen_mft')
        self.im = MultiFileTiff(self.root, output_dir=self.output_dir, anisotropy=self.anisotropy, offset=self.offset, numz=self.numz, numc=self.numc, frames=self.frames, regen=_regen_mft)
        self.im.save()
        self.im.t = 0
        self.blobthreadtracker_params = {k: v for k, v in kwargs.items() if k not in vars(self)}
        self.t = kwargs.get('t', 0)

        ### Dump a record of input parameters
        if kwargs.get('regen'):
            pass
        else:
            kwargs['regen']=True
            with open(os.path.join(self.output_dir, 'params.json'), 'w') as json_file:
                json.dump(kwargs, json_file)

    def calc_blob_threads(self):
        """peakfinding and tracking"""
        x = BlobThreadTracker(mft=self.im, params=self.blobthreadtracker_params)
        self.spool = x.calc_blob_threads()
        print('Saving blob timeseries as numpy object...')
        self.spool.export(f=os.path.join(self.output_dir, 'threads.obj'))

    def quantify(self, quant_function=default_quant_function):
        """generates timeseries based on calculated threads"""
        self.timeseries = quantify(mft=self.im, spool=self.spool, quant_function=quant_function)
        self.save_timeseries()

    def save_timeseries(self):
        print('Saving timeseries as text file...')
        np.savetxt(os.path.join(self.output_dir, 'timeseries.txt'), self.timeseries)

    def results_as_dataframe(self):
        ## TODO: dims should be a class attribute or accessible somewhere
        dims = ['Z', 'Y', 'X']
        df = self.spool.to_dataframe(dims=dims)
        df['gce_quant'] = self.timeseries.T.ravel()
        return df

    def save_dataframe(self):
        csv = os.path.join(self.output_dir, 'spool.csv')
        print('Export threads+quant:', csv)
        df = self.results_as_dataframe()
        df.to_csv(csv, float_format='%6g')

    def save_MIP(self, fname = ''):

        # if passed with no argument, save with default
        if fname == '':
            fname = os.path.join(self.output_dir, 'MIP.tif')
        
        # if just filename specified, save in extractor objects with the filename 
        elif '/' not in fname and '\\' not in fname:
            fname = os.path.join(self.output_dir, fname)


        # if the filename passed is a directory: 
        if os.path.isdir(fname):
            fname = os.path.join(fname, 'MIP.tif')
        # if filename isn't a directory but ends in .tif
        elif fname[-4:] == '.tif':
            pass #save filename as is
        elif fname[-5:] == '.tiff':
            pass

        # if filename isn't a directory and doesn't end in .tif, append .tif to filename
        else:
            fname = fname + '.tif'

        _t = self.im.t
        self.im.t = 0
        _output = np.zeros(tuple([self.t]) + self.im.sizexy, dtype = np.uint16)
        
        with tiff.TiffWriter(fname,bigtiff = True) as tif:
            for i in range(self.t):
                tif.save(np.max(self.im.get_t(), axis = 0))
                print('\r' + 'MIP Frames Saved: ' + str(i+1)+'/'+str(self.t), sep='', end='', flush=True)

        print('\n')



class BlobThreadTracker():
    """peakfinding and tracking

    The aim of this class is to have the current peakfinding and tracking
    scheme(s) implemented in a self contained manner, so that Extractor can
    stay lean and mean.

    This is WIP and should be further split into peakfinding/tracking/MOCO/etc.

    """
    def __init__(self, mft=None, params=None):
        """

        parameters:
        -----------
        mft : (MultiFileTiff)
        params : (dict)
            ALL of the parameters specific to peakfinding, tracking etc
        """
        ## datasource
        self.im = mft
        self.frames = self.im.frames

        ## peakfinding/spooling parameters
        self.gaussian = params.get('gaussian', (25,4,3,1))
        self.median = params.get('median', 3)
        self.quantile = params.get('quantile', 0.99)
        self.reg_peak_dist = params.get('reg_peak_dist', 40)
        self.blob_merge_dist_thresh = params.get('blob_merge_dist_thresh', 6)
        self.remove_blobs_dist = params.get('remove_blobs_dist', 20)
        self.suppress_output = params.get('suppress_output', False)
        self.register = params.get('register_frames', False)
        self.predict = params.get('predict', True)
        self.algorithm = params.get('algorithm', 'template')
        self.algorithm_params = params.get('algorithm_params', {})
        try:
            self.algorithm_params['templates_made'] = type(self.algorithm_params['template']) != bool
        except:
            self.algorithm_params['templates_made'] = False

        # self.t is a time index cutoff for partial analysis
        self.t = params.get('t', 0)
        if self.t==0 or self.t>(self.im.numframes-self.im.offset)//self.im.numz:
            self.t=(self.im.numframes-self.im.offset)//self.im.numz


    def calc_blob_threads(self):
        """
        calculates blob threads

        returns
        -------
        spool: (Spool)
        """
        self.spool = Spool(self.blob_merge_dist_thresh, max_t=self.t, predict=self.predict)

        # handle an ugly workaround for peak_local_max not supporting anisotropy. this stuff is only needed when
        # using skimage or template matching, but putting it here allows us to avoid redoing the matmuls every iteration
        expanded_shape = tuple([dim_len * ani for dim_len, ani in zip(self.im.get_t(0).shape, self.im.anisotropy)])
        mask = np.zeros(expanded_shape, dtype=np.uint16)
        mask[tuple([np.s_[::ani] for ani in self.im.anisotropy])] = 1

        for i in range(self.t):
            im1 = self.im.get_t()
            im1 = medFilter2d(im1, self.median)
            if self.gaussian:
                im1 = gaussian3d(im1, self.gaussian)
            im1 = np.array(im1 * np.array(im1 > np.quantile(im1, self.quantile)))
            if self.algorithm == 'skimage':
                expanded_im = np.repeat(im1, self.im.anisotropy[0], axis=0)
                expanded_im = np.repeat(expanded_im, self.im.anisotropy[1], axis=1)
                expanded_im = np.repeat(expanded_im, self.im.anisotropy[2], axis=2)
                expanded_im *= mask
                try:
                    peaks = peak_local_max(expanded_im, min_distance=self.skimage[1], num_peaks=self.skimage[0])
                    peaks //= self.im.anisotropy
                except:
                    print("No peak_local_max params supplied; falling back to default inference.")
                    peaks = peak_local_max(expanded_im, min_distance=7, num_peaks=50)
            elif self.algorithm == 'threed':
                peaks = findpeaks3d(im1)
                peaks = reg_peaks(im1, peaks,thresh=self.reg_peak_dist)
            elif self.algorithm == 'template':
                if not self.algorithm_params['templates_made']:
                    expanded_im = np.repeat(im1, self.im.anisotropy[0], axis=0)
                    expanded_im = np.repeat(expanded_im, self.im.anisotropy[1], axis=1)
                    expanded_im = np.repeat(expanded_im, self.im.anisotropy[2], axis=2)
                    expanded_im *= mask
                    try:
                        peaks = np.rint(self.algorithm_params["template_peaks"]).astype(int)
                    except:
                        peaks = peak_local_max(expanded_im, min_distance=9, num_peaks=50)
                        peaks //= self.im.anisotropy
                    chunks = get_bounded_chunks(data=im1, peaks=peaks, pad=[1, 25, 25])
                    chunk_shapes = [chunk.shape for chunk in chunks]
                    max_chunk_shape = (max([chunk_shape[0] for chunk_shape in chunk_shapes]), max([chunk_shape[1] for chunk_shape in chunk_shapes]), max([chunk_shape[2] for chunk_shape in chunk_shapes]))
                    self.templates = [np.mean(np.array([chunk for chunk in chunks if chunk.shape == max_chunk_shape]), axis=0)]
                    quantiles = self.algorithm_params.get('quantiles', [0.5])
                    rotations = self.algorithm_params.get('rotations', [0])
                    for quantile in quantiles:
                        for rotation in rotations:
                            try:
                                self.templates.append(rotate(np.quantile(chunks, quantile, axis=0), rotation, axes=(-1, -2)))
                            except:
                                pass
                    print("Total number of computed templates: ", len(self.templates))
                    self.algorithm_params['templates_made'] = True
                peaks = None
                for template in self.templates:
                    template_peaks = peak_filter_2(data=im1, params={'template': template, 'threshold': 0.5})
                    if peaks is None:
                        peaks = template_peaks
                    else:
                        peaks = np.concatenate((peaks, template_peaks))
                peak_mask = np.zeros(im1.shape, dtype=bool)
                peak_mask[tuple(peaks.T)] = True
                peak_masked = im1 * peak_mask
                expanded_im = np.repeat(peak_masked, self.im.anisotropy[0], axis=0)
                expanded_im = np.repeat(expanded_im, self.im.anisotropy[1], axis=1)
                expanded_im = np.repeat(expanded_im, self.im.anisotropy[2], axis=2)
                expanded_im *= mask
                peaks = peak_local_max(expanded_im, min_distance=13)
                peaks //= self.im.anisotropy
            elif self.algorithm == 'curated':
                if 0 == i:
                    peaks = np.array(self.algorithm_params['peaks'])
                    self.last_peaks = peaks
                else:
                    peaks = self.last_peaks
            else:
                peaks = findpeaks2d(im1)
                peaks = reg_peaks(im1, peaks,thresh=self.reg_peak_dist)

            if self.register and i!=0:
                _off = phase_cross_correlation(self.last_im1, im1, upsample_factor=100)[0][1:]
                _off = np.insert(_off, 0,0)
                if self.algorithm == 'curated':
                    self.last_peaks -= _off
                self.spool.reel(peaks,self.im.anisotropy, offset=_off)
            else:
                self.spool.reel(peaks,self.im.anisotropy)

            self.last_im1 = np.copy(im1)

            if not self.suppress_output:
                print('\r' + 'Frames Processed: ' + str(i+1)+'/'+str(self.t), sep='', end='', flush=True)
        print('\nInfilling...')
        self.spool.infill()
        
        
        
        imshape = tuple([len(self.frames)]) + self.im.sizexy
        def collided(positions, imshape, window = 3):
            for i in [1,2]:
                if np.sum(np.rint(positions[:,i]).astype(int) < window) != 0:
                    return True

                if np.sum(imshape[i] - np.rint(positions[:,i]).astype(int) < window+1) != 0:
                    return True

            if np.sum(np.rint(positions[:,0]).astype(int)<0) != 0 or np.sum(np.rint(positions[:,0]).astype(int) > imshape[0]-1) != 0:
                return True

            #if positions[0] < 0 or int(positions[0]) == imshape[0]:
            #    return True
            return False



            #for i in range(1,len(p)):
            #    if p[i] < window:
            #        return True
            #    elif s[i]-p[i] < window:
            #       return True

        print('Removing bad threads')
        self.remove_bad_threads()
        destroy = []
        _a = len(self.spool.threads)
        for i in range(_a):
            if collided(self.spool.threads[i].positions,imshape):
                destroy.append(i)
            print('\r' + 'Blob Threads Checked: ' + str(i+1)+'/'+str(_a), sep='', end='', flush=True)
        print('\n')
        destroy = sorted(list(set(destroy)), reverse = True)
        if destroy:
            for item in destroy:
                self.spool.threads.pop(item)

        self._merge_within_z()
        self.spool.make_allthreads()
        return self.spool

    def remove_bad_threads(self):
        if self.t > 1:
            d = np.zeros(len(self.spool.threads))
            zd = np.zeros(len(self.spool.threads))
            orig = len(self.spool.threads)
            for i in range(len(self.spool.threads)):
                dvec = np.diff(self.spool.threads[i].positions, axis = 0)
                d[i] = np.abs(dvec).max()
                zd[i] = np.abs(dvec[0:self.t,0]).max()

            ans = d > self.remove_blobs_dist
            ans = ans + (zd > self.remove_blobs_dist/self.im.anisotropy[0])
        
            throw_ndx = np.where(ans)[0]
            throw_ndx = list(throw_ndx)

            throw_ndx.sort(reverse = True)

            for ndx in throw_ndx:
                self.spool.threads.pop(int(ndx))
            print('Blob threads removed: ' + str(len(throw_ndx)) + '/' + str(orig))


    def _merge_within_z(self):
        # sort threads by z
        tbyz = self._threads_by_z()
        # calculate distance matrix list
        dmatlist = self._calc_dist_mat_list(self._threads_by_z())

        for i in range(len(dmatlist)):
            if len(dmatlist[i]) > 1:
                sort_mat,b,c = self._compute_serial_matrix(dmatlist[i])
            
    def _calc_dist_mat(self, indices):
        """
        Calculates distance matrix among threads with indices specified

        Arguments:
            indices : list of ints
                list of indices corresponding to which threads are present for the distance matrix calculation
        """
        
        # initialize distance matrix
        dmat = np.zeros((len(indices), len(indices)))

        # calculate dmat, non-diagonals only
        for i in range(len(indices)):
            for j in range(i+1, len(indices)):
                pos1 = self.spool.threads[indices[i]].positions
                pos2 = self.spool.threads[indices[j]].positions

                dmat[i,j] = np.linalg.norm(pos1 - pos2, axis = 1).mean()
        dmat = dmat + dmat.T

        return dmat


    def _calc_dist_mat_list(self, indices: list) -> list:
        """
        Calculates list of distance matrices 

        Arguments:
            e : extractor
                extractor object
            indices : list of list of ints
                list of list of indices made by threads_by_z
        """
        # initialize dmat list
        dmatlist = []

        # iterate over z planes
        for i in range(len(indices)):
            dmat = self._calc_dist_mat(indices[i])
            dmatlist.append(dmat)
        return dmatlist


    def _threads_by_z(self):
        """
        Organizes thread indices by z plane

        Arguments:
            e : extractor
                extractor object
        """

        # make object to store thread indices corresponding to z plane
        threads_by_z = [[] for i in self.frames]


        # iterate over threads, append index to threads_by_z
        for i in range(len(self.spool.threads)):
            z = round(self.spool.threads[i].positions[0,0])
            # ndx = np.where(np.array(self.frames) == z)[0][0]

            threads_by_z[z].append(i)

        return threads_by_z


    def _seriation(self, Z,N,cur_index):
        '''
            input:
                - Z is a hierarchical tree (dendrogram)
                - N is the number of points given to the clustering process
                - cur_index is the position in the tree for the recursive traversal
            output:
                - order implied by the hierarchical tree Z
                
            seriation computes the order implied by a hierarchical tree (dendrogram)
        '''
        if cur_index < N:
            return [cur_index]
        else:
            left = int(Z[cur_index-N,0])
            right = int(Z[cur_index-N,1])
            return (self._seriation(Z,N,left) + self._seriation(Z,N,right))
        
    def _compute_serial_matrix(self, dist_mat,method="ward"):
        '''
        input:
            - dist_mat is a distance matrix
            - method = ["ward","single","average","complete"]
        output:
            - seriated_dist is the input dist_mat,
              but with re-ordered rows and columns
              according to the seriation, i.e. the
              order implied by the hierarchical tree
            - res_order is the order implied by
              the hierarhical tree
            - res_linkage is the hierarhical tree (dendrogram)
        
        compute_serial_matrix transforms a distance matrix into 
        a sorted distance matrix according to the order implied 
        by the hierarchical tree (dendrogram)
        '''
        N = len(dist_mat)
        flat_dist_mat = squareform(dist_mat)
        res_linkage = linkage(flat_dist_mat, method=method,preserve_input=True)
        res_order = self._seriation(res_linkage, N, N + N-2)
        seriated_dist = np.zeros((N,N))
        a,b = np.triu_indices(N,k=1)
        seriated_dist[a,b] = dist_mat[ [res_order[i] for i in a], [res_order[j] for j in b]]
        seriated_dist[b,a] = seriated_dist[a,b]
        
        return seriated_dist, res_order, res_linkage

