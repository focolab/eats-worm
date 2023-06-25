import bz2
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
from improc.segfunctions import peak_local_max as ani_peak_local_max
def mkdir(path):
    try: os.mkdir(path)
    except: pass
import glob
import json
import math
from scipy.ndimage import generate_binary_structure, binary_dilation, rotate
from scipy.spatial.distance import pdist, squareform
import scipy.io as sio
from skimage.feature import peak_local_max
from skimage.filters import rank
from skimage.registration import phase_cross_correlation
from fastcluster import linkage
import scipy.cluster
from threading import Thread, Lock
import improc.NPproc.process.resample as res
import improc.NPproc.process.histogram as hist
import tifffile
import cv2

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

def background_subtraction_quant_function(im, spool, t, frames, quant_radius=3, quant_z_radius=1, quant_voxels=20, background_radius=30, other_pos_radius=None, threads_to_quantify=None, quantified_voxels=None):
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
    intensities = [np.NaN] * len(spool.threads)
    positions = spool.get_positions_t(t, indices=threads_to_quantify)
    positions = np.rint(np.copy(positions)).astype(int)
    max_z = len(frames) # in case of max_x, max_y, max_z, we're using these as indices so don't subtract 1 because slicing is exclusive 
    max_x = im.shape[1]
    max_y = im.shape[2]
    pos_z, pos_x, pos_y = positions.T

    pos_mask = np.zeros(im.shape, dtype=int)
    background_mask = np.zeros(im.shape, dtype=bool)
    collision_mask = np.zeros(im.shape, dtype=bool)

    roi_indices = np.arange(positions.shape[0])
    z_zeros = np.array([0] * len(pos_z))
    z_maxes = np.array([max_z] * len(pos_z))
    x_zeros = np.array([0] * len(pos_x))
    x_maxes = np.array([max_x] * len(pos_x))
    y_zeros = np.array([0] * len(pos_y))
    y_maxes = np.array([max_y] * len(pos_y))
    z_starts = np.maximum(z_zeros, pos_z - quant_z_radius)
    z_stops = np.minimum(z_maxes, pos_z + quant_z_radius + 1)

    quant_x_starts = np.maximum(x_zeros, pos_x - quant_radius)
    quant_x_stops = np.minimum(x_maxes, pos_x + quant_radius + 1)
    quant_y_starts = np.maximum(y_zeros, pos_y - quant_radius)
    quant_y_stops = np.minimum(y_maxes, pos_y + quant_radius + 1)
    for i in range(len(z_starts)):
        pos_mask[z_starts[i]:z_stops[i], quant_x_starts[i]:quant_x_stops[i], quant_y_starts[i]:quant_y_stops[i]] += 1

    background_x_starts = np.maximum(x_zeros, pos_x - background_radius)
    background_x_stops = np.minimum(x_maxes, pos_x + background_radius + 1)
    background_y_starts = np.maximum(y_zeros, pos_y - background_radius)
    background_y_stops = np.minimum(y_maxes, pos_y + background_radius + 1)

    if not other_pos_radius or other_pos_radius == quant_radius:
        background_mask[pos_mask == 0] = True
        collision_mask[pos_mask > 1] = True
    else:
        background_mask[:] = True
        collision_mask = collision_mask.astype(int)
        other_x_starts = np.maximum(x_zeros, pos_x - other_pos_radius)
        other_x_stops = np.minimum(x_maxes, pos_x + other_pos_radius + 1)
        other_y_starts = np.maximum(y_zeros, pos_y - other_pos_radius)
        other_y_stops = np.minimum(y_maxes, pos_y + other_pos_radius + 1)
        for i in range(len(z_starts)):
            background_mask[z_starts[i]:z_stops[i], other_x_starts[i]:other_x_stops[i], other_y_starts[i]:other_y_stops[i]] = False
            collision_mask[z_starts[i]:z_stops[i], other_x_starts[i]:other_x_stops[i], other_y_starts[i]:other_y_stops[i]] += 1
        collision_mask = (collision_mask > 1).astype(bool)

    pos_mask = pos_mask.astype(bool)
    pos_mask *= ~collision_mask

    roi_mask = np.zeros(im.shape, dtype=bool)
    roi_background_mask = np.zeros(im.shape, dtype=bool)
    for i in range(positions.shape[0]):
        roi_mask[z_starts[i]:z_stops[i], quant_x_starts[i]:quant_x_stops[i], quant_y_starts[i]:quant_y_stops[i]] = True
        roi_mask *= ~collision_mask
        roi_intensities = im[roi_mask]
        num_voxels = np.minimum(quant_voxels, len(roi_intensities) - 1)
        top_rois = np.argpartition(-roi_intensities, num_voxels)[:num_voxels]
        top_roi_intensities = roi_intensities[top_rois]
        roi_background_mask[z_starts[i]:z_stops[i], background_x_starts[i]:background_x_stops[i], background_y_starts[i]:background_y_stops[i]] = True
        roi_background_mask *= background_mask
        background_intensities = im[roi_background_mask]
        roi_intensity = np.mean(top_roi_intensities)
        background_intensity = np.mean(background_intensities)
        intensities[threads_to_quantify[i]] = (roi_intensity - background_intensity) / background_intensity
        if quantified_voxels is not None:
            roi_foreground_voxels = np.array(np.where(roi_mask)).T
            quantified_foreground_voxels = roi_foreground_voxels[top_rois]
            quantified_voxels[threads_to_quantify[i]][t] = quantified_foreground_voxels
        roi_mask[:] = False
        roi_background_mask[:] = False
    return intensities

# ported from wb-matlab
def exp_curve_bleach_correction(timeseries):
    baseline_method = 'min'
    start_frames = np.rint(min(np.floor(0.1 * timeseries.shape[0]), 100)).astype(int)
    end_frames = np.rint(min(np.floor(0.1 * timeseries.shape[0]), 100)).astype(int)
    start_range = [0, start_frames]
    end_range = [timeseries.shape[0] - end_frames, timeseries.shape[0]]

    inp = np.arange(timeseries.shape[0])
    ti = np.mean(start_range)
    tf = np.mean(end_range)
    tau = 1. / np.linspace(1/(2*(tf-ti)), 1/(.1*(tf-ti)), num=200)

    bleach_curves = np.empty(timeseries.shape)
    traces_bc = np.empty(timeseries.shape)
    tau_best = np.empty((timeseries.shape[1]))

    for i in range(timeseries.shape[1]):
        yf = np.min(timeseries[:, i])
        if baseline_method == 'min':
            ti = np.argmin(timeseries[start_range[0]:start_range[1], i])
            yi = timeseries[start_range[0]:start_range[1], i][ti]
        else:
            yi = np.mean(timeseries[start_range[0]:start_range[1], i])

        intersect = 1
        j = 0
        while intersect == 1 and j < 100:
            ex = np.exp(-(tf - ti) / tau[j])
            k = (yf - yi * ex) / (1 - ex)
            bleach_curves[:, i] = (yi - k) * np.exp(-inp/tau[j]) + k

            tr=[start_range[1] + 200, end_range[0] - 1000]
            if not np.any(timeseries[tr[0]:tr[1], i] - bleach_curves[tr[0]:tr[1], i] < 0):
                intersect=0
            j += 1

        tau_best[i] = tau[j - 1]

        bleach_curves[:, i] = (yi - k) * np.exp(-inp / tau_best[i]) + k
        traces_bc[:, i] = timeseries[:, i] - bleach_curves[:, i]

    tau = tau_best
    return traces_bc

def quantify(mft=None, extractor=None, quant_function=background_subtraction_quant_function, bleach_correction=None, curation_filter='not trashed', suppress_output=False, start_t=0, parallel_threads=1, save_quantification_voxels=False, **kwargs):
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
    spool = extractor.spool
    num_threads = len(spool.threads)
    num_t = spool.t

    threads_to_quantify = list(range(len(spool.threads)))
    if curation_filter != 'all':
        try:
            output_dir = mft.output_dir.replace('//', '/')
            with open(os.path.join(output_dir, 'curate.json')) as f:
                curated_json = json.load(f)
                if curation_filter == 'not trashed':
                    trashed = [int(roi) for roi in curated_json.keys() if curated_json[roi] == 'trash']
                    threads_to_quantify = [roi for roi in threads_to_quantify if roi not in trashed]
                elif curation_filter == 'kept':
                    kept = [int(roi) for roi in curated_json.keys() if curated_json[roi] == 'keep']
                    threads_to_quantify = kept
        except Exception as e:
            print("Failed to parse curator log. Quantifying all threads. Encountered exception:")
            print(e)

    timeseries = np.empty((num_t,num_threads))
    timeseries[:] = np.NaN
    quantified_voxels = {i: {} for i in range(num_threads)}
    quant_lock = Lock()
    processed_counter = [0]
    def quantify_in_parallel_thread(start, stop):
        thread_timeseries = np.empty((stop - start, num_threads))
        thread_timeseries[:] = np.NaN
        thread_quantified_voxels = None
        if save_quantification_voxels:
            thread_quantified_voxels = {i: {} for i in range(num_threads)}
        for t in range(stop - start):
            quant_lock.acquire()
            im = mft.get_t(start + t)
            quant_lock.release()
            thread_timeseries[t] = quant_function(im, spool, start+t, mft.frames, threads_to_quantify=threads_to_quantify, quantified_voxels=thread_quantified_voxels, **kwargs)
            quant_lock.acquire()
            processed_counter[0] += 1
            quant_lock.release()
            print('\r' + 'Frames Processed (Quantification): ' + str(processed_counter[0])+'/'+str(num_t), sep='', end='', flush=True)
        quant_lock.acquire()
        timeseries[start:stop] = thread_timeseries
        if save_quantification_voxels:
            for key in thread_quantified_voxels.keys():
                quantified_voxels[key].update(thread_quantified_voxels[key])
        quant_lock.release()

    threads = []
    t_per_thread = math.ceil(num_t / parallel_threads)
    start = 0
    while start < num_t:
        stop = min(start + t_per_thread, num_t)
        threads.append(Thread(target=quantify_in_parallel_thread, args=(start, stop)))
        start += t_per_thread
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    print('\r')
    if bleach_correction is not None:
        print('Performing bleach correction using exponential curve relaxation method.')
        timeseries = exp_curve_bleach_correction(timeseries)

    if save_quantification_voxels:
        if not hasattr(extractor, 'curator_layers'):
            extractor.curator_layers = {}
        extractor.curator_layers["quantified_roi_voxels"] = {'data': quantified_voxels, 'type': 'points'}

    return timeseries


def load_extractor(path, root_override=None, output_override=None):
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
    curator_layersf = folders[0]+'/curator_layers.pkl.bz2'

    with open(paramsf) as f:
        params = json.load(f)
        if root_override:
            params['root'] = root_override
        if output_override:
            params['output_dir'] = output_override
        else:
            params['output_dir'] = path
    params['regen_mft'] = False
    e = Extractor(**params)

    # handle curators which were pickled before the repo name was changed to eats-worm (from https://stackoverflow.com/questions/40914066/unpickling-objects-after-renaming-a-module)
    class BackwardsCompatibleUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if 'gcamp_extractor' in module:
                module = module.replace('gcamp_extractor', 'eats_worm')
            return super().find_class(module, name)

    try:
        with open(threadf,'rb') as f:
            thread = BackwardsCompatibleUnpickler(f).load()
        e.spool = thread
    except:
        pass

    try:
        time = np.genfromtxt(timef)
        e.timeseries = time
    except:
        pass

    try:
        with bz2.open(curator_layersf, 'r') as f:
            curator_layers = BackwardsCompatibleUnpickler(f).load()
            e.curator_layers = curator_layers
    except:
        pass

    return e


def render_gaussian_2d(blob_diameter, sigma):
    """
    :param im_width:
    :param sigma:
    :return:
    """

    gaussian = np.zeros((blob_diameter, blob_diameter), dtype=np.float32)

    # gaussian filter
    for i in range(int(-(blob_diameter - 1) / 2), int((blob_diameter + 1) / 2)):
        for j in range(int(-(blob_diameter - 1) / 2), int((blob_diameter + 1) / 2)):
            x0 = int((blob_diameter) / 2)  # center
            y0 = int((blob_diameter) / 2)  # center
            x = i + x0  # row
            y = j + y0  # col
            gaussian[y, x] = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / 2 / sigma / sigma)

    return gaussian


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

        self.output_dir = kwargs.get('output_dir', self.root)
        if not self.output_dir.endswith('extractor-objects') and not self.output_dir.endswith('extractor-objects/'):
            self.output_dir = os.path.join(self.output_dir, 'extractor-objects')
        os.makedirs(self.output_dir, exist_ok=True)
        self.numz = kwargs.get('numz', 10)
        self.numc = kwargs.get('numc', 1)
        self.frames= kwargs.get('frames', list(range(self.numz)))
        self.offset = kwargs.get('offset', 0)
        self.anisotropy = kwargs.get('anisotropy', (6,1,1))
        self.mip_movie = kwargs.get('mip_movie', True)
        self.marker_movie = kwargs.get('marker_movie', True)
        _regen_mft = kwargs.get('regen_mft')
        self.processing_params = kwargs.get('processing_params', {"neuroPAL":False})
        self.im = MultiFileTiff(self.root, output_dir=self.output_dir, anisotropy=self.anisotropy, offset=self.offset, numz=self.numz, numc=self.numc, frames=self.frames, regen=_regen_mft)
        self.im.save()
        self.im.t = 0
        self.start_t = kwargs.get('start_t', 0)
        self.end_t = kwargs.get('end_t', kwargs.get('t', 0))
        if self.end_t==0 or self.end_t>self.im.end_t:
            self.end_t = self.im.end_t
        self.blobthreadtracker_params = {k: v for k, v in kwargs.items() if k not in vars(self)}
        self.blobthreadtracker_params.update({'start_t': self.start_t, 'end_t': self.end_t})

        ### Dump a record of input parameters
        if kwargs.get('regen'):
            pass
        else:
            kwargs['regen']=True
            with open(os.path.join(self.output_dir, 'params.json'), 'w') as json_file:
                json.dump(kwargs, json_file)

    def process_im(self):
        """
        Runs basic image pre-processing steps and saves processed image
        
        Image processing is currently only supported for single volume neuroPAL images. Will
        eventually add support for timeseries volumetric data as well.
        
        Histogram matching currently only supports .mat file input for the reference image, 
        the file type used in the original NeuroPAL paper. We will eventually add support 
        for other filetypes as well, with the eventual goal of moving everything to the NWB
        format. 
        """
        if self.processing_params["neuroPAL"]:
            channels = self.processing_params["RGBW_channels"]
            NP_image = np.transpose(self.im.tf[0].asarray())
            print(self.im.tf[0].asarray().shape)
            NP_image = np.squeeze(NP_image[:,:,:,channels])
            NP_image = NP_image.astype('int32')

            print('Preprocessing image')

            if 'histogram_match' in self.processing_params:
                # TODO: make sure reffiles are in tif format
                print('Matching histogram of image to reference')

                A_max = self.processing_params['histogram_match']["A_max"]
                ref_max = self.processing_params['histogram_match']["ref_max"]

                reffile = sio.loadmat(self.processing_params['histogram_match']["im_to_match"])
                refchannels = reffile['prefs']['RGBW'][0][0]-1
                refdata = reffile['data']
                refRGBW = np.squeeze(refdata[:,:,:,refchannels])

                NP_image = hist.match_histogram(NP_image, refRGBW, A_max, ref_max)

                print('Matched histogram of image to reference')

            if 'resample' in self.processing_params:   
                print('Resampling image') 
                new_res = self.processing_params['resample']["new_resolution"]
                old_res = self.processing_params['resample']["old_resolution"]
                NP_image = res.zoom_interpolate(new_res, old_res, NP_image)

                print('Resampled image to: '+str(new_res))

            if 'median' in self.processing_params:
                print('Median filtering image')
                size = self.processing_params.get("median", 3)

                NP_image = medFilter2d(NP_image, size)

            NP_image = NP_image.astype('uint16')
            NP_image = np.transpose(NP_image, (2,3,1,0))

            tifffile.imwrite(self.root+'/processed_data.tif', NP_image, imagej = True)

            print('Saved processed image to output directory')

    def calc_blob_threads(self):
        """peakfinding and tracking"""
        x = BlobThreadTracker(mft=self.im, params=self.blobthreadtracker_params)
        self.spool = x.calc_blob_threads()
        print('Saving blob timeseries as numpy object...')
        self.spool.export(f=os.path.join(self.output_dir, 'threads.obj'))
        if x.curator_layers:
            self.curator_layers = x.curator_layers
            pickle.dump(self.curator_layers, bz2.open(os.path.join(self.output_dir, 'curator_layers.pkl.bz2'), 'wb'))

    def quantify(self, quant_function=background_subtraction_quant_function, bleach_correction=None, curation_filter='all', parallel_threads=1, **kwargs):
        """generates timeseries based on calculated threads"""
        self.timeseries = quantify(mft=self.im, extractor=self, start_t=self.start_t, quant_function=quant_function, bleach_correction=bleach_correction, curation_filter=curation_filter, parallel_threads=parallel_threads, **kwargs)
        self.save_timeseries()
        self.save_dataframe()
        if hasattr(self, 'curator_layers'):
            if self.curator_layers:
                pickle.dump(self.curator_layers, bz2.open(os.path.join(self.output_dir, 'curator_layers.pkl.bz2'), 'wb'))

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
        csv_blobs = os.path.join(self.output_dir, 'df_peaks_detected.csv')
        print('Export blobs: ', csv_blobs)
        blobs = df.loc[df['T']==0]
        newcols = ['X', 'Y', 'Z', 'prov', 'ID']
        blobs = blobs[newcols]
        blobs = blobs.replace('infilled', 'detected')
        blobs = blobs.fillna("")
        blobs = blobs.astype({'X':'int32', 'Y':'int32', 'Z':'int32'})
        blobs['Status'] = 0
        blobs['ID'] = blobs['ID'].astype('string')
        blobs['ID'] = ""
        blobs.to_csv(csv_blobs, na_rep='')

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
        _output = np.zeros(tuple([self.end_t - self.start_t]) + self.im.sizexy, dtype = np.uint16)
        
        with tiff.TiffWriter(fname,bigtiff = True) as tif:
            for i in range(self.start_t, self.end_t):
                tif.save(np.max(self.im.get_t(i), axis = 0))
                print('\r' + 'MIP Frames Saved: ' + str(i+1-self.start_t)+'/'+str(self.end_t - self.start_t), sep='', end='', flush=True)

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
        self.peakfind_channel = params.get('peakfind_channel',0)
        self.blob_merge_dist_thresh = params.get('blob_merge_dist_thresh', 6)
        self.remove_blobs_dist = params.get('remove_blobs_dist', 20)
        self.suppress_output = params.get('suppress_output', False)
        self.register = params.get('register_frames', False)
        self.predict = params.get('predict', True)
        self.algorithm = params.get('algorithm', 'template')
        self.algorithm_params = params.get('algorithm_params', {})
        self.curator_layers = params.get('curator_layers', False)
        try:
            self.algorithm_params['templates_made'] = type(self.algorithm_params['template']) != bool
        except:
            self.algorithm_params['templates_made'] = False

        # self.start_t and self.end_t are time index cutoffs for partial analysis
        self.start_t = params['start_t']
        self.end_t = params['end_t']


    def calc_blob_threads(self):
        """
        calculates blob threads

        returns
        -------
        spool: (Spool)
        """
        self.spool = Spool(self.blob_merge_dist_thresh, max_t=self.end_t - self.start_t, predict=self.predict)

        # handle an ugly workaround for peak_local_max not supporting anisotropy. this stuff is only needed when
        # using skimage or template matching, but putting it here allows us to avoid redoing the matmuls every iteration
        if all(isinstance(dimension, int) for dimension in self.im.anisotropy):
            expanded_shape = tuple([dim_len * ani for dim_len, ani in zip(self.im.get_t(0).shape, self.im.anisotropy)])
            mask = np.zeros(expanded_shape, dtype=np.uint16)
            mask[tuple([np.s_[::ani] for ani in self.im.anisotropy])] = 1

        if 'tmip' in self.algorithm:
            window_size = self.algorithm_params.get('window_size', 10)
            ims = []
            offsets = []
            last_offset = None
            last_im = None
            if self.algorithm.endswith('2d_template') and self.curator_layers:
                self.curator_layers = {'filtered': {'type': 'image', 'data': []}}
            for i in range(self.start_t, self.end_t):
                im1 = self.im.get_t(i, channel = self.peakfind_channel)

                im1 = medFilter2d(im1, self.median)
                if self.gaussian:
                    im1 = gaussian3d(im1, self.gaussian)

                if self.algorithm.startswith('seeded') and i==self.start_t:
                    peaks = np.array(self.algorithm_params['peaks'])
                    self.spool.reel(peaks,self.im.anisotropy)
                    if 'labels' in self.algorithm_params:
                        for thread, label in zip(self.spool.threads, self.algorithm_params['labels']):
                            thread.label = label
                    last_offset = np.array([0, 0, 0])
                    last_im = im1

                else:
                    if self.register and i!=self.start_t:
                        _off = phase_cross_correlation(last_im, im1, upsample_factor=100)[0][1:]
                        _off = np.insert(_off, 0,0)
                        # track offset relative to t=0 so that all tmips are spatially aligned
                        _off += last_offset
                        offsets.append(_off)
                        shift = tuple(np.rint(_off).astype(int))
                        axis = tuple(np.arange(im1.ndim))
                        last_im = np.copy(im1)
                        if np.max(shift) > 0:
                            im1 = np.roll(im1, shift, axis)
                            if shift[0] >= 0:
                                im1[:shift[0], :, :] = 0
                            else:
                                im1[shift[0]:, :, :] = 0
                            if shift[1] >= 0:
                                im1[:, :shift[1], :] = 0
                            else:
                                im1[:, shift[1]:, :] = 0
                            if shift[2] >- 0:
                                im1[:, :, :shift[2]] = 0
                            else:
                                im1[:, :, shift[2]:] = 0
                    else:
                        last_im = np.copy(im1)
                        offsets.append(np.array([0, 0, 0]))
                    last_offset = offsets[-1]
                    ims.append(np.copy(im1))

                    if len(ims) == window_size or i == self.end_t - 1:
                        tmip = np.max(np.array(ims), axis=0)

                        if self.algorithm.endswith('2d_template'):
                            vol_tmip = np.max(np.array(ims), axis=0).astype(np.float32)

                            for z in range(vol_tmip.shape[0]):
                                im_filtered = vol_tmip[z,:,:]
                                fb_threshold_margin = self.algorithm_params.get('fb_threshold_margin', 20)
                                threshold = np.median(im_filtered) + fb_threshold_margin
                                im_filtered = (im_filtered > threshold) * im_filtered
                        
                                gaussian_template_filter = render_gaussian_2d(self.algorithm_params.get('gaussian_diameter', 13), self.algorithm_params.get('gaussian_sigma', 9))
                                im_filtered = cv2.matchTemplate(im_filtered, gaussian_template_filter, cv2.TM_CCOEFF)
                                pad = [int((x-1)/2) for x in gaussian_template_filter.shape]
                                im_filtered = np.pad(im_filtered, tuple(zip(pad, pad)))

                                im_filtered = (im_filtered > threshold) * im_filtered
                                vol_tmip[z,:,:] = im_filtered
                            
                            if self.curator_layers:
                                vol_tmip_mask_as_list = (vol_tmip > 0).tolist()
                                for timepoint in range(len(ims)):
                                    self.curator_layers['filtered']['data'].append(vol_tmip_mask_as_list)

                            # reset local var
                            tmip = vol_tmip

                            # get peaks, min distance is in 3 dimensions
                            if all(isinstance(dimension, int) for dimension in self.im.anisotropy):
                                expanded_im = np.repeat(tmip, self.im.anisotropy[0], axis=0)
                                expanded_im = np.repeat(expanded_im, self.im.anisotropy[1], axis=1)
                                expanded_im = np.repeat(expanded_im, self.im.anisotropy[2], axis=2)
                                expanded_im *= mask
                                peaks = peak_local_max(expanded_im, min_distance=self.algorithm_params.get('min_distance', 9), exclude_border=self.algorithm_params.get('exclude_border', True)).astype(float)
                                peaks /= self.im.anisotropy
                            else:
                                peaks = ani_peak_local_max(tmip, min_distance=self.algorithm_params.get('min_distance', 9), exclude_border=self.algorithm_params.get('exclude_border', True), anisotropy=self.im.anisotropy).astype(float)
                        else:
                            tmip = np.max(np.array(ims), axis=0)
                            expanded_im = np.repeat(tmip, self.im.anisotropy[0], axis=0)
                            expanded_im = np.repeat(expanded_im, self.im.anisotropy[1], axis=1)
                            expanded_im = np.repeat(expanded_im, self.im.anisotropy[2], axis=2)
                            expanded_im *= mask
                            peaks = peak_local_max(expanded_im, min_distance=self.algorithm_params.get('min_distance', 9), num_peaks=self.algorithm_params.get('num_peaks', 50)).astype(float)
                            peaks /= self.im.anisotropy

                        self.spool.reel(peaks - last_offset, self.im.anisotropy, delta_t=len(ims))
                        ims = []
                        offsets = []

                if not self.suppress_output:
                    print('\r' + 'Frames Processed: ' + str(i+1-self.start_t)+'/'+str(self.end_t - self.start_t), sep='', end='', flush=True)


        else:
            for i in range(self.start_t, self.end_t):
                im1 = self.im.get_t(i, channel = self.peakfind_channel)
                im1 = medFilter2d(im1, self.median)
                if self.gaussian:
                    im1 = gaussian3d(im1, self.gaussian)
                im1 = np.array(im1 * np.array(im1 > np.quantile(im1, self.quantile)))
                if self.algorithm == 'skimage':
                    expanded_im = np.repeat(im1, self.im.anisotropy[0], axis=0)
                    expanded_im = np.repeat(expanded_im, self.im.anisotropy[1], axis=1)
                    expanded_im = np.repeat(expanded_im, self.im.anisotropy[2], axis=2)
                    expanded_im *= mask
                    peaks = peak_local_max(expanded_im, min_distance=self.algorithm_params.get('min_distance', 9), num_peaks=self.algorithm_params.get('num_peaks', 50))
                    peaks //= self.im.anisotropy
                    min_filter_size = self.algorithm_params.get('min_filter', False)
                    if min_filter_size:
                        min_filtered = rank.minimum(im1.astype(bool), np.ones((1, min_filter_size, min_filter_size)))
                        peaks_mask = np.zeros(im1.shape, dtype=bool)
                        peaks_mask[tuple(peaks.T)] = True
                        peaks = np.array(np.nonzero(min_filtered * peaks_mask)).T
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
                            peaks = peak_local_max(expanded_im, min_distance=self.algorithm_params.get('min_distance', 9), num_peaks=self.algorithm_params.get('num_peaks', 50))
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
                elif self.algorithm == 'seeded_skimage':
                    if 0 == i:
                        peaks = np.array(self.algorithm_params['peaks'])
                    else:
                        expanded_im = np.repeat(im1, self.im.anisotropy[0], axis=0)
                        expanded_im = np.repeat(expanded_im, self.im.anisotropy[1], axis=1)
                        expanded_im = np.repeat(expanded_im, self.im.anisotropy[2], axis=2)
                        expanded_im *= mask
                        peaks = peak_local_max(expanded_im, min_distance=self.algorithm_params.get('min_distance', 9), num_peaks=self.algorithm_params.get('num_peaks', 50))
                        peaks //= self.im.anisotropy
                else:
                    peaks = findpeaks2d(im1)
                    peaks = reg_peaks(im1, peaks,thresh=self.reg_peak_dist)

                if self.register and i!=self.start_t:
                    _off = phase_cross_correlation(self.last_im1, im1, upsample_factor=100)[0][1:]
                    _off = np.insert(_off, 0,0)
                    if self.algorithm == 'curated':
                        self.last_peaks -= _off
                        self.spool.reel(peaks,self.im.anisotropy, offset=_off)
                else:
                    self.spool.reel(peaks,self.im.anisotropy)
                    if self.algorithm.startswith('seeded'):
                        if 'labels' in self.algorithm_params:
                            for thread, label in zip(self.spool.threads, self.algorithm_params['labels']):
                                thread.label = label

                self.last_im1 = np.copy(im1)

                if not self.suppress_output:
                    print('\r' + 'Frames Processed: ' + str(i+1-self.start_t)+'/'+str(self.end_t - self.start_t), sep='', end='', flush=True)

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
                if self.spool.threads[i].label:
                    print("removing thread for neuron", self.spool.threads[i].label, "due to collision with edge of imaging window.")
                destroy.append(i)
            print('\r' + 'Blob Threads Checked: ' + str(i+1)+'/'+str(_a), sep='', end='', flush=True)
        print('\n')
        destroy = sorted(list(set(destroy)), reverse = True)
        if destroy:
            for item in destroy:
                self.spool.threads.pop(item)

        self._merge_within_z()
        self.spool.make_allthreads()
        self.spool.manage_collisions(method=self.algorithm_params.get('manage_collisions', None), anisotropy=self.im.anisotropy)
        return self.spool

    def remove_bad_threads(self):
        if self.end_t > 1:
            d = np.zeros(len(self.spool.threads))
            zd = np.zeros(len(self.spool.threads))
            orig = len(self.spool.threads)
            for i in range(len(self.spool.threads)):
                dvec = np.diff(self.spool.threads[i].positions, axis = 0)
                d[i] = np.abs(dvec).max()
                zd[i] = np.abs(dvec[0:,0]).max()

            ans = d > self.remove_blobs_dist
            ans = ans + (zd > self.remove_blobs_dist/self.im.anisotropy[0])
        
            throw_ndx = np.where(ans)[0]
            throw_ndx = list(throw_ndx)

            throw_ndx.sort(reverse = True)

            for ndx in throw_ndx:
                if self.spool.threads[ndx].label:
                    print("removing thread for neuron", self.spool.threads[ndx].label, "due to illegal inter-frame movement.")
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
            left = round(Z[cur_index-N,0])
            right = round(Z[cur_index-N,1])
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

