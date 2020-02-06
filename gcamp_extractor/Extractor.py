import numpy as np
#from pycpd import deformable_registration 
import pdb
import time
import scipy.spatial
import copy
from .Threads import *
import pickle
import os
from .multifiletiff import *
from .segfunctions import *
def mkdir(path):
    try: os.mkdir(path)
    except: pass
import glob
import json
import imreg_dft as ird

default_arguments = {
    'root':'/Users/stevenban/Documents/Data/20190917/binned',
    'numz':20,
    'frames':[0,1,2,3,4,5,6,7,8,9,10,11],
    'offset':0,
    't':999,
    'gaussian':(25,4,3,1),
    'quantile':0.99,
    'reg_peak_dist':10,
    'anisotropy':(7,1,1),
    'blob_merge_dist_thresh':6.8,
    'mip_movie':True,
    'marker_movie':True,
    'infill':True,
    'suppress_output':False,
    'regen':False,
}

def default_quant_function(im, positions):
    """
    Default quantification function, to be used in conjunction with the Extractor class. Takes the 10 brightest pixels in a 6x6 square around the specified position 

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
                z.append(int(center[0]))
                y.append(int(center[1] + j))
                x.append(int(center[2] + i))
        masked = im[z,y,x]
        masked.sort()
        timeseries.append(np.mean(masked[-10:]))
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
        
        
    paramsf = folders[0]+'params.json'
    mftf = folders[0]+'mft.obj'
    threadf = folders[0]+'threads.obj'
    timef = folders[0]+'timeseries.txt'

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

    def __init__(self,*args, **kwargs):

        ### Specifying all parameters, and filling in initial values if they weren't passed
        if len(args)!=0:
            self.root = args[0]
            kwargs['root'] = args[0]
        elif kwargs.get('root'):
            self.root = kwargs['root']
        else:
            print('did not pass root directory')
            return 0
        #self.root = kwargs['root']
        if self.root[-1] != '/':
            self.root += '/'
        try:self.numz = kwargs['numz']
        except:self.numz = 10
        try:self.frames= kwargs['frames']
        except:self.frames = list(range(self.numz))
        try:self.offset= kwargs['offset']
        except:self.offset = 0
        try:self.t = kwargs['t']
        except:self.t = 0


        try:self.gaussian= kwargs['gaussian']
        except:self.gaussian = (25,4,3,1)
        try:self.quantile= kwargs['quantile']
        except:self.quantile = 0.99
        try:self.reg_peak_dist= kwargs['reg_peak_dist']
        except:self.reg_peak_dist = 40
        try:self.anisotropy= kwargs['anisotropy']
        except:self.anisotropy = (6,1,1)
        try:self.blob_merge_dist_thresh= kwargs['blob_merge_dist_thresh']
        except:self.blob_merge_dist_thresh = 6
        try:self.mip_movie= kwargs['mip_movie']
        except:self.mip_movie = True
        try:self.marker_movie= kwargs['marker_movie']
        except:self.marker_movie = True
        try:self.suppress_output= kwargs['suppress_output']
        except:self.suppress_output = False
        try:self.incomplete = kwargs['incomplete']
        except:self.incomplete = False 
        try: self.register = kwargs['register_frames']
        except: self.register = False
        try: self.predict = kwargs['predict']
        except: self.predict = True 


        self.threed = kwargs.get('3d')
        mkdir(self.root+'extractor-objects')
        
        _regen_mft = kwargs.get('regen_mft')
        self.im = MultiFileTiff(self.root, offset=self.offset, numz=self.numz, frames=self.frames, regen=_regen_mft)
        self.im.save()
        #self.im.set_frames(self.frames)
        #e.imself.im.numz = self.numz
        self.im.t = 0
        

        if self.t==0 or self.t>(self.im.numframes-self.offset)//self.im.numz:
            self.t=(self.im.numframes-self.offset)//self.im.numz

        self.spool = Spool(self.blob_merge_dist_thresh, max_t = self.t,predict = self.predict)
        self.timeseries = []

        if kwargs.get('regen'):
            pass
        else:
            mkdir(self.root+'extractor-objects')
            kwargs['regen']=True
            with open(self.root + 'extractor-objects/params.json', 'w') as json_file:
                json.dump(kwargs, json_file)

    def calc_blob_threads(self):
        """
        calculates blob threads

        Parameters
        ----------
        infill : bool
            whether or not to infill the 
        """
        for i in range(self.t):
            im1 = self.im.get_t()
            im1 = medFilter2d(im1)
            im1 = gaussian3d(im1,self.gaussian)
            if self.threed:
                peaks = findpeaks3d(np.array(im1 * np.array(im1 > np.quantile(im1,self.quantile))))
            else:
                peaks = findpeaks2d(np.array(im1 * np.array(im1 > np.quantile(im1,self.quantile))))
            peaks = reg_peaks(im1, peaks,thresh=self.reg_peak_dist)

            if self.register and i!=0:
                _off = ird.translation(self.im.get_tbyf(i-1,self.frames[int(len(self.frames)/2)]), im1[int(len(self.frames)/2)])['tvec']

                _off = np.insert(_off, 0,0)
                #peaks = peaks+ _off
                #print(_off)
            #print(peaks)
                self.spool.reel(peaks,self.anisotropy, offset=_off)
            else:
                self.spool.reel(peaks,self.anisotropy)
            
            if not self.suppress_output:
                print('\r' + 'Frames Processed: ' + str(i+1)+'/'+str(self.t), sep='', end='', flush=True)
        print('\nInfilling...')
        self.spool.infill()
        
        
        
        imshape = tuple([len(self.frames)]) + self.im.sizexy
        def collided(positions, imshape, window = 3):
            for i in [1,2]:
                if np.sum(positions[:,i].astype(int) < window) != 0:
                    return True

                if np.sum(imshape[i] - positions[:,i].astype(int) < window+1) != 0:
                    return True

            if np.sum(positions[:,0].astype(int)<0) != 0 or np.sum(positions[:,0].astype(int) > imshape[0]-1) != 0:
                return True

            #if positions[0] < 0 or int(positions[0]) == imshape[0]:
            #    return True
            return False



            #for i in range(1,len(p)):
            #    if p[i] < window:
            #        return True
            #    elif s[i]-p[i] < window:
             #       return True

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

        
        print('Saving blob timeseries as numpy object...')
        mkdir(self.root+'extractor-objects')
        file_pi = open(self.root + 'extractor-objects/threads.obj', 'wb') 
        pickle.dump(self.spool, file_pi)
        file_pi.close()

    def quantify(self, quant_function=default_quant_function):
        """
        generates timeseries based on calculated threads. 

        Parameters
        ----------
        quant_function : function
            function that takes in a list of positions/pixel indices, and returns a list of floats that describe neuronal activity. A default quantification function is included. It takes the 10 brightest pixels in a 6x6 square around the position and averages them. 

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
        self.im.t = 0
        self.timeseries = np.zeros((self.t,len(self.spool.threads)))
        for i in range(self.t):
            self.timeseries[i] = quant_function(self.im.get_t(),[self.spool.threads[j].get_position_t(i) for j in range(len(self.spool.threads))])
            if not self.suppress_output:
                print('\r' + 'Frames Processed (Quantification): ' + str(i+1)+'/'+str(self.t), sep='', end='', flush=True)


        mkdir(self.root + 'extractor-objects')
        np.savetxt(self.root+'extractor-objects/timeseries.txt',self.timeseries)
        print('\nSaved timeseries as text file...')
    
    def save_threads(self):
        print('Saving blob threads as pickle object...')
        mkdir(self.root+'extractor-objects')
        file_pi = open(self.root + 'extractor-objects/threads.obj', 'wb') 
        pickle.dump(self.spool, file_pi)
        file_pi.close()


    def save_timeseries(self):
        print('Saving blob threads as pickle object...')
        mkdir(self.root+'extractor-objects')
        file_pi = open(self.root + 'extractor-objects/threads.obj', 'wb') 
        pickle.dump(self.spool, file_pi)
        file_pi.close()

    def save_MIP(self, fname = ''):

        # if passed with no argument, save with default
        if fname == '':
            fname = self.root + "/extractor-objects/MIP.tif"
        
        # if just filename specified, save in extractor objects with the filename 
        elif '/' not in fname and '\\' not in fname:
            fname = self.root + '/extractor-objects/' + fname


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

