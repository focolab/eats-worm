import numpy as np
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
from scipy.spatial.distance import pdist, squareform
from skimage.feature import peak_local_max
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
        timeseries[i] = quant_function(mft.get_t(),[spool.threads[j].get_position_t(i) for j in range(len(spool.threads))])

        if not suppress_output:
            print('\r' + 'Frames Processed (Quantification): ' + str(i+1)+'/'+str(num_t), sep='', end='', flush=True)

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

class BlobThreadTracker_alpha():
    """peakfinding and tracking

    The aim of this class is to have the current peakfinding and tracking
    scheme(s) implemented in a self contained manner, so that Extractor can
    handle alternatives without growing in complexity.

    """
    def __init__(self, mft=None, t_max=None, kwargs=None):
        """

        parameters:
        -----------
        mft: (MultiFileTiff)
        t_max: (int)
            Analyze only the first `t_max` volumes
        kwargs: (dict)
            ALL of the parameters specific to peakfinding, tracking etc
        """
        ## datasource
        self.im = mft

        ## parameters
        self.t = t_max
        self.frames = self.im.frames
        self.output_dir = kwargs['output_dir']

        ## TODO: tighten this up
        try:self.gaussian= kwargs['gaussian']
        except:self.gaussian = (25,4,3,1)
        try:self.median= kwargs['median']
        except:self.median = 3
        try:self.quantile= kwargs['quantile']
        except:self.quantile = 0.99
        try:self.reg_peak_dist= kwargs['reg_peak_dist']
        except:self.reg_peak_dist = 40
        try:self.anisotropy= kwargs['anisotropy']
        except:self.anisotropy = (6,1,1)
        try:self.blob_merge_dist_thresh= kwargs['blob_merge_dist_thresh']
        except:self.blob_merge_dist_thresh = 6
        try:self.remove_blobs_dist= kwargs['remove_blobs_dist']
        except:self.remove_blobs_dist = 20
        try:self.suppress_output= kwargs['suppress_output']
        except:self.suppress_output = False
        try:self.incomplete = kwargs['incomplete']
        except:self.incomplete = False
        try: self.register = kwargs['register_frames']
        except: self.register = False
        try: self.predict = kwargs['predict']
        except: self.predict = True
        try: self.skimage = kwargs['skimage']
        except: self.skimage = False
        try:
            self.template = kwargs['template']
            self.template_made = type(self.template) != bool
        except:
            self.template = False
            self.template_made = False
        self.threed = kwargs.get('3d')


        self.spool = Spool(self.blob_merge_dist_thresh, max_t=self.t, predict=self.predict)

    def calc_blob_threads(self):
        """
        calculates blob threads

        returns
        -------
        spool: (Spool)
        """

        for i in range(self.t):
            im1 = self.im.get_t()
            im1 = medFilter2d(im1, self.median)
            im1 = gaussian3d(im1,self.gaussian)
            im1 = np.array(im1 * np.array(im1 > np.quantile(im1,self.quantile)))
            if self.skimage:
                expanded_im = np.repeat(im1, self.anisotropy[0], axis=0)
                expanded_im = np.repeat(expanded_im, self.anisotropy[1], axis=1)
                expanded_im = np.repeat(expanded_im, self.anisotropy[2], axis=2)
                try:
                    peaks = peak_local_max(expanded_im, min_distance=self.skimage[1], num_peaks=self.skimage[0])
                    peaks //= self.anisotropy
                except:
                    print("No peak_local_max params supplied; falling back to default inference.")
                    peaks = peak_local_max(expanded_im, min_distance=7, num_peaks=50)
            elif self.threed:
                peaks = findpeaks3d(im1)
                peaks = reg_peaks(im1, peaks,thresh=self.reg_peak_dist)
            elif self.template:
                if not self.template_made:
                    expanded_im = np.repeat(im1, self.anisotropy[0], axis=0)
                    expanded_im = np.repeat(expanded_im, self.anisotropy[1], axis=1)
                    expanded_im = np.repeat(expanded_im, self.anisotropy[2], axis=2)
                    peaks = peak_local_max(expanded_im, min_distance=11, num_peaks=45)
                    peaks //= self.anisotropy
                    avg_3d_chunk, blobs = peakfinder(data=im1, peaks=peaks, pad=[15//dim for dim in self.anisotropy])
                    self.template = BlobTemplate(data=avg_3d_chunk, scale=self.anisotropy, blobs='blobs')
                    self.template_made = True
                peaks = peak_filter_2(data=im1,params={'template': self.template.data, 'threshold': 0.7})
            else:
                peaks = findpeaks2d(im1)
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
        print('Saving blob timeseries as numpy object...')
        mkdir(self.output_dir+'/extractor-objects')
        file_pi = open(self.output_dir + '/extractor-objects/threads.obj', 'wb')
        pickle.dump(self.spool, file_pi)
        file_pi.close()

        return self.spool

    def remove_bad_threads(self):
        d = np.zeros(len(self.spool.threads))
        zd = np.zeros(len(self.spool.threads))
        orig = len(self.spool.threads)
        for i in range(len(self.spool.threads)):
            dvec = np.diff(self.spool.threads[i].positions, axis = 0)
            d[i] = np.abs(dvec).max()
            zd[i] = np.abs(dvec[0:self.t,0]).max()

        ans = d > self.remove_blobs_dist
        ans = ans + (zd > self.remove_blobs_dist/2.5)
    
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
            z = int(self.spool.threads[i].positions[0,0])
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






class ExtractorBFD:
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
        try:
            self.output_dir = kwargs['output_dir']
            if self.output_dir[-1] != '/':
                self.output_dir += '/'
        except:
            self.output_dir = self.root
        try:self.numz = kwargs['numz']
        except:self.numz = 10
        try:self.numc = kwargs['numc']
        except:self.numc = 1
        try:self.frames= kwargs['frames']
        except:self.frames = list(range(self.numz))
        try:self.offset= kwargs['offset']
        except:self.offset = 0
        try:self.t = kwargs['t']
        except:self.t = 0


        try:self.gaussian= kwargs['gaussian']
        except:self.gaussian = (25,4,3,1)
        try:self.median= kwargs['median']
        except:self.median = 3
        try:self.quantile= kwargs['quantile']
        except:self.quantile = 0.99
        try:self.reg_peak_dist= kwargs['reg_peak_dist']
        except:self.reg_peak_dist = 40
        try:self.anisotropy= kwargs['anisotropy']
        except:self.anisotropy = (6,1,1)
        try:self.blob_merge_dist_thresh= kwargs['blob_merge_dist_thresh']
        except:self.blob_merge_dist_thresh = 6
        try:self.remove_blobs_dist= kwargs['remove_blobs_dist']
        except:self.remove_blobs_dist = 20
        try:self.suppress_output= kwargs['suppress_output']
        except:self.suppress_output = False
        try:self.incomplete = kwargs['incomplete']
        except:self.incomplete = False 
        try: self.register = kwargs['register_frames']
        except: self.register = False
        try: self.predict = kwargs['predict']
        except: self.predict = True
        try: self.skimage = kwargs['skimage']
        except: self.skimage = False
        try:
            self.template = kwargs['template']
            self.template_made = type(self.template) != bool
        except:
            self.template = False
            self.template_made = False
        self.threed = kwargs.get('3d')

        self.input_kwargs = kwargs


        os.makedirs(self.output_dir+'/extractor-objects', exist_ok=True)
        # mkdir(self.output_dir+'extractor-objects')

        _regen_mft = kwargs.get('regen_mft')
        self.im = MultiFileTiff(self.root, output_dir=self.output_dir, offset=self.offset, numz=self.numz, numc=self.numc, frames=self.frames, regen=_regen_mft)
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
            mkdir(self.output_dir+'/extractor-objects')
            kwargs['regen']=True
            with open(self.output_dir + '/extractor-objects/params.json', 'w') as json_file:
                json.dump(kwargs, json_file)

    def calc_blob_threadsv2(self):
        """to replace calc_blob_threads
        
        Replicates calc_blob_threads but all of the implementation logic is
        outside of Extractor.
        """
        x = BlobThreadTracker_alpha(mft=self.im, t_max=self.t, kwargs=self.input_kwargs)
        self.spool = x.calc_blob_threads()


    def quantify(self, quant_function=default_quant_function):
        """generates timeseries based on calculated threads"""
        self.timeseries = quantify(mft=self.im, spool=self.spool)

        mkdir(self.output_dir + '/extractor-objects')
        np.savetxt(self.output_dir+'extractor-objects/timeseries.txt', self.timeseries)
        print('\nSaved timeseries as text file...')

    def save_threads(self):
        print('Saving blob threads as pickle object...')
        mkdir(self.output_dir+'/extractor-objects')
        file_pi = open(self.output_dir + '/extractor-objects/threads.obj', 'wb')
        pickle.dump(self.spool, file_pi)
        file_pi.close()

    def save_timeseries(self):
        print('Saving blob threads as pickle object...')
        mkdir(self.output_dir+'/extractor-objects')
        file_pi = open(self.output_dir + '/extractor-objects/threads.obj', 'wb')
        pickle.dump(self.spool, file_pi)
        file_pi.close()

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
            fname = self.output_dir + "/extractor-objects/MIP.tif"
        
        # if just filename specified, save in extractor objects with the filename 
        elif '/' not in fname and '\\' not in fname:
            fname = self.output_dir + '/extractor-objects/' + fname


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

    # def remove_bad_threads(self):
    #     d = np.zeros(len(self.spool.threads))
    #     zd = np.zeros(len(self.spool.threads))
    #     orig = len(self.spool.threads)
    #     for i in range(len(self.spool.threads)):
    #         dvec = np.diff(self.spool.threads[i].positions, axis = 0)
    #         d[i] = np.abs(dvec).max()
    #         zd[i] = np.abs(dvec[0:self.t,0]).max()

    #     ans = d > self.remove_blobs_dist
    #     ans = ans + (zd > self.remove_blobs_dist/2.5)
    
    #     throw_ndx = np.where(ans)[0]
    #     throw_ndx = list(throw_ndx)

    #     throw_ndx.sort(reverse = True)

    #     for ndx in throw_ndx:
    #         self.spool.threads.pop(int(ndx))
    #     print('Blob threads removed: ' + str(len(throw_ndx)) + '/' + str(orig))


    # def _merge_within_z(self):
    #     # sort threads by z
    #     tbyz = self._threads_by_z()
    #     # calculate distance matrix list
    #     dmatlist = self._calc_dist_mat_list(self._threads_by_z())

    #     for i in range(len(dmatlist)):
    #         if len(dmatlist[i]) > 1:
    #             sort_mat,b,c = self._compute_serial_matrix(dmatlist[i])
            
    # def _calc_dist_mat(self, indices):
    #     """
    #     Calculates distance matrix among threads with indices specified

    #     Arguments:
    #         indices : list of ints
    #             list of indices corresponding to which threads are present for the distance matrix calculation
    #     """
        
    #     # initialize distance matrix
    #     dmat = np.zeros((len(indices), len(indices)))

    #     # calculate dmat, non-diagonals only
    #     for i in range(len(indices)):
    #         for j in range(i+1, len(indices)):
    #             pos1 = self.spool.threads[indices[i]].positions
    #             pos2 = self.spool.threads[indices[j]].positions

    #             dmat[i,j] = np.linalg.norm(pos1 - pos2, axis = 1).mean()
    #     dmat = dmat + dmat.T

    #     return dmat


    # def _calc_dist_mat_list(self, indices: list) -> list:
    #     """
    #     Calculates list of distance matrices 

    #     Arguments:
    #         e : extractor
    #             extractor object
    #         indices : list of list of ints
    #             list of list of indices made by threads_by_z
    #     """
    #     # initialize dmat list
    #     dmatlist = []

    #     # iterate over z planes
    #     for i in range(len(indices)):
    #         dmat = self._calc_dist_mat(indices[i])
    #         dmatlist.append(dmat)
    #     return dmatlist


    # def _threads_by_z(self):
    #     """
    #     Organizes thread indices by z plane

    #     Arguments:
    #         e : extractor
    #             extractor object
    #     """

    #     # make object to store thread indices corresponding to z plane
    #     threads_by_z = [[] for i in self.frames]


    #     # iterate over threads, append index to threads_by_z
    #     for i in range(len(self.spool.threads)):
    #         z = int(self.spool.threads[i].positions[0,0])
    #         # ndx = np.where(np.array(self.frames) == z)[0][0]

    #         threads_by_z[z].append(i)

    #     return threads_by_z


    # def _seriation(self, Z,N,cur_index):
    #     '''
    #         input:
    #             - Z is a hierarchical tree (dendrogram)
    #             - N is the number of points given to the clustering process
    #             - cur_index is the position in the tree for the recursive traversal
    #         output:
    #             - order implied by the hierarchical tree Z
                
    #         seriation computes the order implied by a hierarchical tree (dendrogram)
    #     '''
    #     if cur_index < N:
    #         return [cur_index]
    #     else:
    #         left = int(Z[cur_index-N,0])
    #         right = int(Z[cur_index-N,1])
    #         return (self._seriation(Z,N,left) + self._seriation(Z,N,right))
        
    # def _compute_serial_matrix(self, dist_mat,method="ward"):
    #     '''
    #     input:
    #         - dist_mat is a distance matrix
    #         - method = ["ward","single","average","complete"]
    #     output:
    #         - seriated_dist is the input dist_mat,
    #           but with re-ordered rows and columns
    #           according to the seriation, i.e. the
    #           order implied by the hierarchical tree
    #         - res_order is the order implied by
    #           the hierarhical tree
    #         - res_linkage is the hierarhical tree (dendrogram)
        
    #     compute_serial_matrix transforms a distance matrix into 
    #     a sorted distance matrix according to the order implied 
    #     by the hierarchical tree (dendrogram)
    #     '''
    #     N = len(dist_mat)
    #     flat_dist_mat = squareform(dist_mat)
    #     res_linkage = linkage(flat_dist_mat, method=method,preserve_input=True)
    #     res_order = self._seriation(res_linkage, N, N + N-2)
    #     seriated_dist = np.zeros((N,N))
    #     a,b = np.triu_indices(N,k=1)
    #     seriated_dist[a,b] = dist_mat[ [res_order[i] for i in a], [res_order[j] for j in b]]
    #     seriated_dist[b,a] = seriated_dist[a,b]
        
    #     return seriated_dist, res_order, res_linkage



