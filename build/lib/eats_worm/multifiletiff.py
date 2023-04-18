#!/usr/bin/env python3

import dask
import glob
import multiprocessing
import numpy as np
import os
import tifffile as tiff
import pickle
from concurrent.futures import ThreadPoolExecutor

class MultiFileTiff():
    """
    File for handling volumetric recording data that has been spread over multiple separate tiff files. 

    Parameters
    ----------
    root : string, mandatory (can be passed as non-keyword argument)
        string representing path to folder containing the tiff files. If no tiff files exist in the root folder, will search all subdirectories for tiff files as well. Can end with '/' or not
    output_dir : string, optional (keyword argument only)
        string representing path to folder to use for output objects. Defaults to root. Can end with '/' or not
    offset : int, optional (keyword argument only)
        integer representing the number of frames at the beginning to throw away
    numz : int, optional (keyword argument only)
        integer representing number of z slices per time point
    frames : list, optional (keyword argument only)
        list of integers representing frames to keep mod numz for analysis. Python indexing applies, so to keep the first 5 frames per numz frames, pass frames = [0,1,2,3,4]
    regen : bool, optional (keyword argument only)
        boolean, whether to regenerate mft from 'mft.obj' stored somewhere in root directory
    
    Attributes
    ----------
    filenames : list
        list of filenames in multifiletiff structure in the order that they will be accessed
    tf : list
        list of TiffFile pointer objects for all files
    lens : list    
        list of number of pages for each individual file
    sizexy : tuple
        size of each frame in pixels
    dtype : type
        dtype of the individual pixels
    numz : int
        number of z frames per time point
    frames : list
        list of integers for frames to keep mod numz
    t : int
        internal method of keeping track of what time points have been accessed

    Methods
    -------
    __init__ 
        initializes the file. 


    """
    def __init__(self, *args, **kwargs):

        ## Find root in args/kwargs
        if 'root' in kwargs.keys() or len(args)!=0:
            try:
                self.root = kwargs['root']
            except:
                self.root = args[0]
        ## Check to make sure root is path to directory
        if isinstance(self.root, str):
            if self.root[-1] != '/': #make sure path ends with "/"
                self.root = self.root + '/'

            ## Find all tiff files in root folder
            self.filenames = glob.glob(self.root + '*.tif') + glob.glob(self.root + '*.tiff')

            ## If they don't exist, find all tiff files in subdirectory
            if len(self.filenames) == 0:
                self.filenames = self.list_files_tiff(self.root)
        else:
            print('Not the root to a directory')
            return 0 
        if len(self.filenames) == 0:
            print('Not the root to a directory')
            return 0 
        if len(self.filenames) > 1:
            self.sort_filenames()
        
        try:
            self.output_dir = kwargs['output_dir']
        except:
            self.output_dir = os.path.join(self.root, 'extractor-objects')

        ## Create TiffFile objects for each file in directory
        self.tf = []
        for i in range(len(self.filenames)):
            self.tf.append(tiff.TiffFile(self.filenames[i]))


        ## If MFT is being regenerated from an 'mft.obj' file:
        if kwargs.get('regen'):
            ndx = []
            l = self.list_all_files(self.root)
            for i in range(len(l)):
                if 'mft.obj' in l[i]:
                    ndx.append(i)
            if len(ndx) == 0:
                print('mft.obj file not found. loading defaults')
                self.default_load(*args, **kwargs)
            elif len(ndx) == 1:
                mft_filename = l[ndx[0]]
                mft = pickle.load(open(mft_filename, "rb"))
                self.filenames = mft.filenames
                self.lens = mft.lens
                self.indexing = mft.indexing
                self.numz = mft.numz
                self.numc = mft.numc
                self.frames = mft.frames
                self.sizexy = mft.sizexy
                self.numframes = mft.numframes
                self.anisotropy = mft.anisotropy
                del mft
            else:
                print('loading file at: '+l[ndx[0]])
                mft_filename = l[ndx[0]]
                mft = pickle.load(open(mft_filename, "rb"))
                self.filenames = mft.filenames
                self.lens = mft.lens
                self.indexing = mft.indexing
                self.numz = mft.numz
                self.numc = mft.numc
                self.frames = mft.frames
                self.sizexy = mft.sizexy
                self.numframes = mft.numframes
                self.anisotropy = mft.anisotropy
                del mft

        else:
            self.default_load(*args, **kwargs)

    def default_load(self, *args, **kwargs):
        ## Calculate lengths of each file for indexing purposes
        self.lens = [0 for x in self.tf]
        workers = max(multiprocessing.cpu_count() // 2, 1)
        with ThreadPoolExecutor(max_workers=workers) as executor: 
            for index, length in zip(list(range(len(self.tf))),executor.map(lambda x: len(x.pages),self.tf)):
                self.lens[index] = length
        self.numframes = np.sum(self.lens)

        self.anisotropy = kwargs.get('anisotropy', [6, 1, 1])

        # Process inputs for numz and frames
        self.numz = 10
        self.numc = 1
        if 'numz' in kwargs.keys() and 'numc' in kwargs.keys():
            self.numz = kwargs['numz']
            self.numc = kwargs['numc']
        else:
            max_slice_index = -1
            max_channel_index = -1
            for tif in self.tf:
                for page in tif.pages:
                    max_channel_index = max(max_channel_index, page.tags['MicroManagerMetadata'].value['ChannelIndex'])
                    max_slice_index = max(max_slice_index, page.tags['MicroManagerMetadata'].value['SliceIndex'])
            self.numz = max_slice_index + 1
            self.numc = max_channel_index + 1
        self.numframes //= self.numc
        
        self.frames = np.array([i for i in range(self.numz)])
        if 'frames' in kwargs.keys():
            self.frames = np.array(kwargs['frames'])

        # Calculate indexing function
        ## Process kwargs for offset
        self.offset = 0
        if 'offset' in kwargs.keys():
            self.offset = kwargs['offset']

        ## Calculate indexing
        _s = np.cumsum(self.lens)

        for i in range(len(_s)):
            if self.offset > _s[i]:
                pass
            else:
                filecounter = i
                if i == 0:
                    pagecounter  = self.offset
                else:
                    pagecounter = self.offset - _s[i-1]
                break
        # assumes that multichannel scans are done as channel, slice
        self.indexing = {}
        for i in range(self.numframes-self.offset):
            self.indexing[i] = []
            if i > 0:
                last_filecounter, last_pagecounter = self.indexing[i - 1][len(self.indexing[i])]
                filecounter = last_filecounter
                pagecounter = last_pagecounter + 1
            while len(self.indexing[i]) < self.numc:
                if pagecounter < self.lens[filecounter]:
                    self.indexing[i].append([filecounter, pagecounter])
                    pagecounter += self.numz
                else:
                    pagecounter = pagecounter - self.lens[filecounter]
                    filecounter += 1

        # calculate end timepoint
        self.end_t = int((self.numframes-self.offset)//self.numz)

        # Get some shape data from file
        self.sizexy = self.tf[0].pages[0].asarray().shape
        self.dtype = type(self.tf[0].pages[0].asarray()[0,0])
        self.t = 0

    def list_files_tiff(self, path):
        """
        method for listing all .tif files in directory and all subdirectories. internal method only


        Parameters
        ----------
        path : str
            path to directory


        Outputs
        -------
        names : list
            list of strings, each of which are paths to files existing in the directory. 
        """
        # create a list of file and sub directories 
        # names in the given directory 
        listOfFile = os.listdir(path)
        allFiles = list()
        # Iterate over all the entries
        for entry in listOfFile:
            # Create full path
            fullPath = os.path.join(path, entry)
            # If entry is a directory then get the list of files in this directory 
            if os.path.isdir(fullPath):
                allFiles = allFiles + self.list_files_tiff(fullPath)
            else:
            	if fullPath[-4:]=='.tif' or fullPath[-5:]=='.tiff':
                	allFiles.append(fullPath)

        return allFiles

    def list_all_files(self,path):
        """
        method for listing all files in directory and all subdirectories. internal method only


        Parameters
        ----------
        path : str
            path to directory


        Outputs
        -------
        names : list
            list of strings, each of which are paths to files existing in the directory. 
        """
        # create a list of file and sub directories 
        # names in the given directory 
        listOfFile = os.listdir(path)
        allFiles = list()
        # Iterate over all the entries
        for entry in listOfFile:
            # Create full path
            fullPath = os.path.join(path, entry)
            # If entry is a directory then get the list of files in this directory 
            if os.path.isdir(fullPath):
                allFiles = allFiles + self.list_all_files(fullPath)
            else:
                allFiles.append(fullPath)
        return allFiles

    def get_dask_array(self):
        """
        method for getting entire recording as dask array with single timepoint chunks
        useful for viewing with napari
        """
        shape = self.get_t(0).shape
        timepoints = []
        for t in range(self.end_t):
            frames = dask.array.from_delayed(dask.delayed(self.get_t)(t), shape=shape, dtype=np.int16)
            timepoints.append(frames)
        timepoints =  dask.array.stack(timepoints)
        return timepoints

    def get_frame(self, frame, channel = 0):
        """
        method for getting a single frame from your recording, mostly used internally. if you're going to call this function, please make sure you know what's going on.

        Parameters
        ----------
        frame : int
            frame (python index) from your recording that you want to get, indexed by frames from your original recording. for example, if you want the 10th frame from your recording, call get_frame(9). 

        Outputs
        -------
        frame : np.array
            2d numpy array of the image data requested 
        """
        frame = int(frame)
        channels = []
        for filecounter, pagecounter in self.indexing[frame]:
            channels.append(self.tf[filecounter].pages[pagecounter].asarray())
        return channels[channel]

    def get_frames(self, frames, channel = 0, suppress_output = True):
        """
        method for getting multiple frames at a time. mostly used internally. if you're going to call this function, please make sure you know what's going on. 

        Parameters
        ----------
        frames : list of int
            list of frames (note, frames and not z-steps) that you want to get. for example, if you want the first 5 frames of your *recording*, then call get_frames([0,1,2,3,4]). 
        suppress_output : bool (optional)
            boolean for whether you want the Python console output (frames accessed) to be suppressed. Default is False

        Output
        ------
        frames : np.array
            3d numpy array of image data requested. note that the return is 3 dimensional, so if you call this with 1 frame, you still get a 3d array with size (1,x,y). 
        """
        if not suppress_output:
            print(frames)

        frames = list(frames)
        returnim = np.zeros(((len(frames),) + self.sizexy), dtype=np.uint16)

        #[np.zeros(self.sizexy,dtype=np.uint16) for x in range(len(frames))]


        #np.zeros(tuple([len(frames)]) + self.sizexy,dtype=np.uint16)
        '''
        workers = max(multiprocessing.cpu_count() // 2, 1)
        with ThreadPoolExecutor(max_workers=workers) as executor: 
            for index, im in zip(list(range(len(frames))),executor.map(self.get_frame,frames)):
                print(index,im)
                try:
                    returnim[index] = im.astype(np.uint16)
                except:
                    pass
        '''
        for i in range(len(frames)):
            returnim[i] = self.get_frame(frames[i], channel = channel)
        #returnim = np.array(returnim)
        return returnim

    def set_frames(self, frames):
        """
        sets the frames (z-steps) that you want to keep. i.e. if there are 10 z's and you want to keep the first 7, then call set_frames([0,1,2,3,4,5,6]).

        Parameters
        ----------
        frames : list (of ints)
            list of ints (python indexes) for the frames (z's in your recording) that you want to keep. 
        """
        self.frames = np.array(frames)

    def set_numz(self, numz):
        """
        sets the number of z frames in the recording

        Parameters
        ----------
        numz : int
            number of z steps in the recording
        """

        self.numz = numz
    def set_offset(self, offset):
        """
        sets the offsets (number of frames to throw away at the beginning)

        Parameters
        ----------
        offset : int
            number of frames to throw away at the beginning
        """
        self.indexing = {}
        pagecounter = 0
        filecounter = 0
        for i in range(self.numframes):
            if pagecounter < self.lens[filecounter]:
                if not i < offset:
                    self.indexing[i] = [filecounter, range(pagecounter, pagecounter + self.numc)]
                pagecounter += self.numc
            else:
                pagecounter = 1
                filecounter += 1
                if not i < offset:
                    self.indexing[i] = [filecounter, range(pagecounter, pagecounter + self.numc)]
    def reset_t(self):
        """
        resets internal time counter to the beginning. no arguments necessary
        """
        self.t = 0
    def get_t(self, *args, **kwargs):
        """
        get frames for a particular time point

        Parameters
        ----------
        t : int (optional, arg or keyword-arg)
            time point to get accessed. if not passed, then will return the next time point that hasn't gotten accessed

        suppress_output : bool (optional, kwarg only)
            output for get_t is the list of frames that have been accessed. default is False

        Returns
        -------
        im : numpy.array
            3D numpy array of the image data at time t
        """

        # Input argument processing
        t_in = False
        ti = self.t
        if len(args) != 0:
            ti = args[0]
            t_in = True
        elif 't' in kwargs.keys():
            ti = kwargs['t']
            t_in = True
        sup = True
        if 'suppress_output' in kwargs.keys():
            sup = kwargs['suppress_output']
        channel = 0
        if 'channel' in kwargs.keys():
            channel = kwargs['channel']

        if ti > (self.numframes)/self.numz:
            print('end of file')
            return False
        else:
            if t_in:
                #self.t = ti
                return self.get_frames(self.frames + (ti) * self.numz, channel = channel, suppress_output = sup)
            else:
                self.t += 1
                return self.get_frames(self.frames + (self.t-1) * self.numz, channel = channel, suppress_output = sup)
    def get_tbyf(self,*args,**kwargs):
        """
        get frame for a particular z step by time

        Parameters
        ----------
        t : int
            time point that you want to access
        f : int
            frame (z step)

        Returns
        -------
        im : np.array
            2d image file of time point at particular z step
        """
        t_in = False
        ti = self.t
        if len(args) != 0:
            ti = args[0]
            t_in = True
        elif 't' in kwargs.keys():
            ti = kwargs['t']
            t_in = True
        sup = True
        if 'suppress_output' in kwargs.keys():
            sup = kwargs['suppress_output']
        channel = 0
        if 'channel' in kwargs.keys():
            channel = kwargs['channel']

        if len(args) == 2:
            f = args[1]
        elif 'f' in kwargs.keys():
            f = kwargs['f']

        if ti > (self.numframes)/self.numz:
            print('end of file')
            return False
        else:
            if t_in:
                self.t = ti
                return self.get_frame(f + (ti) * self.numz, channel = channel)
            else:
                self.t += 1
                return self.get_frame(f + (self.t-1) * self.numz, channel = channel)



    def sort_filenames(self):
        """
        sort filenames by index. written for compatibility between files with 'XXX-1.tif' and 'XXX-02.tif'
        if a single file without an index like 'XXX.tif' is found, that file will be assumed to be the first in the series
        if more than one such unindexed file is found, this method will break
        """

        ### Remove .tif or .ome.tif

        if self.filenames[0][-8:] == '.ome.tif':    
            files = [self.filenames[i][:-8] for i in range(len(self.filenames))]
        elif self.filenames[0][-9:] == '.ome.tiff':    
            files = [self.filenames[i][:-9] for i in range(len(self.filenames))]
        elif self.filenames[0][-4:] == '.tif':
            files = [self.filenames[i][:-4] for i in range(len(self.filenames))]
        elif self.filenames[0][-5:] == '.tiff':
            files = [self.filenames[i][:-5] for i in range(len(self.filenames))]

        unindexed_file_found = False
        ndx = []
        for i in range(len(files)):
            for j in range(1,10):
                if files[i][-j].isdigit():
                    pass
                else:
                    try:
                        ndx.append(int(files[i][-j+1:]))
                        break
                    except Exception as e:
                        if not unindexed_file_found:
                            ndx.append(0)
                            unindexed_file_found = True
                            break
                        else:
                            raise e
        self.filenames = [x for _,x in sorted(zip(ndx,self.filenames))]

    def save(self, *args, **kwargs):
        path = os.path.join(self.output_dir, 'mft.obj')

        if kwargs.get('path'):
            path = kwargs['path']
        elif len(args) == 1:
            path = args[0]


        m  = minimal_mft(self)
        f = open(path, 'wb') 
        pickle.dump(m, f)
        f.close()

class minimal_mft:
    def __init__(self,mft):
        self.filenames = mft.filenames
        self.lens = mft.lens
        self.indexing = mft.indexing
        self.numz = mft.numz
        self.numc = mft.numc
        self.frames = mft.frames
        self.sizexy = mft.sizexy
        self.numframes = mft.numframes
        self.anisotropy = mft.anisotropy
