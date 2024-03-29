import dask
import glob 
import multiprocessing
import numpy as np
import os
import tifffile as tiff
import pickle
from concurrent.futures import ThreadPoolExecutor
import pynwb

from .multifiletiff import *

from pynwb import NWBHDF5IO
import pandas as pd
import h5py
#from dandi.dandiapi import DandiAPIClient
import remfile
import pickle

from pynwb import load_namespaces, get_class, register_class, NWBFile, TimeSeries, NWBHDF5IO
from pynwb.file import MultiContainerInterface, NWBContainer, Device, Subject
from pynwb.ophys import ImageSeries, OnePhotonSeries, OpticalChannel, ImageSegmentation, PlaneSegmentation, Fluorescence, DfOverF, CorrectedImageStack, MotionCorrection, RoiResponseSeries, ImagingPlane
from pynwb.core import NWBDataInterface
from pynwb.epoch import TimeIntervals
from pynwb.behavior import SpatialSeries, Position
from pynwb.image import ImageSeries

from ndx_multichannel_volume import CElegansSubject, OpticalChannelReferences, OpticalChannelPlus, ImagingVolume, VolumeSegmentation, MultiChannelVolume, MultiChannelVolumeSeries


class NWBLoader(MultiFileTiff):

    def __init__(self, *args, **kwargs):
        if 'root' in kwargs.keys() or len(args)!=0:
            try:
                self.root = kwargs['root']
            except:
                self.root = args[0]

        if isinstance(self.root, str):
            if not self.root[-4:] == '.nwb':
                print('Not a path to NWB file')
                return 0 
            
        self.output_dir = kwargs['output_dir']
            
        self.tf = [] #To store TiffFile objects. Currently only used when reading NP files

        self.frames = list(range(kwargs['numz']))

        if kwargs.get('regen'):
            try:
                regen_path = self.output_dir + '/extractor-objects/mft.obj'
                mft = pickle.load(open(regen_path, 'rb'))
                self.numz = mft.numz
                self.numc = mft.numc
                self.calc_shape = mft.calc_shape
                self.sizexy = mft.sizexy
                self.numframes = mft.numframes
                self.anisotropy = mft.anistropy
                self.frames = mft.frames
                del mft
                print('Loaded from previous extractor')
            except:
                print('mft.obj file not found. loading defaults')
                self.default_load(*args, **kwargs)

        else:
            self.default_load(*args, **kwargs)

    def default_load(self, *args, **kwargs):

        io = NWBHDF5IO(self.root, 'r', load_namespaces=True)

        read_nwb = io.read()

        self.io = io

        self.nwb = read_nwb

        self.calc_shape = read_nwb.acquisition['CalciumImageSeries'].data.shape

        self.anisotropy = kwargs.get('anisotropy', [6,1,1])

        self.numframes = self.calc_shape[0]

        if 'numz' in kwargs.keys() and 'numc' in kwargs.keys():
            self.numz = kwargs['numz']
            self.numc = kwargs['numc']
        else:
            self.numz = self.calc_shape[3]
            self.numc = self.calc_shape[4]
        
        self.end_t = self.calc_shape[0]

        self.sizexy = self.calc_shape[1:3]

        self.dtype = read_nwb.acquisition['CalciumImageSeries'].data.dtype

        self.t = 0

    def get_frames(self, frames, channel = 0, suppress_output=True):

        if not suppress_output:
            print(frames)

        if len(self.calc_shape) ==4:
            load_frames = self.nwb.acquisition['CalciumImageSeries'].data[frames, :,:,:]
        else:
            load_frames = self.nwb.acquisition['CalciumImageSeries'].data[frames, :,:,:,channel]
        
        return np.transpose(load_frames, [2,0,1]) # transpose to be ZXY

    def get_t(self, *args, **kwargs):
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

        if ti > self.numframes:
            print('end of file')
            return False
        else:
            if t_in:
                return self.get_frames(ti, channel=channel, suppress_output=sup)
            else:
                self.t +=1
                return self.get_frames(self.t-1, channel=channel, suppress_output=sup)


    def save(self, *args, **kwargs):

        path = os.path.join(self.output_dir, 'mft.obj')

        if kwargs.get('path'):
            path = kwargs['path']

        elif len(args) == 1:
            path = args[0]

        m = minimal_mft(self)
        f = open(path, 'wb')
        pickle.dump(m, f)
        f.close()

class minimal_mft:
    def __init__(self, mft):
        self.calc_shape = mft.calc_shape
        self.numz = mft.numz
        self.numc = mft.numc
        self.sizexy = mft.sizexy
        self.numframes = mft.numframes
        self.anisotropy = mft.anisotropy
        self.frames = mft.frames
 
