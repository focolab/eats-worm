import os
from pynwb import load_namespaces, get_class
from pynwb.file import MultiContainerInterface, NWBContainer
import skimage.io as skio
from collections.abc import Iterable
import numpy as np
from pynwb import register_class
from hdmf.utils import docval, get_docval, popargs
from pynwb.ophys import ImageSeries 
from pynwb.core import NWBDataInterface
from hdmf.common import DynamicTable
from hdmf.utils import docval, popargs, get_docval, get_data_shape, popargs_to_dict
from pynwb.file import Device
import pandas as pd
import numpy as np
from pynwb import NWBFile, TimeSeries, NWBHDF5IO
from pynwb.epoch import TimeIntervals
from pynwb.file import Subject
from pynwb.behavior import SpatialSeries, Position
from pynwb.image import ImageSeries
from pynwb.ophys import OnePhotonSeries, OpticalChannel, ImageSegmentation, Fluorescence, CorrectedImageStack, MotionCorrection, RoiResponseSeries
from datetime import datetime
from dateutil import tz
import pandas as pd
import scipy.io as sio
from datetime import datetime, timedelta
from .ndx_multichannel_volume import *

# Set path of the namespace.yaml file to the expected install location
MultiChannelVol_specpath = os.path.join(
    os.path.dirname(__file__),
    'spec',
    'ndx-multichannel-volume.namespace.yaml'
)

# If the extension has not been installed yet but we are running directly from
# the git repo

if not os.path.exists(MultiChannelVol_specpath):
    MultiChannelVol_specpath = os.path.abspath(os.path.join(
        os.path.dirname(__file__),
        '..', '..', '..',
        'spec',
        'ndx-multichannel-volume.namespace.yaml'
    ))


# Load the namespace
load_namespaces(MultiChannelVol_specpath)




