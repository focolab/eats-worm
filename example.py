#!/usr/bin/env python3
## Example use case of gcamp extractor package


#from gcamp_extractor import *

import sys
sys.path.append('/Users/stevenban/Documents/gcamp-extractor/gcamp-extractor')\


import numpy as np
from Extractor import *
from Threads import *
from Curator import *

from gcamp_extractor import *

arguments = {
    'root':'/Users/stevenban/Documents/Data/20190917/binned',
    'numz':20,
    'frames':[0,1,2,3,4,5,6,7,8,9,10,11],
    'offset':0,
    't':999,
    'gaussian':(25,4,3,1),
    'quantile':0.99,
    'reg_peak_dist':40,
    'anisotropy':(6,1,1),
    'blob_merge_dist_thresh':7,
    'mip_movie':True,
    'marker_movie':True,
    'infill':True,
    'save_threads':True,
    'save_timeseries':True,
    'suppress_output':False,
    'regen':False,
}

e = Extractor(**arguments)
e.calc_blob_threads()
e.quantify()
c = Curator(e)


