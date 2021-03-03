#!/usr/bin/env python3
## Example use case of gcamp extractor package


import numpy as np
from gcamp_extractor.Extractor import *
from gcamp_extractor.FilterSweeper import *
#from gcamp_extractor.Threads import *
from gcamp_extractor.Curator import *
#from gcamp_extractor import *

FCD = '../FOCO_FCD_v0.1'

arguments = {
    'root':FCD+'/RLD_selected_recordings/shortened_recs/renamed',
    'numz':10,
    'frames':[0,1,2,3,4,5,6,7,8,9],
    'offset':0,
    't':499,
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

#uncomment these lines to enable filter/threshold parameter sweep
sweeper = FilterSweeper.from_extractor(e)
sweeper.sweep_parameters()
e.gaussian, e.median, e.quantile = sweeper.gaussian, sweeper.median, sweeper.quantile


e.calc_blob_threads()
e.quantify()
c = Curator(e)


