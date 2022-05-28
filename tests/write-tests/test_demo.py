#
#import sys
#sys.path.append('/Users/stevenban/Documents/eats-worm/eats-worm')
'''
from Extractor import *
from Threads import *
from segfunctions import *
from Curator import *
'''

from eats_worm import *
arguments = {
    'root':'/Users/stevenban/Documents/Data/20190917/test',
    'numz':20,
    'frames':[0,1,2,3,4,5,6,7,8,9,10,11],
    'offset':0,
    't':2000,
    'gaussian':(25,4,3,1),
    'quantile':0.99,
    'reg_peak_dist':5.99,
    'anisotropy':(6,1,1),
    'blob_merge_dist_thresh':5.8,
    'mip_movie':True,
    'marker_movie':True,
    'infill':True,
    'save_threads':True,
    'save_timeseries':True,
    'suppress_output':False,
    'regen':False,
}

#e = Extractor(**default_arguments)
e.calc_blob_threads()
e.quantify()


e = load_extractor(arguments['root'])
Curator(e)