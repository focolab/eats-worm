
#
#   This tests that Extractor results (timeseries and spool) remain unchanged
#   while refactoring. 
#
#   The hashes were computed using the 8/26/2021 'db85113' commit on the
#   template branch
#

from gcamp_extractor import Extractor
from gcamp_extractor import FilterSweeper
from gcamp_extractor import Curator
from hashlib import sha1

FCD = '/home/gbubnis/prj/bfd/FOCO_FCD_v0.1/'

#### CASE 1
print('=============================================================')
print('=============================================================')
print('=============================================================')
arguments = {
    'root':FCD+'RLD_selected_recordings/shortened_recs',
    'numz':10,
    #'frames':[1,2,3,4,5,6,7,8,9,10,11],
    'offset': 0,
    't':8,
    'gaussian':(41,5,3,1),
    'quantile':0.99,
    'reg_peak_dist':8,
    'anisotropy':(10,1,1),
    'blob_merge_dist_thresh': 8,
    'register_frames':True,
    'predict':True,
    'regen_mft':False,
    '3d':True,
    'output_dir':'gce_output',
}

e = Extractor(**arguments)
e.calc_blob_threads()
e.quantify()

print('--------')
print('threads hash    :', sha1(e.spool.allthreads).hexdigest())
print('threads shape   :', e.spool.allthreads.shape)
print('timeseries hash :', sha1(e.timeseries).hexdigest())
print('timeseries shape:', e.timeseries.shape)

assert sha1(e.spool.allthreads).hexdigest()[:6] == 'dce176'
assert sha1(e.timeseries).hexdigest()[:6] == 'b3e7ff'
assert e.timeseries.shape == (8, 60)
assert e.spool.allthreads.shape == (8, 180)


#### CASE 2
print('=============================================================')
print('=============================================================')
print('=============================================================')
arguments['skimage'] = True
e = Extractor(**arguments)
e.calc_blob_threads()
e.quantify()

print('--------')
print('threads hash    :', sha1(e.spool.allthreads).hexdigest())
print('threads shape   :', e.spool.allthreads.shape)
print('timeseries hash :', sha1(e.timeseries).hexdigest())
print('timeseries shape:', e.timeseries.shape)

assert sha1(e.spool.allthreads).hexdigest()[:6] == 'f7b526'
assert sha1(e.timeseries).hexdigest()[:6] == 'dc64fe'
assert e.timeseries.shape == (8, 13)
assert e.spool.allthreads.shape == (8, 39)





# #### CASE 3
# print('=============================================================')
# sw = FilterSweeper(e)
# sw.sweep_parameters()
#
# c = Curator(e)
# c.log_curate()
