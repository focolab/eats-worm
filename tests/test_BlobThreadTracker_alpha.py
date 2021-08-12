
#
#   Invokes two Extractor methods `calc_blob_threads` and `calc_blob_threadsv2`
#   and verifies that they produce identical output
#
from gcamp_extractor import Extractor
from gcamp_extractor import Curator

arguments = {
    'root':'/media/gbubnis/Extreme SSD/yakalo/jackson/2021-05-29/worm5_act1/GCAMP',
    'numz':12,
    'frames':[1,2,3,4,5,6,7,8,9,10,11],
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

print('--------')
e.calc_blob_threads()
df1 = e.spool.to_dataframe(dims=['Z', 'Y', 'X'])
print('--------')
e.im.reset_t()
e.calc_blob_threadsv2()
df2 = e.spool.to_dataframe(dims=['Z', 'Y', 'X'])
print('--------')

assert all(df1['Z'].values == df2['Z'].values)
assert all(df1['X'].values == df2['X'].values)
assert all(df1['Y'].values == df2['Y'].values)
assert all(df1['prov'].values == df2['prov'].values)

print(df1.head(10))
print(df2.head(10))

#e.quantify()
#c = Curator(e)
#c.log_curate()
