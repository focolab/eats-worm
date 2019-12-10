# GCAMP Processing
## Author: Steven Ban


### Installation Instructions
1. Setup your development environment. If you need help with this, there are many resources on Google. Most people in the lab use Conda as a package manager and develop in Conda virtual environments as well. 

2. Start your virtual environment, and run the command
```bash
pip install git+https://github.com/focolab/gcamp_extractor#egg=gcamp_extractor
```
And that's basically it!

### Usage
An example use case is found in `example.py` in the root directory. The minimal use case is to extract GCaMP timeseries out of a recording, and can be accomplished with the following lines:

```python3
from gcamp_extractor import *
arguments = {
    'root':'/Users/stevenban/Documents/Data/20190917/binned',
    'numz':20,
    #'frames':[0,1,2,3,4,5,6,7,8,9,10,11],
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
```

The only 'coding' necessary here is to modify the 'root' directory that contains all your .tif files from your recording, the number of z-planes used, what frames you want to keep. If you're feeling really fancy, maybe even parameters like the size of your Gaussian filter, the percentile you threshold, the anisotropy/voxel size of your recording, and etc. 



