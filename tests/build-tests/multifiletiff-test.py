from gcamp_extractor import *
im = MultiFileTiff('/Users/stevenban/Documents/Data/20190917/binned')



import numpy as np
import tifffile as tiff


a = np.zeros((16,16),dtype=np.uint16)

file1 = np.zeros((10,16,16),dtype=np.uint16)
filename1 = 'file01.tif'

file2 = np.zeros((2,16,16),dtype=np.uint16)
filename2 = 'file2.tif'

file3 = np.zeros((8,16,16),dtype=np.uint16)
filename3 = 'file03.tif'







