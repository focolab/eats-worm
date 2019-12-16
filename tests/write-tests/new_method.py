import tifffile as tiff
from multifiletiff import *
from rainbow_tools import *
from Threads import *


im = MultiFileTiff('/Users/stevenban/Documents/Data/20190917/binned')

im.set_frames([0,1,2,3,4,5,6,7,8,9,10,11])
im.numz = 20


im.t = 0
#s = Spool()
t0 = time.time()
peaks_total = []
for i in range(999):
	im1 = im.get_t()
	im1 = medFilter2d(im1)
	im1 = gaussian3d(im1,(25,3,3,1))
	peaks1 = findpeaks3d(np.array(im1 * np.array(im1 > np.quantile(im1,.99))))
	peaks1 = reg_peaks(im1, peaks1,thresh=40)
	peaks_total.append(peaks1)
	#s.reel(peaks1)
	print(i)
print('total time:', time.time()-t0)




markers = np.zeros(tuple([999]) + im.sizexy,dtype=np.uint16)
mwidth = 3

for i in range(len(peaks_total)):
	peaks = peaks_total[i]
	for peak in peaks:
		try:
			x,y,z = peak
			markers[i,y-mwidth:y+mwidth,z-mwidth:z+mwidth] += 1
		except: pass

import pickle
with open('20190917-markers-segonly.obj', 'wb') as file:
	pickle.dump(peaks_total, file)



file_pi = open('20190917-markers-segonly.obj', 'wb') 
pickle.dump(peaks_total, file_pi)
p = pickle.load( open( "20190917-markers-segonly.obj", "rb" ) )




tiff.imsave('20190917-markers-segonly.tif',markers.astype(np.uint16))
