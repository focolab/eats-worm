import sys
sys.path.append('/Users/stevenban/Documents/gcamp-extractor/gcamp-extractor')

import numpy as np
import scipy
from Threads import *
import pickle
from multifiletiff import MultiFileTiff
from segtools import *
import time 
peaks = pickle.load( open( "misc-files/20190917-markers-segonly.obj", "rb" ) )
import tifffile as tiff
import copy
"""
Tracking based on updated velocities from other blobs

"""

im = MultiFileTiff('/Users/stevenban/Documents/Data/20190917/binned')

im.set_frames([0,1,2,3,4,5,6,7,8,9,10,11])
im.numz = 20


im.t = 0
s = Spool()
t0 = time.time()
for i in range(999):
    im1 = im.get_t()
    im1 = medFilter2d(im1)
    im1 = gaussian3d(im1,(25,4,3,1))
    peaks1 = findpeaks3d(np.array(im1 * np.array(im1 > np.quantile(im1,.98))))
    peaks1 = reg_peaks(im1, peaks1,thresh=40)
    s.reel(peaks1)
    print(i)

s.infill()
def get_circlemask(center):
    z,y,x = [],[],[]
    for i in range(-3,4):
        for j in range(-3,4):
            z.append(center[0])
            y.append(center[1] + j)
            x.append(center[2] + i)
    return z,y,x


im.t = 0
timeseries = np.zeros((999,len(s.threads)))
for i in range(999):
    im1 = im.get_t()
    for j in range(len(s.threads)):
        center = s.threads[j].get_position_t(i).astype(int)
        masked = im1[get_circlemask(center)]
        masked.sort()
        timeseries[i,j] = np.mean(masked[-10:])
    print(i)

#np.savetxt('brightextpix-timeseries.txt',timeseries)

print('total time:', time.time()-t0)


import pickle
file_pi = open('20190917.obj', 'wb') 
pickle.dump(s, file_pi)


'''
num_neurons = len(s.threads)
markers = np.zeros(tuple([999]) + im.sizexy,dtype=np.uint16)
mwidth = 3
for i in range(999):
    #im1 = im.get_t()
    for j in range(num_neurons):
        try:
            pos = s.threads[j].get_position_t(i).astype(int)
            markers[i,pos[1]-mwidth:pos[1]+mwidth,pos[2]-mwidth:pos[2]+mwidth] += 1
        except: pass
    if i%10 == 0:
        print(i)


tiff.imsave('20190917-markers.tif',markers.astype(np.uint16))

'''




'''



im = MultiFileTiff('/Users/stevenban/Documents/Data/20190917/binned')

im.set_frames([0,1,2,3,4,5,6,7,8,9,10,11])
im.numz = 20


im.t = 0
s = Spool()
t0 = time.time()

for i in range(999):

    # iterating over time points
    im1 = im.get_t()
    peaks_t = peaks[i]
    peaks_t = reg_peaks(im1, peaks_t, thresh=6)
    s.reel(peaks_t)
    print(i)
print('total time:', time.time()-t0)

def missing_t(thread):
	print(set(range(998)) - set(thread.t))
'''
'''
num_neurons = len(s.threads)
markers = np.zeros(tuple([999]) + im.sizexy,dtype=np.uint16)
mwidth = 3
for i in range(999):
    #im1 = im.get_t()
    for j in range(num_neurons):
        try:
            pos = s.threads[j].get_position_t(i).astype(int)
            markers[i,pos[1]-mwidth:pos[1]+mwidth,pos[2]-mwidth:pos[2]+mwidth] += 1
        except: pass
    if i%10 == 0:
        print(i)

tiff.imsave('20190917-markers-flocking.tif',markers.astype(np.uint16))


'''

'''
num_neurons = len(s.threads)
markers = np.zeros(tuple([999]) + im.sizexy,dtype=np.uint16)
mwidth = 3
for i in range(999):
    #im1 = im.get_t()
    for j in range(num_neurons):
        try:
            pos = s.threads[j].get_position_t(i).astype(int)
            markers[i,pos[1]-mwidth:pos[1]+mwidth,pos[2]-mwidth:pos[2]+mwidth] += 1
        except: pass
    if i%10 == 0:
        print(i)

tiff.imsave('20190917-markers-flocking-infilled.tif',markers.astype(np.uint16))

def get_circlemask(center):
	z,y,x = [],[],[]
	for i in range(-3,4):
		for j in range(-3,4):
			z.append(center[0])
			y.append(center[1] + j)
			x.append(center[2] + i)
	return z,y,x


im.t = 0
timeseries = np.zeros((999,len(s.threads)))
for i in range(999):
	im1 = im.get_t()
	for j in range(len(s.threads)):
		center = s.threads[j].get_position_t(i).astype(int)
		masked = im1[get_circlemask(center)]
		masked.sort()
		timeseries[i,j] = np.mean(masked[-10:])
	print(i)

np.savetxt('brightextpix-timeseries.txt',timeseries)

for i in range(timeseries.shape[1]):
	#timeseries[:,i] = (timeseries[:,i]-np.min(timeseries[:,i]))/(np.max(timeseries[:,i])-np.min(timeseries[:,i]))
    timeseries[:,i] = timeseries[:,i]/timeseries[0,i]

plt.imshow(timeseries.T, aspect='auto')
#plt.axis('off')
plt.show()
plt.savefig('brightestpix-timeseries.png', dpi = 1000, bbox_inches='tight')

for i in range(100):
	plt.plot(timeseries[:,i])
	plt.show()

import os
def mkdir(path):
    try: 
        os.mkdir(path)
    except:
        pass
mkdir('timeseries')

t = []
for i in range(999):
    t.append(0.33 * i)

fig,ax = plt.subplots(nrows = 50, ncols = 8, sharex = False, sharey = False,figsize=(20,9))
counter = 0
for i,row in enumerate(ax):
    for j,axs in enumerate(row):
    #axs = fig.add_subplot(10,1,i+1)
        try:
            axs.plot(t, e.timeseries[:,40*j+i],linewidth=0.65)
            axs.set_xlabel('')
            axs.set_ylabel(str(40*j+i),fontsize=4)
            axs.spines['top'].set_visible(False)
            axs.spines['right'].set_visible(False)
            axs.spines['bottom'].set_visible(False)
            axs.spines['left'].set_visible(False)
            axs.get_xaxis().set_ticks([])
            axs.get_yaxis().set_ticks([])
            counter += 1
        except:
            axs.set_xlabel('')
            axs.set_ylabel(str(counter),fontsize=4)
            axs.spines['top'].set_visible(False)
            axs.spines['right'].set_visible(False)
            axs.spines['bottom'].set_visible(False)
            axs.spines['left'].set_visible(False)
            axs.get_xaxis().set_ticks([])
            axs.get_yaxis().set_ticks([])
    #if i == 0:
    #    ax.set_aspect(aspect=1.5)
#fig.xlabel('t (s)')
#axs.set_xlabel('t (s)')
#fig.text(0.1, 0.5, 'grayscale units', va='center', rotation='vertical')
#axs.get_xaxis().set_ticks([0,100,200,300])
fig.savefig('timeseries/almostall' + "{:03d}".format(i) + '.png',bbox_inches='tight',dpi=1000)
plt.close()

im.t = 0
m = np.zeros((999,1))




for i in range(999):
    m[i] = np.mean(im.get_t())
    if i%10 == 0:
        print(i)
plt.plot(m)
plt.savefig('background.png', dpi = 1000)



"""
PLOT THE 'INTERESTING' NEURONS
"""


neurons = [5,11,13,15,18,26,30,34,65,79,94,98,102,114,141,180,232,304,192,198]
neurons.sort()
fig,ax = plt.subplots(nrows = 10, ncols = 2, sharex = False, sharey = False,figsize=(8,5))
counter = 0
for i,row in enumerate(ax):
    for j,axs in enumerate(row):
    #axs = fig.add_subplot(10,1,i+1)
        try:
            axs.plot(t, timeseries[:,neurons[j*10+i]],linewidth=0.75)
            axs.set_xlabel('')
            axs.set_ylabel(str(neurons[j*10+i]),fontsize=4)
            axs.spines['top'].set_visible(False)
            axs.spines['right'].set_visible(False)
            axs.spines['bottom'].set_visible(False)
            axs.spines['left'].set_visible(False)
            axs.get_xaxis().set_ticks([])
            axs.get_yaxis().set_ticks([])
            counter += 1
        except:
            axs.set_xlabel('')
            axs.set_ylabel(str(counter),fontsize=6)
            axs.spines['top'].set_visible(False)
            axs.spines['right'].set_visible(False)
            axs.spines['bottom'].set_visible(False)
            axs.spines['left'].set_visible(False)
            axs.get_xaxis().set_ticks([])
            axs.get_yaxis().set_ticks([])
    #if i == 0:
    #    ax.set_aspect(aspect=1.5)
#fig.xlabel('t (s)')
#axs.set_xlabel('t (s)')
#fig.text(0.1, 0.5, 'grayscale units', va='center', rotation='vertical')
#axs.get_xaxis().set_ticks([0,100,200,300])
fig.savefig('timeseries/interesting' + "{:03d}".format(i) + '.png',bbox_inches='tight',dpi=500)
plt.close()




markers = np.zeros(tuple([999]) + im.sizexy,dtype=np.uint16)
mwidth = 3
for i in range(999):
    #im1 = im.get_t()
    for j in range(len(neurons)):
        try:
            pos = s.threads[neurons[j]].get_position_t(i).astype(int)
            markers[i,pos[1]-mwidth:pos[1]+mwidth,pos[2]-mwidth:pos[2]+mwidth] += 1
        except: pass
    if i%10 == 0:
        print(i)
tiff.imsave('20190917-markers-interesting.tif',markers.astype(np.uint16))
'''
