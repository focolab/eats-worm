#from Threads import *
import tifffile as tiff
from multifiletiff import *
from rainbow_tools import *
import sys
sys.path.append('/Users/stevenban/Documents/eats-worm/eats-worm')
from Threads import *


im = MultiFileTiff('/Users/stevenban/Documents/Data/20190917/binned')

im.set_frames([0,1,2,3,4,5,6,7,8,9,10,11])
im.numz = 20


im.t = 0
s = Spool()
t0 = time.time()
for i in range(20):
	im1 = im.get_t()
	im1 = medFilter2d(im1)
	im1 = gaussian3d(im1,(25,4,3,1))
	peaks1 = findpeaks3d(np.array(im1 * np.array(im1 > np.quantile(im1,.98))))
	peaks1 = reg_peaks(im1, peaks1,thresh=40)
	s.reel(peaks1)
	print('\r' + 'Frame: ' + str(i), sep='', end='', flush=True)

print('\nInfilling...')
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
    print('\r' + 'Frame: ' + str(i), sep='', end='', flush=True)


#print('total time:', time.time()-t0)

import pickle
file_pi = open('20190917-medfilt-regpeaks.obj', 'wb') 
pickle.dump(s, file_pi)


num_neurons = len(s.threads)
markers = np.zeros(tuple([999]) + im.sizexy,dtype=np.uint16)
mwidth = 3
for i in range(999):
	#im1 = im.get_t()
	for j in range(num_neurons):
		try:
			x,y,z = s.threads[j].get_position_t(i)
			markers[i,y-mwidth:y+mwidth,z-mwidth:z+mwidth] += 1
		except: pass
	if i%10 == 0:
		print(i)

tiff.imsave('20190917-markers-medfilt-regpeaks.tif',markers.astype(np.uint16))


##############
'''
PLOTTING BLOB TRAJECTORIES IN plotly
'''
##############
import pandas as pd
x,y,z,color = [],[],[],[]

for j in range(num_neurons):
	for i in range(999):
		try:
			c,b,a = s.threads[j].get_position_t(i)
			x.append(a)
			y.append(b)
			z.append(c)
		except: 
			if x:
				x.append(x[-1])	
				y.append(y[-1])	
				z.append(z[-1])
			else:
				x.append(0)
				y.append(0)
				z.append(0)	
		color.append(j)
'''
for i in range(num_neurons):
	x.append(list(trajectories[0,i,:]))
	y.append(list(trajectories[0,i,:]))
	z.append(list(trajectories[0,i,:]))
	color.append([i]*999)

flatten = lambda l: [item for sublist in l for item in sublist]
x = flatten(x)
y = flatten(y)
z = flatten(z)
color = flatten(color)
'''
df = pd.DataFrame(dict(
	X = x,
	Y = y,
	Z = z,
	color = color
))



fig = px.line_3d(df, x='X', y='Y', z='Z', color="color")
po.plot(fig, auto_open=True, filename='Trajectories-Flocking.html')

### Plot color by t
color = []
for j in range(999):
	for i in range(num_neurons):
		color.append(i)

fig = px.line_3d(df, x='X', y='Y', z='Z', color="color")
fig.update_layout(
	scene = dict(
		aspectratio=dict(x=950*0.1639/26,y=288*0.1639/26,z=26/26)
	)
)
po.plot(fig, auto_open=True, filename='Trajectories-Flocking-byt.html')


'''
im.t = 0
timeseries = []
for i in range(50):
	im1 = im.get_t()
	timeseries.append(im1[tuple(s.threads[20].positions[i])])

im.t = 0
timeseries = []
'''
'''
im.t = 0
width = 15
num_neurons = len(s.threads)
a = np.zeros((num_neurons,999,2*width,2*width))
for i in range(999):
	im1 = im.get_t()
	for j in range(num_neurons):
		try:
			x,y,z = s.threads[j].get_position_t(i)
			a[j,i] = im1[x,y-width:y+width,z-width:z+width]
		except: pass
	if i%10 == 0:
		print(i)
a = a.astype(np.uint16)
for i in range(num_neurons):
	tiff.imsave('neurons/tifs/neuron'+ "{:03d}".format(i) + '.tif',a[i])

	#timeseries.append(im1[tuple(s.threads[20].positions[i])])
'''
'''
im.t = 0
MIP = np.zeros(tuple([999]) + im.sizexy)
for i in range(999):
	MIP[i] = np.max(im.get_t(suppress_output = False),axis = 0)
tiff.imsave('20190917-MIP.tif',MIP.astype(np.uint16))




'''


'''
import pdb
pdb.set_trace()








from Threads import *
import tifffile as tiff
from multipagetiff import *
from rainbow_tools import *


im = MultiPageTiff('/Users/stevenban/Documents/Data/20190917')
im.set_frames([0,1,2,3,4,5,6,7,8,9,10,11])
im.numz = 20
import pickle
s = pickle.load( open( "20190917.obj", "rb" ) )


a = []
for i in range(len(s.threads)):
	a.append(s.threads[i].t[0])

'''
'''
im.t = 10
im2 = im.get_t()

#im2 = tf.get_t()
im2 = gaussian3d(im2)
peaks2 = findpeaks3d(im2 * np.array(im2 > np.quantile(im2,.99)))


s.reel(peaks2)


t = Threads(peaks1, im1)






reg = deformable_registration(**{ 'X': peaks2, 'Y': peaks1 })
reg.register()



reg.TY	



incoming_points = peaks2
registered_points = reg.TY

diff = np.zeros((len(incoming_points),len(registered_points)))
for i in range(len(incoming_points)):
	for j in range(len(registered_points)):
		diff[i,j] = np.linalg.norm(incoming_points[i]-registered_points[j])

diff = diff.T
### Match current points to registered incoming points ###
matchings = []
for i in range(diff.shape[0]): # iterate over points that already exist 
	matchings.append([i,np.where(diff[i,:] == np.min(diff[i,:]))[0][0]])

	#diff[i,:] = 10000
	diff[:,np.where(diff[i,:] == np.min(diff[i,:]))[0][0]] = 10000



from Threads import *

t = Thread([1,2,3])
t.update_position([1,2,3])
t.update_position([1.1,2.1,3.1],3)
t.update_position([1,2,3])

t.get_position_mostrecent()
t.get_position_t(0)
t.get_position_t(1)
t.get_position_t(2)
t.get_position_t(3)
'''











