import sys
sys.path.append('/Users/stevenban/Documents/gcamp-extractor/gcamp-extractor')

from Extractor import *
from Threads import *
from segfunctions import *
from Curator import *
import json
#default_arguments['t']=20
#e = Extractor(**default_arguments)
e = load_extractor(default_arguments['root'])
#e.calc_blob_threads()
#e.quantify()
#e.save_self()

#c = Curator(e)



path = '/Users/stevenban/Documents/Data/20190917/binned/extractor-objects/curate.json'

with open(path) as f:
    curate = json.load(f)
            
counter = 0
for value in curate.values():
    if value == 'keep':
        counter += 1



t = [0.33 * n for n in range(999)]
fig,ax = plt.subplots(nrows = 50, ncols = 6, sharex = False, sharey = False,figsize=(20,9))
counter = 0
for i,row in enumerate(ax):
    for j,axs in enumerate(row):
        while curate.get(str(counter)) != 'keep' and counter < len(e.spool.threads):
            counter += 1
    #axs = fig.add_subplot(10,1,i+1)
        if counter < len(e.spool.threads):
            axs.plot(t, e.timeseries[:,counter],linewidth=0.65)
            axs.set_xlabel('')
            axs.set_ylabel(str(counter),fontsize=4)
            axs.spines['top'].set_visible(False)
            axs.spines['right'].set_visible(False)
            axs.spines['bottom'].set_visible(False)
            axs.spines['left'].set_visible(False)
            axs.get_xaxis().set_ticks([])
            axs.get_yaxis().set_ticks([])
            counter += 1
        else:
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
fig.savefig('timeseries/curated' + "{:03d}".format(i) + '.png',bbox_inches='tight',dpi=1000)
plt.close()



s = Spool()

for i in range(len(e.spool.threads)):
	if curate.get(str(i)) == 'keep':
		s.threads



num_neurons = len(e.spool.threads)
markers = np.zeros(tuple([999]) + e.im.sizexy,dtype=np.uint16)
mwidth = 2
for i in range(1):
    #im1 = im.get_t()
    for j in range(num_neurons):
    	if curate.get(str(j)) == 'keep':
	        
            pos = e.spool.threads[j].get_position_t(i).astype(int)
            markers[i,pos[1]-mwidth:pos[1]+mwidth,pos[2]-mwidth:pos[2]+mwidth] += 1
    if i%10 == 0:
        print(i)

tiff.imsave(e.root + 'extractor-objects/trashed_markers.tif',markers.astype(np.uint16))




source='/Users/stevenban/Documents/Data/20190917/binned/'

x = np.genfromtxt(source+'x.txt',dtype=int,delimiter=',')
y = np.genfromtxt(source+'y.txt',dtype=int,delimiter=',')
z = np.genfromtxt(source+'z.txt',dtype=int,delimiter=',')

markers_wb = np.zeros(tuple([999]) + e.im.sizexy,dtype=np.uint16)
for i in range(352):
	markers_wb[0,x[0,i]-mwidth:x[0,i]+mwidth,y[0,i]-mwidth:y[0,i]+mwidth] += 1


im1 = np.fliplr(markers_wb[0])
im2 = markers[0]

tiff.imsave('wb.tif',im1)
tiff.imsave('ge.tif',im2)

plt.imshow(im1,cmap='gray')
plt.axis('off')
plt.savefig('wb.png',dpi=1000,bbox_inches='tight')
plt.close()


plt.imshow(im2,cmap='gray')
plt.savefig('ge.png',dpi=1000,bbox_inches='tight')
plt.close()









