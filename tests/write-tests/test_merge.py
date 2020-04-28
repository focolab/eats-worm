from gcamp_extractor import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from fastcluster import linkage
import scipy.cluster



args_6 = {
	"root": "/Users/stevenban/Desktop/20191104_KP_FC083_worm6_gcamp6s_1/", 
	"numz": 10, 
	"frames": [0, 1, 2, 3, 4, 5], 
	"offset": 23, 
	#"t": 1000, 
	"gaussian": [25, 2, 3, 1], 
	"quantile": 0.98, 
	"reg_peak_dist": 4, 
	"anisotropy": [10, 1, 1], 
	"blob_merge_dist_thresh": 5, 
	"register_frames": False, 
	"predict": False, 
	"regen_mft": False, 
	"3d": False, 
	"regen": False
}

args_17 = {"root": "/Users/stevenban/Desktop/20191203_KP_FC083_worm17_gcamp6f_1/",
 "numz": 13,
 "frames": [0,1,2,3,4,5,6,7,8],
 "offset": 13,
 "gaussian": [51,8,3,1],
 "quantile": 0.985,
 "reg_peak_dist": 7,
 "anisotropy": [15,1,1],
 "blob_merge_dist_thresh": 7,
 "register_frames": True,
 "predict": False,
 "regen_mft": False,
 "3d": True,
 "regen": False
}

e = load_extractor(args_17['root'] + 'extractor-objects')

e = load_extractor(args_6['root'] + 'extractor-objects-hold')
'''
import os
os.mkdir('tests/images')
os.mkdir('tests/images/merging')
'''

#dmatlist = calc_dist_mat_list(e, threads_by_z(e))





class hoverViewer:
	def __init__(self, e, indices, z,show = True):
		self.s = e.spool
		self.fig, self.ax = plt.subplots()

		self.volume = e.im.get_t(0)

		self.indices = indices[z]
		self.im = self.ax.imshow(self.volume[z])
		self.z = z

		self.positions = get_positions_by_index(e, self.indices)
		self.y = self.positions[:,0,1]
		self.x = self.positions[:,0,2]
		self.names = [str(i) for i in range(len(self.x))]
		self.scatter = plt.scatter(self.x,self.y,s=1,c='r')
		self.annotations = self.ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
					bbox=dict(boxstyle="round", fc="w"),
					arrowprops=dict(arrowstyle="->"))

		self.annotations.set_visible(False)
		self.fig.canvas.mpl_connect("motion_notify_event", self.hover)
		if show:
			plt.show()

	def update_annot(self,ind):
		pos = self.scatter.get_offsets()[ind["ind"][0]]
		self.annotations.xy = pos

		#### TODO EDIT THIS
		#text = '{},{}'.format(" ".join([self.names[n] for n in ind['ind']]), " ".join([str(self.x[n]) for n in ind['ind']]))
		text = '{},{}'.format(" ".join([str(n) for n in ind['ind']]), " ".join([str(self.x[n]) for n in ind['ind']]))
		#x = [self.x[n] for n in ind['ind']]


		'''
		text = "{}, {}".format(" ".join(list(map(str,ind["ind"]))), 
							   " ".join([names[n] for n in ind["ind"]]))
		'''
		self.annotations.set_text(text)
		#self.annotations.get_bbox_patch().set_facecolor(cmap(norm(c[ind["ind"][0]])))
		self.annotations.get_bbox_patch().set_alpha(0.4)


	def hover(self, event):
		vis = self.annotations.get_visible()
		if event.inaxes == self.ax:
			cont, ind = self.scatter.contains(event)
			if cont:
				self.update_annot(ind)
				self.annotations.set_visible(True)
				self.fig.canvas.draw_idle()
			else:
				if vis:
					self.annotations.set_visible(False)
					self.fig.canvas.draw_idle()

	def save_plot(self,path):
		#self.s = e.spool
		self.fig, self.ax = plt.subplots()

		self.volume = e.im.get_t(0)

		#self.indices = indices[z]
		self.im = self.ax.imshow(self.volume[self.z])
		self.positions = get_positions_by_index(e, self.indices)
		self.y = self.positions[:,0,1]
		self.x = self.positions[:,0,2]
		#self.names = [str(i) for i in range(len(self.x))]
		self.scatter = plt.scatter(self.x,self.y,s=1,c='r')
		'''
		self.annotations = self.ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
					bbox=dict(boxstyle="round", fc="w"),
					arrowprops=dict(arrowstyle="->"))

		self.annotations.set_visible(False)
		'''
		#plt.title('z = 10')
		plt.axis('off')
		plt.savefig(path,bbox_inches = 'tight',dpi=400)
		plt.close()

#h = hoverViewer(e,threads_by_z(e), 0, show = False)
#h.save_plot('test.png')












def get_positions_by_index(e, indices):
	positions = []
	for ndx in indices:
		positions.append(e.spool.threads[ndx].positions)
	return np.array(positions)


'''

x = np.random.rand(15)
y = np.random.rand(15)
names = np.array(list("ABCDEFGHIJKLMNO"))
c = np.random.randint(1,5,size=15)

norm = plt.Normalize(1,4)
cmap = plt.cm.RdYlGn

fig,ax = plt.subplots()

sc = plt.scatter(x,y,c=c, s=100, cmap=cmap, norm=norm)

annot = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
					bbox=dict(boxstyle="round", fc="w"),
					arrowprops=dict(arrowstyle="->"))
annot.set_visible(False)

def update_annot(ind):

	pos = sc.get_offsets()[ind["ind"][0]]
	annot.xy = pos
	text = "{}, {}".format(" ".join(list(map(str,ind["ind"]))), 
						   " ".join([names[n] for n in ind["ind"]]))
	annot.set_text(text)
	annot.get_bbox_patch().set_facecolor(cmap(norm(c[ind["ind"][0]])))
	annot.get_bbox_patch().set_alpha(0.4)


def hover(event):
	vis = annot.get_visible()
	if event.inaxes == ax:
		cont, ind = sc.contains(event)
		if cont:
			update_annot(ind)
			annot.set_visible(True)
			fig.canvas.draw_idle()
		else:
			if vis:
				annot.set_visible(False)
				fig.canvas.draw_idle()

fig.canvas.mpl_connect("motion_notify_event", hover)

plt.show()








'''


def calc_dist_mat(e: Extractor, indices: list) -> np.array:
	"""
	Calculates distance matrix among threads with indices specified

	Arguments:
		e : Extractor
			extractor object
		indices : list of ints
			list of indices corresponding to which threads are present for the distance matrix calculation
	"""
	
	# initialize distance matrix
	dmat = np.zeros((len(indices), len(indices)))

	# calculate dmat, non-diagonals only
	for i in range(len(indices)):
		for j in range(i+1, len(indices)):
			pos1 = e.spool.threads[indices[i]].positions
			pos2 = e.spool.threads[indices[j]].positions

			dmat[i,j] = np.linalg.norm(pos1 - pos2, axis = 1).mean()
	dmat = dmat + dmat.T

	return dmat


def calc_dist_mat_list(e: Extractor, indices: list) -> list:
	"""
	Calculates list of distance matrices 

	Arguments:
		e : extractor
			extractor object
		indices : list of list of ints
			list of list of indices made by threads_by_z
	"""
	# initialize dmat list
	dmatlist = []

	# iterate over z planes
	for i in range(len(indices)):
		dmat = calc_dist_mat(e, indices[i])
		dmatlist.append(dmat)
	return dmatlist







def threads_by_z(e : Extractor) -> list:
	"""
	Organizes thread indices by z plane

	Arguments:
		e : extractor
			extractor object
	"""

	# make object to store thread indices corresponding to z plane
	threads_by_z = [[] for i in e.frames]


	# iterate over threads, append index to threads_by_z
	for i in range(len(e.spool.threads)):
		z = int(e.spool.threads[i].positions[0,0])
		ndx = np.where(np.array(e.frames) == z)[0][0]

		threads_by_z[ndx].append(i)

	return threads_by_z

def seriation(Z,N,cur_index):
    '''
        input:
            - Z is a hierarchical tree (dendrogram)
            - N is the number of points given to the clustering process
            - cur_index is the position in the tree for the recursive traversal
        output:
            - order implied by the hierarchical tree Z
            
        seriation computes the order implied by a hierarchical tree (dendrogram)
    '''
    if cur_index < N:
        return [cur_index]
    else:
        left = int(Z[cur_index-N,0])
        right = int(Z[cur_index-N,1])
        return (seriation(Z,N,left) + seriation(Z,N,right))
    
def compute_serial_matrix(dist_mat,method="ward"):
    '''
        input:
            - dist_mat is a distance matrix
            - method = ["ward","single","average","complete"]
        output:
            - seriated_dist is the input dist_mat,
              but with re-ordered rows and columns
              according to the seriation, i.e. the
              order implied by the hierarchical tree
            - res_order is the order implied by
              the hierarhical tree
            - res_linkage is the hierarhical tree (dendrogram)
        
        compute_serial_matrix transforms a distance matrix into 
        a sorted distance matrix according to the order implied 
        by the hierarchical tree (dendrogram)
    '''
    N = len(dist_mat)
    flat_dist_mat = squareform(dist_mat)
    res_linkage = linkage(flat_dist_mat, method=method,preserve_input=True)
    res_order = seriation(res_linkage, N, N + N-2)
    seriated_dist = np.zeros((N,N))
    a,b = np.triu_indices(N,k=1)
    seriated_dist[a,b] = dist_mat[ [res_order[i] for i in a], [res_order[j] for j in b]]
    seriated_dist[b,a] = seriated_dist[a,b]
    
    return seriated_dist, res_order, res_linkage



def merge_threads(e, thresh = None):
	if thresh == None:
		thresh = 1.5*e.blob_merge_dist_thresh
	dmatlist = calc_dist_mat_list(e, threads_by_z(e))

	# create a list of list of blob thread indices, outer list corresponding to z plane blob is found in
	tbyz = threads_by_z(e)

	# create list object to contain lists of blob indices that have been merged together
	all_clusters = []

	# iterate over all frames in the image
	for i in range(len(e.frames)):
		# compute in-plane distance matrix
		sort_mat,b,c = compute_serial_matrix(dmatlist[i])

		# assign indices to clusters	
		cluster_array = scipy.cluster.hierarchy.fcluster(c, thresh, criterion='distance')

		# create create array to hold clusters within the z plane
		clusters = [[] for i in range(cluster_array.max())]

		# add blob thread index to cluster 
		for j in range(len(cluster_array)):
			clusters[cluster_array[j]-1].append(tbyz[i][j])

		# add clusters to list of clusters
		for j in range(len(clusters)):
			all_clusters.append(clusters[j])

	# calculate initial number of threads
	orig_len = len(e.spool.threads)

	# iterate over all clusters
	for i in range(len(all_clusters)):


		# iterate over all blob thread indices within a cluster
		for j in range(len(all_clusters[i])):
			btndx = all_clusters[i][j]
			if j == 0:
				avpos = e.spool.threads[btndx].positions
			else:
				avpos += e.spool.threads[btndx].positions

		# calculate average
		avpos = avpos/len(all_clusters[i])

		# initialize new blob thread with the position specified
		thread = Thread(maxt = e.spool.maxt)
		thread.positions = avpos

		e.spool.threads.append(thread)

	for i in range(orig_len):
		e.spool.threads.pop(0)


	return e

#def get_groups(distmat, thresh):

'''

for i in range(len(e.frames)):
	h = hoverViewer(e, threads_by_z(e), i, show = False)
	h.save_plot('tests/images/worm17/z=' + str(i) + '.png')




os.mkdir('tests/images/worm17/cluster')
for i in range(len(e.frames)):
	sort_mat,b,c = compute_serial_matrix(np.array(dmatlist[i] > e.blob_merge_dist_thresh))

	plt.imshow(1-sort_mat)
	plt.axis('off')
	plt.savefig('tests/images/worm17/cluster/threshclust_z=' + str(i) + '.png', bbox_inches='tight')

for i in range(6):
	sort_mat,b,c = compute_serial_matrix(dmatlist[i])

	plt.imshow(sort_mat < e.blob_merge_dist_thresh)
	plt.axis('off')
	plt.savefig('tests/images/worm17/cluster/clustthresh_z=' + str(i) + '.png', bbox_inches='tight')





plt.imshow(sort_mat < e.blob_merge_dist_thresh)
plt.show()

for i in range(len(dmatlist[0])):
	a = dmatlist[0][i]
	a.sort()
	print(i, a[a!=0].min())


'''


e = load_extractor(args_17['root'] + 'extractor-objects')
for i in range(len(e.frames)):
	h = hoverViewer(e, threads_by_z(e), i, show = False)
	h.save_plot('tests/images/worm6/z=' + str(i) + '.png')


e2 = merge_threads(e)
for i in range(len(e2.frames)):
	h = hoverViewer(e2, threads_by_z(e2), i, show = False)
	h.save_plot('tests/images/worm6/merged_z=' + str(i) + '.png')



dmatlist= calc_dist_mat_list(e, threads_by_z(e))
sorted_dmat, _, _ = compute_serial_matrix(dmatlist[1])
plt.imshow(sorted_dmat)
plt.axis('off')
plt.savefig('tests/images/worm6/distmat.png',dpi=400, bbox_inches = 'tight')
plt.close()


for i in range(len(e.frames)):
	h = hoverViewer(e, threads_by_z(e), i, show = False)
	h.save_plot('tests/images/worm6/merged_z=' + str(i) + '.png')




h = hoverViewer(e, threads_by_z(e), 0, show = True)

