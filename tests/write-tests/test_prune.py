
from eats_worm import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from fastcluster import linkage

# load an extractor
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
 "quantile": 0.99,
 "reg_peak_dist": 7,
 "anisotropy": [15,1,1],
 "blob_merge_dist_thresh": 7,
 "register_frames": True,
 "predict": False,
 "regen_mft": False,
 "3d": False,
 "regen": False
}

'''
e = Extractor(**args_6)
e.calc_blob_threads()
e.quantify()
e.spool.make_allthreads()
e.save_threads()

e = Extractor(**args_17)
e.calc_blob_threads()
e.quantify()
e.spool.make_allthreads()
e.save_threads()
'''



e = load_extractor(args_6['root'] + 'extractor-objects-hold')


e = load_extractor(args_17['root'])
c = Curator(e, window = 100)



ec = load_curated_extractor(args_6['root'])




c = load_curate_json(args_6['root'] + 'extractor-objects-hold/')

compare(e, metric_test, load_curate_json(args_6['root']))


for i in range(e.t):
    apply_dvec(e.spool.threads[0].positions,e.spool.dvec, origin = i)
    if i % 100 == 0:
        print(i)


dPosDvec = np.zeros((e.t-1,len(e.spool.threads)))

for i in range(len(e.spool.threads)):
    dvec = np.diff(e.spool.threads[i].positions, axis = 0)
    dPosDvec[:,i] = np.linalg.norm(e.spool.dvec - dvec, axis = 1)
import matplotlib.pyplot as plt

plt.imshow(dPosDvec.T, aspect='auto')
plt.show()














'''


def metric(extractor):

    for i in range(1): #range(len(extractor.spool.threads))
        pos = extractor.spool.threads[i].positions
        dvec = extractor.spool.dvec

        for j in range(pos.shape[0]):
            
'''
curate = load_curate_json(args_6['root'] + 'extractor-objects-hold')




def correlation(metric, extractor, curate):

    results = metric(extractor)
    gt = get_curate_ans(extractor, curate)
    return np.corrcoef(results, gt)




correlation(metric_maxdiff, e, curate)
correlation(metric_numfound, e, curate)
correlation(super_mega_meandiff,e,curate)




ans = maxdiff_threshold(e, 4)
compare(e, ans, curate)




def maxdiff_threshold(extractor, thresh):
    maxdiff = metric_maxdiff(extractor)
    ans = maxdiff <= thresh
    return ans


def apply_dvec(positions, dvec, origin = 0):
    """
    Applies displacement vector to original positions around some origin. default is the beginning
    """
    
    # basic convert to numpy array
    positions = np.array(positions)
    dvec = np.array(dvec)

    # insert row of 0s to beginning of dvec (cus first element always has 0 offset from itself)
    dvec = np.insert(dvec, 0, [0,0,0], axis = 0)


    ## create hypothetical position array composed of just the first element
    newpos = np.zeros((positions.shape))
    newpos[:] = positions[0]

    # apply dvec to the positions
    newpos += dvec

    # center around 'anchor'/'origin' position 

    newpos += positions[origin] - newpos[origin]

    return newpos


def super_mega_meandiff(extractor):
    ans = np.zeros((len(extractor.spool.threads)))

    for i in range(len(extractor.spool.threads)):
        tmp = 0
        for j in range(len(extractor.spool.threads[i].positions)):
            newpos = apply_dvec(extractor.spool.threads[i].positions, e.spool.dvec, origin = j)
            newpos = newpos-extractor.spool.threads[i].positions
            tmp += np.linalg.norm(newpos, axis = 1).sum()
        ans[i] = tmp
        print('\r' + 'Threads Processed: ' + str(i+1)+'/'+str(len(extractor.spool.threads)), sep='', end='', flush=True)
    return ans


def metric_maxdiff(extractor):
    ans = np.zeros((len(extractor.spool.threads)))

    for i in range(len(extractor.spool.threads)):
        dvec = np.diff(extractor.spool.threads[i].positions, axis = 0)
        ans[i] = np.abs(extractor.spool.dvec - dvec).max()

    return ans



def metric_numfound(extractor):
    numfound = np.zeros((len(extractor.spool.threads)))


    for i in range(len(extractor.spool.threads)):

        numfound[i] = extractor.spool.threads[i].found.sum()

        #dvec = np.diff(extractor.spool.threads[i].positions, axis = 0)
        #numfound[i] = extractor.t - (extractor.spool.dvec - dvec == np.array([0,0,0])).all(-1).sum() 

    #numfound = numfound/numfound.max()

    return numfound


def metric_test(extractor):

    a = len(extractor.spool.threads)
    results = np.zeros((a))
    return results


def get_curate_ans(extractor, curate):

    keys = list(curate.keys())
    destroy = []
    for i in range(len(keys)):
        try:
            keys[i] = int(keys[i])
        except: 
            destroy.append(i)
    ## remove keys that aren't numeric
    destroy.sort(reverse = True)
    for i in range(len(destroy)):
        keys.pop(destroy[i])

    keys.sort()
    ans = np.zeros((len(keys)))

    for i in range(len(ans)):
        if curate.get(str(keys[i])) == 'trash':
            ans[i] = 0
        else:
            ans[i] = 1

    return ans

def compare(extractor,results, curate):
    ans = get_curate_ans(extractor, curate)
    tn = 0
    tp = 0
    fn = 0
    fp = 0
    for i in range(len(results)):
        a = (ans[i], results[i])
        if a == (0,0):
            tn += 1
        elif a == (1,1):
            tp += 1
        elif a == (1,0):
            fn += 1
        else:
            fp += 1
    return tp, tn, fp, fn



def compare_func(extractor, func, curate):
    results = func(extractor)
    ans = get_curate_ans(extractor, curate)
    results = np.array(results).reshape((results.size))

    tn = 0
    tp = 0
    fn = 0
    fp = 0
    for i in range(len(results)):
        a = (ans[i], results[i])
        if a == (0,0):
            tn += 1
        elif a == (1,1):
            tp += 1
        elif a == (1,0):
            fn += 1
        else:
            fp += 1
    return tp, tn, fp, fn




def load_curated_extractor(path):

    # Load the extractor
    e = load_extractor(path)
    
    curate = load_curate_json(path)
    if curate == 0:
        return 0


    keys = list(curate.keys())
    destroy = []
    for i in range(len(keys)):
        try:
            keys[i] = int(keys[i])
        except: 
            destroy.append(i)
    ## remove keys that aren't numeric
    destroy.sort(reverse = True)
    for i in range(len(destroy)):
        keys.pop(destroy[i])

    keys.sort(reverse = True)


    # if the keys link to trash, delete them
    for key in keys:
        key = str(key)

        if curate.get(key) == 'trash':
            e.spool.threads.pop(int(key))
            e.timeseries = np.delete(e.timeseries, int(key), axis = 1)
            e.spool.make_allthreads()

    return e

def load_curate_json(path):
    c = get_curate_filename(path)


    # do some exception checking
    if len(c) == 0: 
        print('No curate.json found')
        return 0
    elif len(c) != 1:
        print("Multiple curate.json found")
        return 0
    else:

        ## get keys and reverse-sort
        with open(c[0],'r') as f:
            curate = json.load(f)
    return curate


def get_curate_filename(path):
    filelist = getListOfFiles(path)

    r = []

    for file in filelist:
        if 'curate.json' in file:
            r.append(file)
    return r

def getListOfFiles(dirName):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
                
    return allFiles









'''Processing Script

from eats_worm import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from fastcluster import linkage

# load an extractor
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

e = Extractor(**args_6)
e.calc_blob_threads()
e.quantify()
e.spool.make_allthreads()
e.save_threads()


'''










