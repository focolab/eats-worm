import numpy as np
from gcamp_extractor.FilterSweeper import FilterSweeper

"""
This is a demo of using FilterSweeper for a numpy array of synthetic blobs
(with some white noise)
"""

def gen_image_stacks():
    """make image volume(s) of gaussian blobs plus noise
    
    returns
    -------
    stacks: list
        a list of 3D image volumes (ZYX order)
    """
    imsize = (41, 81)
    numT = 3
    numZ = 9
    b1 = dict(xy=[20,20], cov=[4,0,0,8], _int=20)
    b2 = dict(xy=[20,40], cov=[12,0,0,4], _int=15)
    blobs = [b1, b2]

    img = np.zeros(shape=imsize)
    for b in blobs:
        xy = np.asarray(b['xy'])
        cov = np.asarray(b['cov']).reshape(2,2)
        icov = np.linalg.inv(cov)

        # pixel coordinates, relative to the blob center
        axes = [range(n) for n in imsize]
        arrs = np.meshgrid(*axes, indexing='ij')
        arrs = [a.ravel() for a in arrs]
        vox = np.vstack(arrs).T
        x = (vox-xy)

        # evaluate gaussian
        px = np.sum(np.dot(x, icov)*x, 1)
        px = b['_int']*np.exp(-0.5*px)
        img += px.reshape(imsize)

    # stack this image numZ times
    data = np.zeros(shape=(numT, numZ, *imsize))
    for it in range(numT):
        for iz in range(numZ):
            data[it,iz] = img

    # add some noise
    data += np.random.randn(*data.shape)*0.5

    # have to convert datatype for the segtools machinery to work
    data_min = np.min(data.ravel())
    data_max = np.max(data.ravel())
    data_del = data_max-data_min
    data = (data-data_min)/data_del * 254
    data = data.astype(np.uint8)
#    print('min/max/del=', data_min, data_max, data_del)
#    print(data.ravel()[:40])

    return [im for im in data]


#### run it
FilterSweeper(stacks=gen_image_stacks()).sweep_parameters()
