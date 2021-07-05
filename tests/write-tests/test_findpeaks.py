import numpy as np
from gcamp_extractor.segfunctions import *
from sklearn.datasets import make_blobs

# generate three points, use them as bright voxels in image, and verify that they are detected
def test_findpeaks2d():
    image = np.zeros((25, 25))
    samples, labels, blob_centers = make_blobs(n_samples=3, centers=3, n_features=2, center_box=(0, 20), return_centers=True)
    bright_voxels = np.sort(samples.astype(int), axis=0)
    for bright_voxel in bright_voxels:
        image[tuple(bright_voxel)] = 1
    image = np.expand_dims(image, axis=0)
    detected_centers = np.sort(findpeaks2d(image)[:,1:3], axis=0)
    assert(np.array_equal(bright_voxels, detected_centers.astype(int)))

# generate three points, use them as bright voxels in image, and verify that they are detected
def test_findpeaks3d():
    image = np.zeros((25, 25, 25))
    samples, labels, blob_centers = make_blobs(n_samples=3, centers=3, n_features=3, center_box=(0, 20), return_centers=True)
    bright_voxels = np.sort(samples.astype(int), axis=0)
    for bright_voxel in bright_voxels:
        image[tuple(bright_voxel)] = 1
    detected_centers = np.sort(findpeaks2d(image), axis=0)
    assert(np.array_equal(bright_voxels, detected_centers.astype(int)))

if __name__ == '__main__':
    test_findpeaks2d()
    test_findpeaks3d()