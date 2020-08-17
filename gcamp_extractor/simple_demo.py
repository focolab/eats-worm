import napari
import json
import numpy as np
import sys
from skimage import io
from sklearn import mixture

np.set_printoptions(threshold=sys.maxsize)

worm = io.imread("/media/jack/0010-1820/steven_rainbow/20181001-col00-0.tif")
channels = [worm[:, i, :, : ] for i in range(worm.shape[1])]

with napari.gui_qt():
    viewer = napari.Viewer()
    panneuronal_channel = channels[3]
    viewer.add_image(panneuronal_channel, blending='additive', interpolation='bicubic', colormap='gray', name='panneuronal')

    with open("/media/jack/0010-1820/steven_rainbow/dicts/neuron2pixel.json") as f:
        rois = json.load(f)
        neuron = rois["ADAL"]
        neuron_pixels = np.zeros(panneuronal_channel.shape)
        neuron_pixels[neuron["z"], neuron["x"], neuron["y"]] = 1
        viewer.add_image(neuron_pixels, name="ADAL", blending='additive', colormap="magenta")

        X = np.transpose(np.vstack((neuron["z"], neuron["x"], neuron["y"])))
        num_components = 1
        gmm = mixture.GaussianMixture(num_components)
        gmm.fit(X)

        center = gmm.means_[0].astype(int)
        covariance = gmm.covariances_[0]
        s = 7.815 # 95% chi-square prob from https://people.richland.edu/james/lecture/m170/tbl-chi.html
        eigenvalues, eigenvectors = np.linalg.eig(covariance)
        radii = np.sqrt(s * eigenvalues)

        # use matrix equation from https://math.stackexchange.com/questions/1403126/what-is-the-general-equation-equation-for-rotated-ellipsoid
        radial_length_upper_bound = np.max(radii).astype(int) + 1
        indices = np.indices(neuron_pixels.shape).transpose(1, 2, 3, 0)
        x_minus_v = indices - center
        rotated = np.zeros(x_minus_v.shape)
        for i in range(center[0] - radial_length_upper_bound, center[0] + radial_length_upper_bound + 1):
            for j in range(center[1] - radial_length_upper_bound, center[1] + radial_length_upper_bound + 1):
                for k in range(center[2] - radial_length_upper_bound, center[2] + radial_length_upper_bound + 1):
                    rotated[i, j, k] = np.dot(covariance, x_minus_v[i,j,k])
        gmm_ellipsoid = np.zeros(neuron_pixels.shape)
        for i in range(center[0] - radial_length_upper_bound, center[0] + radial_length_upper_bound + 1):
            for j in range(center[1] - radial_length_upper_bound, center[1] + radial_length_upper_bound + 1):
                for k in range(center[2] - radial_length_upper_bound, center[2] + radial_length_upper_bound + 1):
                    gmm_ellipsoid[i, j, k] = np.dot(x_minus_v[i,j,k].T, rotated[i,j,k])
        viewer.add_image( (gmm_ellipsoid > 0).astype(int) * (gmm_ellipsoid <= s ** 3).astype(int), name="gmm ellipsoid", blending='additive', colormap="cyan")

        # region_of_interest = gmm_ellipsoid[center[0] - radial_length_upper_bound: center[0] + radial_length_upper_bound + 1, center[1] - radial_length_upper_bound: center[1] + radial_length_upper_bound + 1, center[2] - radial_length_upper_bound: center[2] + radial_length_upper_bound + 1]
        # print(np.where(region_of_interest <= 60))
        # print(region_of_interest <= s ** 2)
        # print(np.min(neuron_pixels), np.max(neuron_pixels))
        # print(np.min(panneuronal_channel), np.max(panneuronal_channel))
        # raise(Exception)