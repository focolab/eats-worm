import napari
import json
import numpy as np
import sys
from scipy import linalg
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
        print([(label, len(rois[label]["x"])) for label in rois["neurons"]])
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

        A = covariance
        # calculate eigenvectors and eigenvalues
        values, vectors = np.linalg.eig(A)
        # multiply values by sqrt s
        values = 1. / values
        # create matrix from eigenvectors
        Q = vectors
        # create inverse of eigenvectors matrix
        R = np.linalg.inv(Q)
        # create diagonal matrix from eigenvalues
        L = np.diag(values)
        # reconstruct the original matrix
        A = Q.dot(L).dot(R)
        covariance = A

        eigenvalues, eigenvectors = np.linalg.eig(covariance)

        radii = np.sqrt(1. / eigenvalues)

        # use matrix equation from https://math.stackexchange.com/questions/1403126/what-is-the-general-equation-equation-for-rotated-ellipsoid
        radial_length_upper_bound = np.max(radii).astype(int) + 10
        indices = np.indices(neuron_pixels.shape).transpose(1, 2, 3, 0)
        x_minus_v = indices - center
        rotated = np.zeros(x_minus_v.shape)
        for i in range(center[0] - radial_length_upper_bound, center[0] + radial_length_upper_bound + 1):
            for j in range(center[1] - radial_length_upper_bound, center[1] + radial_length_upper_bound + 1):
                for k in range(center[2] - radial_length_upper_bound, center[2] + radial_length_upper_bound + 1):
                    rotated[i, j, k] = np.dot(covariance, np.array([i, j, k]) - center)
        gmm_ellipsoid = np.zeros(neuron_pixels.shape)
        for i in range(center[0] - radial_length_upper_bound, center[0] + radial_length_upper_bound + 1):
            for j in range(center[1] - radial_length_upper_bound, center[1] + radial_length_upper_bound + 1):
                for k in range(center[2] - radial_length_upper_bound, center[2] + radial_length_upper_bound + 1):
                    gmm_ellipsoid[i, j, k] = np.dot( (np.array([i, j, k]) - center).T, rotated[i,j,k])
        
        # gmm_ellipsoid = gmm_ellipsoid / np.product(eigenvalues)

        inside = 0
        for observation in X:
            z, x, y = observation
            if gmm_ellipsoid[z, x, y] > 0 and gmm_ellipsoid[z, x, y] <= s:
                inside += 1
        print("percent covered: ", 100 * inside / len(X))
        print("precision: ", inside / np.count_nonzero(gmm_ellipsoid) )
        viewer.add_image( (gmm_ellipsoid > 0).astype(int) * (gmm_ellipsoid <= s).astype(int), name="gmm ellipsoid", blending='additive', colormap="cyan")
