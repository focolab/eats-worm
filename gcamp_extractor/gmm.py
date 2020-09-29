import numpy as np
import napari
import scipy.spatial
import copy
import os
from .segfunctions import *
from sklearn import mixture
from skimage import draw
import json
import matplotlib.pyplot as plt
from scipy.ndimage import label, generate_binary_structure
import sys
np.set_printoptions(threshold=sys.maxsize)

def get_radii_lengths(gmm_covariance):
    s = 7.815 # 95% chi-square prob from https://people.richland.edu/james/lecture/m170/tbl-chi.html
    eigenvalues, eigenvectors = np.linalg.eig(gmm_covariance)
    if not np.iscomplex(eigenvalues).any():
        return (2 * np.sqrt(s * eigenvalues)).tolist()
    else:
        return None
    
def get_major_axis_orientation(gmm_covariance):
    eigenvalues, eigenvectors = np.linalg.eig(gmm_covariance)
    if not np.iscomplex(eigenvalues).any():
        return eigenvectors[:, np.argmax(eigenvalues)].tolist()
    else:
        return None

# from https://stackoverflow.com/questions/7819498/plotting-ellipsoid-with-matplotlib
def draw_ellipsoid(shape, gmm_covariance, gmm_mean):
    zxy_cov = gmm_covariance
    center = gmm_mean.astype(int)
    indices = np.indices(shape).transpose(1,2,3,0)
    
    covariance = gmm_covariance
    s = 6.251 # 90% chi-square prob from https://people.richland.edu/james/lecture/m170/tbl-chi.html

    A = covariance
    # calculate eigenvectors and eigenvalues
    values, vectors = np.linalg.eig(A)
    # matrix equation expects reciprocals of squares of radial lengths
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

    if not np.iscomplex(eigenvalues).any():

        radii = np.sqrt(1. / eigenvalues)

        # use matrix equation from https://math.stackexchange.com/questions/1403126/what-is-the-general-equation-equation-for-rotated-ellipsoid
        radial_length_upper_bound = np.max(radii).astype(int) + 10
        x_minus_v = indices - center
        rotated = np.zeros(x_minus_v.shape)
        for i in range(max(0, center[0] - radial_length_upper_bound), min(indices.shape[0], center[0] + radial_length_upper_bound + 1)):
            for j in range(max(0, center[1] - radial_length_upper_bound), min(indices.shape[1], center[1] + radial_length_upper_bound + 1)):
                for k in range(max(0, center[2] - radial_length_upper_bound), min(indices.shape[2], center[2] + radial_length_upper_bound + 1)):
                    rotated[i, j, k] = np.dot(covariance, np.array([i, j, k]) - center)
        gmm_ellipsoid = np.zeros(shape)
        for i in range(max(0, center[0] - radial_length_upper_bound), min(indices.shape[0], center[0] + radial_length_upper_bound + 1)):
            for j in range(max(0, center[1] - radial_length_upper_bound), min(indices.shape[1], center[1] + radial_length_upper_bound + 1)):
                for k in range(max(0, center[2] - radial_length_upper_bound), min(indices.shape[2], center[2] + radial_length_upper_bound + 1)):
                    gmm_ellipsoid[i, j, k] = np.dot( (np.array([i, j, k]) - center).T, rotated[i,j,k])
        
        gmm_ellipsoid_pixels = (gmm_ellipsoid > 0).astype(int) * (gmm_ellipsoid <= s).astype(int)

        return gmm_ellipsoid_pixels
    else:
        return None


def do_fitting(e, volumes_to_output):
    """
    1: do brightness thresholding
    2: for each volume, separate bright pixels into labeled contiguous 3d regions
    3: for each bright region, fit gmms with 1 and 2 components, and keep the fitting with lower bic
    4: if volume index in volumes_to_output, for each bright region, color pixels according to the predictions 
       of the associated gmm and save resulting image
    """
    num_components = {'peaks': [], 'gauss': []}
    unfiltered = []
    drawn = []
    points = []
    filtered = []
    ellipsoids = []
    for i in range(e.t):
        im1 = e.im.get_t(t=i)
        im1_unfiltered = copy.deepcopy(im1)
        im1 = medFilter2d(im1)
        im1 = gaussian3d(im1,e.gaussian)
        unfiltered.append(im1_unfiltered)
        filtered.append(im1)
        drawn.append(np.zeros(im1_unfiltered.shape + (3,)))
        # ellipsoids.append(np.zeros(im1_unfiltered.shape + (3,)))

        gauss_peaks = []
        brightest_quantile = np.array(im1 > np.quantile(im1, 0.99))
        brightest_pixels = brightest_quantile * im1
        # structure = np.tile(generate_binary_structure(2,2), (3, 1, 1))
        structure = np.array(
            [[[False,  False, False],
              [False,  True,  False],
              [False,  False, False]],
             [[False,  True,  False],
              [ True,  True,  True],
              [False,  True,  False]],
             [[False,  False, False],
              [False,  True,  False],
              [False,  False, False]]],
            dtype=bool)
        brightest_regions, num_bright_regions = label(brightest_quantile, structure=structure)

        for bright_region_label in range(1, num_bright_regions + 1):
            region_locations = np.where(brightest_regions == bright_region_label)

            # this code gets the sub-volume containing the bright region
            #
            # lower_z_bound = np.min(region_locations[0])
            # upper_z_bound = np.max(region_locations[0])
            # lower_y_bound = np.min(region_locations[1])
            # upper_y_bound = np.max(region_locations[1])
            # lower_x_bound = np.min(region_locations[2])
            # upper_x_bound = np.max(region_locations[2])
            # region = brightest_pixels[lower_z_bound:upper_z_bound, lower_y_bound:upper_y_bound, lower_x_bound:upper_x_bound]

            bic = float('inf')
            gauss = None
            # X is the coordinates of the set of pixels assigned to the bright region
            region_pixel_values = im1_unfiltered[region_locations[0], region_locations[1], region_locations[2]]
            # X = np.transpose(np.vstack((region_locations[0], region_locations[1], region_locations[2], region_pixel_values)))
            X = np.transpose(np.vstack((region_locations[0], region_locations[1], region_locations[2])))
            if X.shape[0] > 1:
                iters_since_best_bic = 0
                num_components = 1
                # while iters_since_best_bic < 2 and X.shape[0] >= num_components:
                    # i_gauss = mixture.GaussianMixture(num_components)
                cov_prior = np.array([[1.,  0., 0.], [0., 6.25, 0.], [0., 0., 4.25]])
                gauss = mixture.BayesianGaussianMixture(n_components=min(X.shape[0], 10), covariance_type='tied', covariance_prior=cov_prior, max_iter=250)
                gauss.fit(X)
                    # if i_gauss.bic(X) < bic:
                    #     bic = i_gauss.bic(X)
                    #     gauss = i_gauss
                    #     iters_since_best_bic = 0
                    # else:
                    #      iters_since_best_bic += 1
                    # num_components += 1
                gauss_peaks.append(gauss.means_)

                if X.shape[0] > 0:
                    predictions = gauss.predict(X)
                    roi_points = [[] for component in range(len(gauss.means_))]
                    for x, pred in zip(X, predictions):
                        roi_points[pred].append([i, x[0], x[1], x[2]])
                    points.extend(roi_points)
                    for mean in gauss.means_:
                        ellipsoid = draw_ellipsoid(im1.shape, gauss.covariances_, mean)
                        if ellipsoid is not None:
                            ellipsoids.append(ellipsoid)
                # filtered, unfiltered = np.array(im1[z]), np.array(im1_unfiltered[z])
                # filtered = cv2.normalize(filtered, filtered, 65535, 0, cv2.NORM_MINMAX)
                # unfiltered = cv2.normalize(unfiltered, unfiltered, 65535, 0, cv2.NORM_MINMAX)

            print('\r' + 'Finished region {} of {} in frame {} of {} '.format(bright_region_label, num_bright_regions, i, e.t))
        
        if not e.suppress_output:
            print('\r' + 'Frames Processed: ' + str(i+1)+'/'+str(e.t), sep='', end='', flush=True)
    
    with napari.gui_qt():
        viewer = napari.Viewer()
        colors = [tuple(np.random.random(size=3)) for roi in range(len(points))]
        drawn = np.array(drawn)
        for i in range(len(points)):
            roi = np.array(points[i])
            roi_t = roi.T
            if roi_t.shape[0] == 4:
                drawn[roi_t[0], roi_t[1], roi_t[2], roi_t[3], :] += colors[i]
            # viewer.add_points(roi, opacity=0.25, size=1, face_color=[colors[i]], symbol='square')
        viewer.add_image(drawn, scale=[1, 5, 1, 1], opacity=0.65, blending='additive')
        filtered = np.array(filtered)
        viewer.add_image(filtered, scale=[1, 5, 1, 1], blending='additive')
        unfiltered = np.array(unfiltered)
        viewer.add_image(unfiltered, scale=[1, 5, 1, 1], blending='additive', visible=False)
        ellipsoids = np.array(ellipsoids)
        viewer.add_image(ellipsoids, scale=[1, 5, 1, 1], blending='additive', visible=False, colormap='cyan')

    
    # old plotting code. won't produce anything right now, but leaving here to adapt in near future because I hate looking up plotting syntax
    #
    # plt.plot(num_components['gauss'], 'ro', label='GMM (lowest BIC)')
    # plt.plot(num_components['peaks'], 'bo', label='peak finding')
    # plt.grid(True)
    # plt.title('Num components per frame')
    # plt.legend(loc='upper right')
    # plt.savefig('num_components.png')
    # num_components_json = json.dumps(num_components)
    # f = open("num_components.json","w")
    # f.write(num_components_json)
    # f.close()
    # print(num_components['peaks'])
    # print("gmm idenfified {}x as many clusters as peakfinding.".format(sum(num_components['gauss']) / sum(num_components['peaks'])))
