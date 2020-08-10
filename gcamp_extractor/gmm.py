import numpy as np
import napari
import scipy.spatial
import copy
import os
from .segfunctions import *
from sklearn import mixture
import json
import matplotlib.pyplot as plt
from scipy.ndimage import label, generate_binary_structure

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
    for i in range(e.t):
        im1 = e.im.get_t(t=i)
        im1_unfiltered = copy.deepcopy(im1)
        im1 = medFilter2d(im1)
        im1 = gaussian3d(im1,e.gaussian)
        unfiltered.append(im1_unfiltered)
        filtered.append(im1)
        drawn.append(np.zeros(im1_unfiltered.shape + (3,)))

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
            X = np.transpose(np.vstack((region_locations[0], region_locations[1], region_locations[2], region_pixel_values)))
            if X.shape[0] > 1:
                iters_since_best_bic = 0
                num_components = 1
                while iters_since_best_bic < 2 and X.shape[0] >= num_components:
                    i_gauss = mixture.GaussianMixture(num_components)
                    i_gauss.fit(X)
                    if i_gauss.bic(X) < bic:
                        bic = i_gauss.bic(X)
                        gauss = i_gauss
                        iters_since_best_bic = 0
                    else:
                         iters_since_best_bic += 1
                    num_components += 1
                gauss_peaks.append(gauss.means_)

                if X.shape[0] > 0:
                    predictions = gauss.predict(X)
                    roi_points = [[] for component in range(len(set(predictions)))]
                    for x, pred in zip(X, predictions):
                        roi_points[pred].append([i, x[0], x[1], x[2]])
                    points.extend(roi_points)
                # filtered, unfiltered = np.array(im1[z]), np.array(im1_unfiltered[z])
                # filtered = cv2.normalize(filtered, filtered, 65535, 0, cv2.NORM_MINMAX)
                # unfiltered = cv2.normalize(unfiltered, unfiltered, 65535, 0, cv2.NORM_MINMAX)

        
        if not e.suppress_output:
            print('\r' + 'Frames Processed: ' + str(i+1)+'/'+str(e.t), sep='', end='', flush=True)
    
    with napari.gui_qt():
        viewer = napari.Viewer()
        colors = [tuple(np.random.random(size=3)) for roi in range(len(points))]
        drawn = np.array(drawn)
        for i in range(len(points)):
            roi = np.array(points[i])
            roi_t = roi.T
            drawn[roi_t[0], roi_t[1], roi_t[2], roi_t[3], :] += colors[i]
            # viewer.add_points(roi, opacity=0.25, size=1, face_color=[colors[i]], symbol='square')
        viewer.add_image(drawn, scale=[1, 5, 1, 1], opacity=0.65, blending='additive')
        filtered = np.array(filtered)
        viewer.add_image(filtered, scale=[1, 5, 1, 1], blending='additive')
        unfiltered = np.array(unfiltered)
        viewer.add_image(unfiltered, scale=[1, 5, 1, 1], blending='additive')

    
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
