import numpy as np
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
    for i in range(e.t):
        im1 = e.im.get_t(t=i)
        im1_unfiltered = copy.deepcopy(im1)
        im1 = medFilter2d(im1)
        im1 = gaussian3d(im1,e.gaussian)

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
        drawn = []

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
            X = np.transpose(np.vstack((region_locations[0], region_locations[1], region_locations[2])))
            if X.shape[0] > 1:
                for num_components in range(1, 3):
                    i_gauss = mixture.GaussianMixture(num_components)
                    i_gauss.fit(X)
                    if i_gauss.bic(X) < bic:
                        bic = i_gauss.bic(X)
                        gauss = i_gauss
                gauss_peaks.append(gauss.means_)

                if i in volumes_to_output:
                    colors = [tuple(np.random.randint(256, size=3)) for region in range(3 * num_bright_regions)]
                    for z in range(im1_unfiltered.shape[0]):
                        img = np.array(im1_unfiltered[z])
                        img = cv2.normalize(img, img, 65535, 0, cv2.NORM_MINMAX)
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                        drawn.append(img)
                        img = drawn[z]
                        X_z = X[np.where(X[:, 0] == z)]
                        if X_z.shape[0] > 0:
                            predictions = gauss.predict(X_z)
                            for x, pred in zip(X_z, predictions):
                                color = colors[2 * bright_region_label + pred]
                                img[x[1], x[2]] = np.array([65535 // 255 * color[0], 65535 // 255 * color[1], 65535 // 255 * color[2]])
                        filtered, unfiltered = np.array(im1[z]), np.array(im1_unfiltered[z])
                        filtered = cv2.normalize(filtered, filtered, 65535, 0, cv2.NORM_MINMAX)
                        unfiltered = cv2.normalize(unfiltered, unfiltered, 65535, 0, cv2.NORM_MINMAX)
                        cv2.imwrite('{}_{}_labeled.png'.format(i, z), img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                        cv2.imwrite('{}_{}_filtered.png'.format(i, z), filtered, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                        cv2.imwrite('{}_{}_unfiltered.png'.format(i, z), unfiltered, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        
        if not e.suppress_output:
            print('\r' + 'Frames Processed: ' + str(i+1)+'/'+str(e.t), sep='', end='', flush=True)
    
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
