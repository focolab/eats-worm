import numpy as np
import scipy.spatial
import copy
import os
from .segfunctions import *
from sklearn import mixture
import json
import matplotlib.pyplot as plt
from scipy.ndimage import label, generate_binary_structure

def do_fitting(e):
    """
    1: do brightness thresholding
    2: for each volume, separate bright pixels into labeled contiguous 3d regions
    3: for each bright region, fit gmms with 1 and 2 components, and keep the fitting with lower bic
    4: if volume index in volumes_to_output, for each bright region, color pixels according to the predictions 
       of the associated gmm and save resulting image
    """
    num_components = {'peaks': [], 'gauss': []}
    radii_lengths = {}
    major_axis_orientations = {}
    for i in range(e.t):
        im1 = e.im.get_t(t=i)
        im1_unfiltered = copy.deepcopy(im1)
        im1 = medFilter2d(im1)
        im1 = gaussian3d(im1,e.gaussian)

        gauss_peaks = []
        radii_lengths[i] = []
        major_axis_orientations[i] = []
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
                radii_lengths[i] += get_radii_lengths(gauss)
                major_axis_orientations[i] += get_major_axis_orientations(gauss)
        
        if not e.suppress_output:
            print('\r' + 'Frames Processed: ' + str(i+1)+'/'+str(e.t), sep='', end='', flush=True)

    f = open("radii_lengths.json","w")
    f.write(json.dumps(radii_lengths))
    f.close()

    f = open("major_axis_orientations.json","w")
    f.write(json.dumps(major_axis_orientations))
    f.close()
    
# implemented following guidance at https://www.visiondummy.com/2014/04/draw-error-ellipse-representing-covariance-matrix/
def get_radii_lengths(gmm):
    radii_lengths = []
    s = 7.815 # 95% chi-square prob from https://people.richland.edu/james/lecture/m170/tbl-chi.html
    covariances = gmm.covariances_
    for component in covariances:
        eigenvalues, eigenvectors = np.linalg.eig(component)
        radii_lengths.append((2 * np.sqrt(s * eigenvalues)).tolist())
    return radii_lengths
    
def get_major_axis_orientations(gmm):
    orientations = []
    covariances = gmm.covariances_
    for component in covariances:
        eigenvalues, eigenvectors = np.linalg.eig(component)
        orientations.append(eigenvectors[:, np.argmax(eigenvalues)].tolist())
    return orientations

def visualize_radii_lengths(input_file):
    rounded_major_axis_lengths = []
    rounded_major_to_secondary_axis_ratios = []
    with open(input_file) as json_file:
        data = json.load(json_file)
        for t in data.keys():
            for component in data[t]:
                sorted_radii_lengths = sorted(component, reverse=True)
                rounded_major_axis_lengths.append(round(sorted_radii_lengths[0]))
                rounded_major_to_secondary_axis_ratios.append(round(sorted_radii_lengths[0]/sorted_radii_lengths[1]))

    plt.hist(rounded_major_axis_lengths, bins=len(set(rounded_major_axis_lengths)))
    plt.xlim(0, max(rounded_major_axis_lengths))
    plt.title("major axis length across all components of all volumes")
    plt.xlabel("length in pixels (rounded to nearest int)")
    plt.ylabel("count")
    plt.savefig("radii_lengths.png")
    plt.clf()

    plt.hist(rounded_major_to_secondary_axis_ratios, bins=len(set(rounded_major_axis_lengths)))
    plt.xlim(0, max(rounded_major_to_secondary_axis_ratios))
    plt.title("ratio of major to secondary axis length across all components of all volumes")
    plt.xlabel("ratio (rounded to nearest int)")
    plt.ylabel("count")
    plt.savefig("major_to_secondary_axis_ratios.png")
    plt.clf()

    print("counts of ratios of major to secondary axis length across all components of all volumes:")
    rounded_major_to_secondary_axis_ratio_counts = {value: rounded_major_to_secondary_axis_ratios.count(value) for value in sorted(set(rounded_major_to_secondary_axis_ratios))}
    for k, v in rounded_major_to_secondary_axis_ratio_counts.items():
        print(k, v)

def visualize_major_axis_orientations(input_file):
    all_eigenvectors = []
    z, x, y = [], [], []
    with open(input_file) as json_file:
        data = json.load(json_file)
        for t in data.keys():
            timepoint_eigenvectors = []
            for unit_eigenvector in data[t]:
                all_eigenvectors.append(unit_eigenvector)
                timepoint_eigenvectors.append(unit_eigenvector)
            timepoint_eigenvectors = np.array(timepoint_eigenvectors)
            timepoint_std = np.std(np.array(timepoint_eigenvectors), axis=0)
            z.append(timepoint_std[0])
            x.append(timepoint_std[1])
            y.append(timepoint_std[2])
    print(np.std(np.array(all_eigenvectors), axis=0))
    print(np.mean(all_eigenvectors, axis=0))

    plt.plot(z, "-r", label="z")
    plt.plot(x, "-g", label="x")
    plt.plot(y, "-b", label="y")
    plt.title("std dev of ellipsoid major axis orientation")
    plt.xlabel("time point")
    plt.ylabel("std dev")
    plt.legend()
    plt.savefig("major_axis_orientations.png")
    plt.clf()