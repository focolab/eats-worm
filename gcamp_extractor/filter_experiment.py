import numpy as np
import scipy.spatial
import copy
import napari
import os
import json
import matplotlib.pyplot as plt

from .segfunctions import *
from sklearn import mixture
from dask import array as da
from scipy.ndimage import label, generate_binary_structure

med_filter_sizes = [3,5,7,9]

width_x_width_y_values = [7, 15, 19, 27]
width_z_values = [3, 5, 7, 9]
sigma_x_sigma_y_values = [1, 3, 6, 8]
sigma_z_values = [1, 2, 3, 4]

def threshold(image, quantile):
    arr = da.from_array(image, chunks=image.shape)
    return arr > np.quantile(image, quantile)

def gaussian_filter(image, params):
  arr = da.from_array(image)
  return gaussian3d(image, params)

def median_filtered(image, size):
  arr = da.from_array(image)
  return medFilter2d(image)

def do_experiment(e):
    """
    do ...
    """
    time_points = []
    filters = []
    # for i in range(e.t):
    for i in range(15):
        im1 = e.im.get_t(t=i)
        im1_unfiltered = copy.deepcopy(im1)
    
        # all_thresholds = da.stack([np.array(im1_unfiltered > np.quantile(im1_unfiltered, quantile)) for quantile in np.arange(0.75, 1., 0.005)])
        # all_medians = da.stack([medFilter2d(im1_unfiltered, size) for size in np.arange(3, 7, 2)])
        # all_gaussians = da.stack([gaussian3d(im1_unfiltered, (width_x_width_y_values[gauss_arg], width_z_values[gauss_arg], sigma_x_sigma_y_values[gauss_arg], sigma_z_values[gauss_arg])) for gauss_arg in np.arange(4)])
        all_filters = da.stack([[np.array(gaussian_filtered > np.quantile(gaussian_filtered, quantile)) for quantile in np.arange(0.875, 1., 0.005)] for gaussian_filtered in
          [gaussian3d(medFilter2d(im1), (width_x_width_y_values[gauss_arg], width_z_values[gauss_arg], sigma_x_sigma_y_values[gauss_arg], sigma_z_values[gauss_arg])) for gauss_arg in np.arange(4)]
        ])
        time_points.append(im1_unfiltered)
        filters.append(all_filters)
    with napari.gui_qt():
      viewer = napari.Viewer()
      time_points = da.array(time_points)
      filters = da.stack(filters)
      viewer.add_image(time_points, name='timepoints', blending='additive')
      # viewer.add_image(all_gaussians, name='gaussians', colormap='red', blending='additive')
      # viewer.add_image(all_medians, name='medians', colormap='green', blending='additive')
      # viewer.add_image(all_thresholds, name='thresholds', colormap='blue', blending='additive')
      viewer.add_image(filters.transpose(1, 2, 0, 3, 4, 5), name='filters', colormap='blue', blending='additive', opacity=.5)
      print(time_points.shape, filters.shape)
