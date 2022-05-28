import numpy as np
import scipy.spatial
import copy
import dask
import napari
import os
import json
import matplotlib.pyplot as plt

from .segfunctions import *
from sklearn import mixture

med_filter_sizes = [1, 3, 5]

width_x_width_y_values = [1, 3, 5, 7, 27]
width_z_values = [1, 3, 5, 7, 9]
sigma_x_sigma_y_values = [1, 3, 5, 7, 9]
sigma_z_values = [1, 2, 3, 4, 5]

gaussian_params = [(width_x_width_y_values[gauss_arg], width_z_values[gauss_arg], sigma_x_sigma_y_values[gauss_arg], sigma_z_values[gauss_arg]) for gauss_arg in np.arange(5)]

quantiles = np.arange(0.925, 1., 0.005)

def threshold(image, quantile):
    return image > np.quantile(copy.deepcopy(image), quantile)

def gaussian_filter(image, params):
  return gaussian3d(copy.deepcopy(image), params)

def median_filter(image, size):
  return medFilter2d(copy.deepcopy(image), size)

@dask.delayed
def filter_and_threshold(image, gaussian_params, median_filter_size, quantile):
  return dask.array.from_array(threshold(median_filter(gaussian_filter(image, gaussian_params), median_filter_size), quantile), chunks=image.shape)

def do_experiment(e):
    """
    do ...
    """
    time_points = []
    filters = []
    for i in range(e.t):
        im1 = e.im.get_t(t=i)
        im1_unfiltered = copy.deepcopy(im1)

        time_points.append(im1_unfiltered)
      
    time_points = dask.array.array(time_points)
    all_filters = [filter_and_threshold(im, gaussian_param, med_filter_size, quantile) for im in time_points for gaussian_param in gaussian_params for med_filter_size in med_filter_sizes for quantile in quantiles]
    dask_arrays = [dask.array.from_delayed(filtered_image, shape=im1.shape, dtype=im1.dtype) for filtered_image in all_filters]
    with napari.gui_qt():
      viewer = napari.Viewer()
      filters = dask.array.stack(dask_arrays).reshape((len(gaussian_params), len(med_filter_sizes), len(quantiles)) + im1.shape)
      viewer.add_image(time_points, name='timepoints', blending='additive')
      viewer.add_image(filters, name='filters', colormap='blue', blending='additive', opacity=.5)
      viewer.dims.axis_labels=["gauss. filt. params", "med. filt. size", "threshold", "z", "y", "x"]
    selected = viewer.dims.point
    selected_gaussian_params = gaussian_params[selected[0]]
    selected_median_filter_size = med_filter_sizes[selected[1]]
    selected_quantile = quantiles[selected[2]]
    print("Selected parameters:\nGaussian: {}\nMedian Filter: {}\nThreshold:{}".format(selected_gaussian_params, selected_median_filter_size, selected_quantile))
    
    return selected_gaussian_params, selected_median_filter_size, selected_quantile
