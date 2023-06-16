import os
from pynwb import load_namespaces, get_class
from pynwb.file import MultiContainerInterface, NWBContainer
import skimage.io as skio
from collections.abc import Iterable
import numpy as np
from pynwb import register_class
from hdmf.utils import docval, get_docval, popargs
from pynwb.ophys import ImageSeries 
from pynwb.core import NWBDataInterface
from hdmf.common import DynamicTable
from hdmf.utils import docval, popargs, get_docval, get_data_shape, popargs_to_dict
from pynwb.file import Device
import pandas as pd
import numpy as np
from pynwb import NWBFile, TimeSeries, NWBHDF5IO
from pynwb.epoch import TimeIntervals
from pynwb.file import Subject
from pynwb.behavior import SpatialSeries, Position
from pynwb.image import ImageSeries
from pynwb.ophys import OnePhotonSeries, OpticalChannel, ImageSegmentation, Fluorescence, CorrectedImageStack, MotionCorrection, RoiResponseSeries
from datetime import datetime
from dateutil import tz
import pandas as pd
import scipy.io as sio
from datetime import datetime, timedelta



# Set path of the namespace.yaml file to the expected install location
MultiChannelVol_specpath = os.path.join(
    os.path.dirname(__file__),
    'spec',
    'ndx-multichannel-volume.namespace.yaml'
)

# If the extension has not been installed yet but we are running directly from
# the git repo

if not os.path.exists(MultiChannelVol_specpath):
    MultiChannelVol_specpath = os.path.abspath(os.path.join(
        os.path.dirname(__file__),
        '..', '..', '..',
        'spec',
        'ndx-multichannel-volume.namespace.yaml'
    ))

# Load the namespace
load_namespaces(MultiChannelVol_specpath)

# TODO: import your classes here or define your class using get_class to make
# them accessible at the package level
CElegansSubject = get_class('CElegansSubject', 'ndx-multichannel-volume')
OpticalChannelReferences = get_class('OpticalChannelReferences', 'ndx-multichannel-volume')
OpticalChannelPlus = get_class('OpticalChannelPlus', 'ndx-multichannel-volume')
MultiChannelVolumeSeries = get_class('MultiChannelVolumeSeries', 'ndx-multichannel-volume')

@register_class('ImagingVolume', 'ndx-multichannel-volume')
class ImagingVolume(NWBDataInterface):
    """An imaging plane and its metadata."""

    __nwbfields__ = ({'name': 'optical_channel_plus', 'child': True},
                     'Order_optical_channels',
                     'description',
                     'device',
                     'location',
                     'conversion',
                     'origin_coords',
                     'origin_coords_units',
                     'grid_spacing',
                     'grid_spacing_units',
                     'reference_frame',
                     )

    @docval(*get_docval(NWBDataInterface.__init__, 'name'),  # required
            {'name': 'optical_channel_plus', 'type': ('data', 'array_data', OpticalChannelPlus),  # required
             'doc': 'One of possibly many groups storing channel-specific data.'},
            {'name': 'Order_optical_channels', 'type':OpticalChannelReferences, 'doc':'Order of the optical channels in the data'},
            {'name': 'description', 'type': str, 'doc': 'Description of this ImagingVolume.'},  # required
            {'name': 'device', 'type': Device, 'doc': 'the device that was used to record'},  # required
            {'name': 'location', 'type': str, 'doc': 'Location of image plane.'},  # required
            {'name': 'reference_frame', 'type': str,
             'doc': 'Describes position and reference frame of manifold based on position of first element '
                    'in manifold.',
             'default': None},
            {'name': 'origin_coords', 'type': 'array_data',
             'doc': 'Physical location of the first element of the imaging plane (0, 0) for 2-D data or (0, 0, 0) for '
                    '3-D data. See also reference_frame for what the physical location is relative to (e.g., bregma).',
             'default': None},
            {'name': 'origin_coords_unit', 'type': str,
             'doc': "Measurement units for origin_coords. The default value is 'meters'.",
             'default': 'meters'},
            {'name': 'grid_spacing', 'type': 'array_data',
             'doc': "Space between pixels in (x, y) or voxels in (x, y, z) directions, in the specified unit. Assumes "
                    "imaging plane is a regular grid. See also reference_frame to interpret the grid.",
             'default': None},
            {'name': 'grid_spacing_unit', 'type': str,
             'doc': "Measurement units for grid_spacing. The default value is 'meters'.",
             'default': 'meters'})
    def __init__(self, **kwargs):
        keys_to_set = ('optical_channel_plus',
                       'Order_optical_channels',
                       'description',
                       'device',
                       'location',
                       'reference_frame',
                       'origin_coords',
                       'origin_coords_unit',
                       'grid_spacing',
                       'grid_spacing_unit')
        args_to_set = popargs_to_dict(keys_to_set, kwargs)
        super().__init__(**kwargs)

        if not isinstance(args_to_set['optical_channel_plus'], list):
            args_to_set['optical_channel_plus'] = [args_to_set['optical_channel_plus']]

        for key, val in args_to_set.items():
            setattr(self, key, val)

@register_class('VolumeSegmentation', 'ndx-multichannel-volume')
class VolumeSegmentation(DynamicTable):
    """
    Stores pixels in an image that represent different regions of interest (ROIs)
    or masks. All segmentation for a given imaging volume is stored together, with
    storage for multiple imaging planes (masks) supported. Each ROI is stored in its
    own subgroup, with the ROI group containing both a 3D mask and a list of pixels
    that make up this mask. Segments can also be used for masking neuropil. If segmentation
    is allowed to change with time, a new imaging plane (or module) is required and
    ROI names should remain consistent between them.
    """

    __fields__ = ('imaging_volume','name')

    __columns__ = (
        {'name': 'image_mask', 'description': 'Image masks for each ROI'},
        {'name': 'voxel_mask', 'description': 'Voxel masks for each ROI', 'index': True},
        {'name': 'color_voxel_mask', 'description': 'Color voxel masks for each ROI', 'index':True}
    )

    @docval({'name': 'description', 'type': str,  # required
             'doc': 'Description of image plane, recording wavelength, depth, etc.'},
            {'name': 'imaging_volume', 'type': ImagingVolume,  # required
             'doc': 'the ImagingVolume this ROI applies to'},
            {'name': 'name', 'type': str, 'doc': 'name of VolumeSegmentation.', 'default': None},
            *get_docval(DynamicTable.__init__, 'id', 'columns', 'colnames'))
    def __init__(self, **kwargs):
        imaging_volume = popargs('imaging_volume', kwargs)
        if kwargs['name'] is None:
            kwargs['name'] = imaging_volume.name
        super().__init__(**kwargs)
        self.imaging_volume = imaging_volume

    @docval({'name': 'voxel_mask', 'type': 'array_data', 'default': None,
             'doc': 'voxel mask for 3D ROIs: [(x1, y1, z1, weight1, ID), (x2, y2, z2, weight2, ID), ...]',
             'shape': (None, 5)},
             {'name': 'color_voxel_mask', 'type': 'array_data', 'default': None,
             'doc': 'voxel mask for 3D ROIs with color information',
             'shape': (None, 9)},
            {'name': 'image_mask', 'type': 'array_data', 'default': None,
             'doc': 'image with the same size of image where positive values mark this ROI',
             'shape': [[None]*3]},
            {'name': 'id', 'type': int, 'doc': 'the ID for the ROI', 'default': None},
            allow_extra=True)
    def add_roi(self, **kwargs):
        """Add a Region Of Interest (ROI) data to this"""
        voxel_mask, color_voxel_mask, image_mask = popargs('voxel_mask', 'color_voxel_mask', 'image_mask', kwargs)
        if image_mask is None and voxel_mask is None and color_voxel_mask is None:
            raise ValueError("Must provide 'image_mask' and/or 'voxel_mask' and/or 'color_voxel_mask'")
        rkwargs = dict(kwargs)
        if image_mask is not None:
            rkwargs['image_mask'] = image_mask
        if voxel_mask is not None:
            rkwargs['voxel_mask'] = voxel_mask
        if color_voxel_mask is not None:
            rkwargs['color_voxel_mask'] = color_voxel_mask
        return super().add_row(**rkwargs)

    @staticmethod
    def voxel_to_image(voxel_mask):
        """Converts a #D pixel_mask of a ROI into an image_mask."""
        image_matrix = np.zeros(np.shape(voxel_mask))
        npmask = np.asarray(voxel_mask)
        x_coords = npmask[:, 0].astype(np.int32)
        y_coords = npmask[:, 1].astype(np.int32)
        z_coords = npmask[:, 2].astype(np.int32)
        weights = npmask[:, -1]
        image_matrix[y_coords, x_coords, z_coords] = weights
        return image_matrix

    @staticmethod
    def image_to_pixel(image_mask):
        """Converts an image_mask of a ROI into a pixel_mask"""
        voxel_mask = []
        it = np.nditer(image_mask, flags=['multi_index'])
        while not it.finished:
            weight = it[0][()]
            if weight > 0:
                x = it.multi_index[0]
                y = it.multi_index[1]
                z = it.multi_index[2]
                voxel_mask.append([x, y, z, weight])
            it.iternext()
        return voxel_mask

    @docval({'name': 'description', 'type': str, 'doc': 'a brief description of what the region is'},
            {'name': 'region', 'type': (slice, list, tuple), 'doc': 'the indices of the table', 'default': slice(None)},
            {'name': 'name', 'type': str, 'doc': 'the name of the ROITableRegion', 'default': 'rois'})
    def create_roi_table_region(self, **kwargs):
        return self.create_region(**kwargs)
    
@register_class('MultiChannelVolume', 'ndx-multichannel-volume')
class MultiChannelVolume(NWBDataInterface):
    """An imaging plane and its metadata."""

    __nwbfields__ = ('resolution',
                     'description',
                     'RGBW_channels',
                     'data',
                     'imaging_volume',
                     'Order_optical_channels'
                     )

    @docval(*get_docval(NWBDataInterface.__init__, 'name'),  # required
            {'name': 'resolution', 'type': 'array_data', 'doc':'pixel resolution of the image', 'shape':[None]},
            {'name': 'imaging_volume', 'type': ImagingVolume, 'doc': 'the Imaging Volume the data was generated from'},
            {'name': 'description', 'type': str, 'doc':'description of image'},
            {'name': 'RGBW_channels', 'doc': 'which channels in image map to RGBW', 'type': 'array_data', 'shape':[None]},
            {'name': 'data', 'doc': 'Volumetric multichannel data', 'type': 'array_data', 'shape':[None]*4},
            {'name': 'Order_optical_channels', 'type':OpticalChannelReferences, 'doc':'Order of the optical channels in the data'}
    )
    
    def __init__(self, **kwargs):
        keys_to_set = ('resolution',
                       'description',
                       'RGBW_channels',
                       'data',
                       'imaging_volume',
                       'Order_optical_channels'
                       )
        args_to_set = popargs_to_dict(keys_to_set, kwargs)
        super().__init__(**kwargs)

        for key, val in args_to_set.items():
            setattr(self, key, val)
