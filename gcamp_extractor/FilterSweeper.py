import copy
from magicgui import magicgui
from magicgui.widgets import FloatSlider, Slider
import napari
from napari.layers import Image
from napari.types import LayerDataTuple
from improc.segfunctions import *
import skimage.data
import skimage.filters
from skimage.feature import peak_local_max
from skimage.filters import rank

import scipy.ndimage
from skimage._shared.coord import ensure_spacing
import skimage.feature
import skimage.morphology

class FilterSweeper:
    """
    Provides a GUI for parameter selection for our filtering and thresholding flow and tracks selected parameter values.
    
    Arguments
    ---------
    e: Extractor
        extractor object containing the video/image stack stack for which you want select parameters

    Attributes
    ---------
    gaussian: tuple
        selected gaussian filter parameters
    median: int
        selected median filter parameters
    quantile: float
        selected threshold
    """
    def __init__(self, e):
        self.e = e
        try:
            if not e.blobthreadtracker_params['gaussian']:
                self.width_x_val = 1
                self.width_y_val = 1
                self.width_z_val = 1
                self.sigma_x = 1
                self.sigma_y = 1
                self.sigma_z = 1
            elif len(e.blobthreadtracker_params['gaussian']) == 4:
                self.width_x_val = self.width_y_val = e.blobthreadtracker_params['gaussian'][0]
                self.width_z_val =e.blobthreadtracker_params['gaussian'][2]
                self.sigma_x = self.sigma_y = e.blobthreadtracker_params['gaussian'][1]
                self.sigma_z = e.blobthreadtracker_params['gaussian'][3]
            elif len(e.blobthreadtracker_params['gaussian']) == 6: 
                self.width_x_val = e.blobthreadtracker_params['gaussian'][0]
                self.width_y_val = e.blobthreadtracker_params['gaussian'][1]
                self.width_z_val = e.blobthreadtracker_params['gaussian'][4]
                self.sigma_x = e.blobthreadtracker_params['gaussian'][2]
                self.sigma_y = e.blobthreadtracker_params['gaussian'][3]
                self.sigma_z = e.blobthreadtracker_params['gaussian'][5]
        except:
            self.width_x_val = 1
            self.width_y_val = 1
            self.width_z_val = 1
            self.sigma_x = e.blobthreadtracker_params['gaussian'][2]
            self.sigma_y = e.blobthreadtracker_params['gaussian'][3]
            self.sigma_z = e.blobthreadtracker_params['gaussian'][5]
        try:
            self.quantile = e.blobthreadtracker_params['quantile']
        except:
            self.quantile = 0.99
        self.median_sizes = [1, 3, 5]
        try:
            self.median_index = self.median_sizes.index(e.blobthreadtracker_params['median'])
        except:
            self.median_index = 1
        try:
            self.num_peaks = e.blobthreadtracker_params['algorithm_params']['num_peaks']
            self.min_distance = e.blobthreadtracker_params['algorithm_params']['min_distance']
        except:
            self.num_peaks = 50
            self.min_distance = 3


    # adapted from example at https://magicgui.readthedocs.io/en/latest/examples/napari_parameter_sweep/
    def sweep_parameters(self, timepoints=1):
        """
        use napari and magicgui to do a parameter sweep for our filtering and thresholding flow. upon napari window close, return dict of selected parameters.

        Parameters
        ----------
        timepoints : int
            specifies number of timepoints for which each parameter combination will be evaluated. by default, only the first timepoint is used.
            timepoint 0 is always the first timepoint of the recording.
        """
        stacks = []
        for i in range(timepoints):
            im1 = self.e.im.get_t(t=i)
            im1_unfiltered = copy.deepcopy(im1)
            stacks.append(im1_unfiltered)
        viewer = napari.Viewer(ndisplay=3)
        worm = viewer.add_image(np.array(stacks), name="worm", blending='additive', scale=self.e.anisotropy)


        @worm.events.data.connect
        @magicgui(
            auto_call=True,
            median_index={"widget_type": Slider, "min": 0, "max": 2},
            width_x={"widget_type": Slider, "min": 0, "max": 24},
            width_y={"widget_type": Slider, "min": 0, "max": 24},
            width_z={"widget_type": Slider, "min": 0, "max": 24},
            sigma_x={"widget_type": FloatSlider, "max": 6},
            sigma_y={"widget_type": FloatSlider, "max": 6},
            sigma_z={"widget_type": FloatSlider, "max": 6},
            parent_layer={'bind': worm},
            event={'bind': None}
        )
        def filter(event: None, parent_layer: Image, median_index: int = self.median_index, width_x: int = (self.width_x_val - 1) // 2, width_y: int = (self.width_y_val - 1) // 2, width_z: int = (self.width_z_val - 1) // 2, sigma_x: float = self.sigma_x, sigma_y: float = self.sigma_y, sigma_z: float = self.sigma_z) -> napari.types.ImageData:
            """Apply a gaussian blur to ``layer``."""
            if parent_layer:
                # todo: the order used here does not match the documentation in sefgunctions. change either order or documentation in segfunctions
                filtered = []
                self.median_index = median_index
                if 'gaussian' in self.e.blobthreadtracker_params:
                    self.width_x_val = 2 * width_x + 1
                    self.width_y_val = 2 * width_y + 1
                    self.width_z_val = 2 * width_z + 1
                    self.sigma_x = sigma_x
                    self.sigma_y = sigma_y
                    self.sigma_z = sigma_z
                for stack in stacks:
                    blurred = copy.deepcopy(stack)
                    if 'gaussian' in self.e.blobthreadtracker_params:
                        blurred = gaussian3d(copy.deepcopy(stack), (self.width_x_val, self.width_y_val, self.sigma_x, self.sigma_y, self.width_z_val, self.sigma_z))
                    median_filtered = medFilter2d(blurred, self.median_sizes[self.median_index])
                    filtered.append(median_filtered)
                return np.array(filtered)

        viewer.window.add_dock_widget(filter)
        filter_result = viewer.layers['filter result']

        @filter_result.events.data.connect
        @magicgui(
            auto_call=True,
            quantile={"widget_type": FloatSlider, "min": 0.5, "max": 1},
            parent_layer={'bind': filter_result},
            event={'bind': None}
        )
        def threshold(event: None, parent_layer: Image, quantile: float = self.quantile) -> napari.types.ImageData:
            self.quantile = quantile
            filtered = parent_layer.data
            return filtered > np.quantile(filtered, self.quantile)

        viewer.window.add_dock_widget(threshold)
        threshold_result = viewer.layers['threshold result']

        @threshold_result.events.data.connect
        @magicgui(
            auto_call=True,
            num_peaks={"widget_type": Slider, "min": 1, "max": 302},
            min_distance={"widget_type": Slider, "min": 1, "max": self.e.numz * self.e.anisotropy[0]},
            parent_layer={'bind': threshold_result},
            event={'bind': None}
        )
        def find_peaks(event: None, parent_layer: Image, num_peaks: int = 50, min_distance: int = self.min_distance) -> napari.types.ImageData:
            if parent_layer:
                self.num_peaks = num_peaks
                self.min_distance = min_distance
                peaks = []
                filtered = viewer.layers["filter result"].data
                thresholded = viewer.layers["threshold result"].data
                processed = filtered * thresholded
                for stack in processed:
                    expanded_im = np.repeat(stack, self.e.anisotropy[0], axis=0)
                    expanded_im = np.repeat(expanded_im, self.e.anisotropy[1], axis=1)
                    expanded_im = np.repeat(expanded_im, self.e.anisotropy[2], axis=2)
                    stack_peaks = peak_local_max(np.array(expanded_im), min_distance=min_distance, num_peaks=num_peaks)
                    stack_peaks //= self.e.anisotropy
                    stack_peak_mask = np.zeros(stack.shape, dtype=bool)
                    stack_peak_mask[tuple(stack_peaks.T)] = True
                    peaks.append(stack_peak_mask)
                return np.array(peaks)

        self.last_min_distance = None

        if self.e.blobthreadtracker_params['algorithm'] == 'template':

            @threshold_result.events.data.connect
            @magicgui(
                auto_call=True,
                num_peaks={"widget_type": Slider, "min": 1, "max": 302},
                min_distance={"widget_type": Slider, "min": 1, "max": self.e.numz * self.e.anisotropy[0]},
                parent_layer={'bind': threshold_result},
                event={'bind': None}
            )
            def template_centers(event: None, parent_layer: Image, num_peaks: int = 50, min_distance: int = self.min_distance) -> napari.types.LayerDataTuple:
                self.num_peaks = num_peaks
                self.min_distance = min_distance
                peaks = []
                filtered = viewer.layers["filter result"].data
                thresholded = viewer.layers["threshold result"].data
                processed = filtered * thresholded
                stack = processed[0]

                if min_distance != self.last_min_distance:
                    expanded_im = np.repeat(stack, self.e.anisotropy[0], axis=0)
                    expanded_im = np.repeat(expanded_im, self.e.anisotropy[1], axis=1)
                    expanded_im = np.repeat(expanded_im, self.e.anisotropy[2], axis=2)
                    try:
                        peaks = np.rint(self.e.blobthreadtracker_params['algorithm_params']["template_peaks"]).astype(int)
                    except:
                        peaks = peak_local_max(expanded_im, min_distance=min_distance, num_peaks=num_peaks)
                        peaks //= self.e.anisotropy
                        self.template_peaks = peaks
                return (peaks, {'name': 'template centers', 'blending': 'additive', 'scale': self.e.anisotropy, 'visible':False, 'face_color':'cyan', 'size': 1}, 'points')

            viewer.window.add_dock_widget(template_centers)
            template_centers_result = viewer.layers['template centers']

            @template_centers_result.events.data.connect
            @magicgui(
                auto_call=True,
                parent_layer={'bind': threshold_result},
                event={'bind': None}
            )
            def templates(event: None, parent_layer: Image, num_peaks: int = 50, min_distance: int = self.min_distance) -> napari.types.LayerDataTuple:
                filtered = viewer.layers["filter result"].data
                thresholded = viewer.layers["threshold result"].data
                processed = filtered * thresholded
                stack = processed[0]
                peaks = viewer.layers['template centers'].data
                chunks = get_bounded_chunks(data=im1, peaks=peaks, pad=[1, 25, 25])
                chunk_shapes = [chunk.shape for chunk in chunks]
                max_chunk_shape = (max([chunk_shape[0] for chunk_shape in chunk_shapes]), max([chunk_shape[1] for chunk_shape in chunk_shapes]), max([chunk_shape[2] for chunk_shape in chunk_shapes]))
                self.templates = [np.mean(np.array([chunk for chunk in chunks if chunk.shape == max_chunk_shape]), axis=0)]
                quantiles = self.e.blobthreadtracker_params['algorithm_params'].get('quantiles', [0.5])
                rotations = self.e.blobthreadtracker_params['algorithm_params'].get('rotations', [0])
                for quantile in quantiles:
                    for rotation in rotations:
                        try:
                            self.templates.append(scipy.ndimage.rotate(np.quantile(chunks, quantile, axis=0), rotation, axes=(-1, -2)))
                        except:
                            pass

                stacked = np.array([template.data for template in self.templates]).reshape((self.templates[0].data.shape[0], len(self.templates) * self.templates[0].data.shape[1], self.templates[0].data.shape[2]))
                return (stacked, {'name': 'templates', 'blending': 'additive', 'scale': self.e.anisotropy, 'visible':False}, 'image')

            viewer.window.add_dock_widget(templates)
            templates_result = viewer.layers['templates']

            @templates_result.events.data.connect
            @magicgui(
                auto_call=True,
                template_threshold={"widget_type": FloatSlider, "min": -1, "max": 1},
                erosions={"widget_type": Slider, "min": 0, "max": 20},
                spacing={"widget_type": Slider, "min": 1, "max": self.e.numz * self.e.anisotropy[0]},
                parent_layer={'bind': threshold_result},
                event={'bind': None}
            )
            def find_peaks_template(event: None, parent_layer: Image, template_threshold: float = .5, erosions: int = 0, spacing: int = 9) -> napari.types.ImageData:
                if not 'segmented' in viewer.layers:
                    viewer.add_image(np.zeros(parent_layer.data.shape), name="segmented", blending='additive', scale=self.e.anisotropy, visible=False)
                    viewer.add_image(np.zeros(parent_layer.data.shape), name="segmented thresholded", blending='additive', scale=self.e.anisotropy, visible=False)
                    viewer.add_image(np.zeros(parent_layer.data.shape), name="eroded", blending='additive', scale=self.e.anisotropy, visible=False)
                    viewer.add_points(np.empty((0, 3)), name="found peaks", blending='additive', scale=self.e.anisotropy, visible=False)
                if parent_layer:
                    filtered = viewer.layers["filter result"].data
                    thresholded = viewer.layers["threshold result"].data
                    processed = filtered * thresholded
                    stack = processed[0]

                    peaks = None
                    for template in self.templates:

                        # the template match result needs padding to match the original data dimensions
                        res = skimage.feature.match_template(stack, template)
                        pad = [int((x-1)/2) for x in template.shape]
                        res = np.pad(res, tuple(zip(pad, pad)))
                        filtered = res*np.array(res>template_threshold)
                        footprint = np.zeros((3,3,3))
                        footprint[1,:,1] = 1
                        footprint[1,1,:] = 1
                        eroded = np.copy(filtered)
                        for i in range(erosions):
                            eroded = skimage.morphology.erosion(eroded, selem=footprint)
                        labeled_features, num_features = scipy.ndimage.label(eroded)
                        centers = []
                        for feature in range(num_features):
                            center = scipy.ndimage.center_of_mass(eroded, labels=labeled_features, index=feature)
                            centers.append(list(center))

                        centers = np.array(centers)
                        centers = np.rint(centers[~np.isnan(centers).any(axis=1)]).astype(int)
                        intensities = eroded[tuple(centers.T)]
                        # Highest peak first
                        idx_maxsort = np.argsort(-intensities)
                        centers = centers[idx_maxsort]

                        centers = ensure_spacing(centers, spacing=spacing)
                        template_peaks = np.rint(centers).astype(int)

                        if peaks is None:
                            peaks = template_peaks
                            viewer.layers['segmented'].data = res
                            viewer.layers['segmented thresholded'].data = filtered
                            viewer.layers['eroded'].data = eroded
                        else:
                            peaks = np.concatenate((peaks, template_peaks))
                            viewer.layers['segmented'].data = np.maximum(res, viewer.layers['segmented'].data)
                            viewer.layers['segmented thresholded'].data = np.maximum(filtered, viewer.layers['segmented thresholded'].data)
                            viewer.layers['eroded'].data = np.maximum(eroded, viewer.layers['eroded'].data)

                    peak_mask = np.zeros(im1.shape, dtype=bool)
                    peak_mask[tuple(peaks.T)] = True
                    peak_masked = im1 * peak_mask
                    expanded_im = np.repeat(peak_masked, self.e.anisotropy[0], axis=0)
                    expanded_im = np.repeat(expanded_im, self.e.anisotropy[1], axis=1)
                    expanded_im = np.repeat(expanded_im, self.e.anisotropy[2], axis=2)
                    peaks = peak_local_max(expanded_im, min_distance=13)
                    peaks //= self.e.anisotropy

                    self.peaks_data = peaks

                    stack_peak_mask = np.zeros(stack.shape, dtype=bool)
                    stack_peak_mask[tuple(peaks.T)] = True
                    return stack_peak_mask
        
            viewer.window.add_dock_widget(find_peaks_template)
            viewer.layers["find_peaks_template result"].colormap = "magenta"
            viewer.layers["find_peaks_template result"].blending = "additive"
            viewer.layers["find_peaks_template result"].scale = self.e.anisotropy

        else:
            viewer.window.add_dock_widget(find_peaks)
            find_peaks_result = viewer.layers['find_peaks result']
            viewer.layers["find_peaks result"].colormap = "magenta"
            viewer.layers["find_peaks result"].blending = "additive"
            viewer.layers["find_peaks result"].scale = self.e.anisotropy

            @find_peaks_result.events.data.connect
            @magicgui(
                auto_call=True,
                min_filter_size={"widget_type": Slider, "min": 0, "max": 11},
                parent_layer={'bind': find_peaks_result},
                event={'bind': None}
            )
            def min_filter(event: None, parent_layer: Image, min_filter_size: int = 0) -> napari.types.ImageData:
                threshed = np.copy(viewer.layers['threshold result'].data[0])
                minned = threshed
                if min_filter_size > 0:
                    minned = rank.minimum(threshed, np.ones((1, min_filter_size, min_filter_size)))
                peaks_mask = viewer.layers['find_peaks result'].data[0]
                filtered = minned * viewer.layers['find_peaks result'].data[0]
                return filtered
            viewer.window.add_dock_widget(min_filter)
            viewer.layers["min_filter result"].colormap = "green"
            viewer.layers["min_filter result"].blending = "additive"
            viewer.layers["min_filter result"].scale = self.e.anisotropy

        viewer.layers.events.changed.connect(lambda x: gui.refresh_choices("layer"))
        viewer.layers["threshold result"].colormap = "cyan"
        viewer.layers["threshold result"].blending = "additive"
        viewer.layers["threshold result"].scale = self.e.anisotropy
        viewer.layers["threshold result"].visible = False
        viewer.layers["filter result"].blending = "additive"
        viewer.layers["filter result"].scale = self.e.anisotropy
        viewer.layers["worm"].visible = False

        napari.run()

        final_params = {"gaussian": (self.width_x_val, self.width_y_val, self.sigma_x, self.sigma_y, self.width_z_val, self.sigma_z), "median": self.median_sizes[self.median_index], "quantile": self.quantile, "peaks": (self.num_peaks, self.min_distance)}
        self.gaussian, self.median, self.e.blobthreadtracker_params["algorithm_params"]["min_distance"], self.e.blobthreadtracker_params["algorithm_params"]["num_peaks"] = final_params["gaussian"], final_params["median"], final_params["peaks"][1], final_params["peaks"][0]
        self.e.blobthreadtracker_params['algorithm_params']['peaks'] = self.peaks_data
        print("final parameters from sweep: ", final_params)