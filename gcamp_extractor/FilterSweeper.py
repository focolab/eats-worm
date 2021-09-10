import copy
from magicgui import magicgui
from magicgui.widgets import FloatSlider, Slider
import napari
from napari.layers import Image
from .segfunctions import *
import skimage.data
import skimage.filters
from skimage.feature import peak_local_max

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
            if len(e.gaussian) == 4:
                self.width_x_val = self.width_y_val = e.gaussian[0]
                self.width_z_val =e.gaussian[2]
                self.sigma_x = self.sigma_y = e.gaussian[1]
                self.sigma_z = e.gaussian[3]
            elif len(e.gaussian) == 6: 
                self.width_x_val = e.gaussian[0]
                self.width_y_val = e.gaussian[1]
                self.width_z_val = e.gaussian[4]
                self.sigma_x = e.gaussian[2]
                self.sigma_y = e.gaussian[3]
                self.sigma_z = e.gaussian[5]
        except:
            self.width_x_val = 1
            self.width_y_val = 1
            self.width_z_val = 1
            self.sigma_x = e.gaussian[2]
            self.sigma_y = e.gaussian[3]
            self.sigma_z = e.gaussian[5]
        try:
            self.quantile = e.quantile
        except:
            self.quantile = 0.99
        self.median_sizes = [1, 3, 5]
        try:
            self.median_index = self.median_sizes.index(e.median)
        except:
            self.median_index = 1
        try:
            self.num_peaks = e.skimage[0]
            self.min_distance = e.skimage[1]
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
        viewer.add_image(np.array(stacks), name="worm", blending='additive', scale=self.e.anisotropy)

        @magicgui(
            auto_call=True,
            median_index={"widget_type": Slider, "min": 0, "max": 2},
            width_x={"widget_type": Slider, "min": 0, "max": 24},
            width_y={"widget_type": Slider, "min": 0, "max": 24},
            width_z={"widget_type": Slider, "min": 0, "max": 24},
            sigma_x={"widget_type": FloatSlider, "max": 6},
            sigma_y={"widget_type": FloatSlider, "max": 6},
            sigma_z={"widget_type": FloatSlider, "max": 6},
        )
        def filter(layer: Image, median_index: int = self.median_index, width_x: int = (self.width_x_val - 1) // 2, width_y: int = (self.width_y_val - 1) // 2, width_z: int = (self.width_z_val - 1) // 2, sigma_x: float = self.sigma_x, sigma_y: float = self.sigma_y, sigma_z: float = self.sigma_z) -> napari.types.ImageData:
            """Apply a gaussian blur to ``layer``."""
            if layer:
                # todo: the order used here does not match the documentation in sefgunctions. change either order or documentation in segfunctions
                filtered = []
                self.median_index = median_index
                self.width_x_val = 2 * width_x + 1
                self.width_y_val = 2 * width_y + 1
                self.width_z_val = 2 * width_z + 1
                self.sigma_x = sigma_x
                self.sigma_y = sigma_y
                self.sigma_z = sigma_z
                for stack in stacks:
                    blurred = gaussian3d(copy.deepcopy(stack), (self.width_x_val, self.width_y_val, self.sigma_x, self.sigma_y, self.width_z_val, self.sigma_z))
                    median_filtered = medFilter2d(blurred, self.median_sizes[self.median_index])
                    filtered.append(median_filtered)
                return np.array(filtered)

        viewer.window.add_dock_widget(filter)

        @magicgui(
            auto_call=True,
            quantile={"widget_type": FloatSlider, "min": 0.5, "max": 1},
        )
        def threshold(layer: Image, quantile: float = self.quantile) -> napari.types.ImageData:
            self.quantile = quantile
            filtered = viewer.layers["filter result"].data
            return filtered > np.quantile(layer.data, self.quantile)

        viewer.window.add_dock_widget(threshold)

        @magicgui(
            auto_call=True,
            num_peaks={"widget_type": Slider, "min": 1, "max": 302},
            min_distance={"widget_type": Slider, "min": 1, "max": self.e.numz * self.e.anisotropy[0]},
        )
        def find_peaks(layer: Image, num_peaks: int = 50, min_distance: int = self.min_distance) -> napari.types.ImageData:
            if layer:
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

        @magicgui(
            auto_call=True,
            num_peaks={"widget_type": Slider, "min": 1, "max": 302},
            min_distance={"widget_type": Slider, "min": 1, "max": self.e.numz * self.e.anisotropy[0]},
            template_threshold={"widget_type": FloatSlider, "min": -1, "max": 1},
            erosions={"widget_type": Slider, "min": 0, "max": 20},
            spacing={"widget_type": Slider, "min": 1, "max": self.e.numz * self.e.anisotropy[0]},
        )
        def find_peaks_template(layer: Image, num_peaks: int = 50, min_distance: int = self.min_distance, template_threshold: float = .5, erosions: int = 3, spacing: int = 9) -> napari.types.ImageData:
            if not 'segmented' in viewer.layers:
                viewer.add_image(np.zeros(viewer.layers['filter result'].data.shape), name="segmented", blending='additive', scale=self.e.anisotropy, visible=False)
                viewer.add_image(np.zeros(viewer.layers['filter result'].data.shape), name="segmented thresholded", blending='additive', scale=self.e.anisotropy, visible=False)
                viewer.add_image(np.zeros(viewer.layers['filter result'].data.shape), name="eroded", blending='additive', scale=self.e.anisotropy, visible=False)
            if layer:
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
                        peaks = np.rint(self.e.peakfinding_params["template_peaks"]).astype(int)
                    except:
                        peaks = peak_local_max(expanded_im, min_distance=min_distance, num_peaks=num_peaks)
                        peaks //= self.e.anisotropy
                    chunks, blobs = peakfinder(data=stack, peaks=peaks, pad=[self.e.anisotropy[0]//dim for dim in self.e.anisotropy])
                    avg_3d_chunk = np.mean(chunks, axis=0)
                    self.template = BlobTemplate(data=avg_3d_chunk, scale=self.e.anisotropy, blobs='blobs')
                    self.last_min_distance = min_distance

                # the template match result needs padding to match the original data dimensions
                res = skimage.feature.match_template(stack, self.template.data)
                pad = [int((x-1)/2) for x in self.template.data.shape]
                res = np.pad(res, tuple(zip(pad, pad)))
                viewer.layers['segmented'].data = res
                filtered = res*np.array(res>template_threshold)
                viewer.layers['segmented thresholded'].data = filtered
                footprint = np.zeros((3,3,3))
                footprint[1,:,1] = 1
                footprint[1,1,:] = 1
                eroded = np.copy(filtered)
                for i in range(erosions):
                    eroded = skimage.morphology.erosion(eroded, selem=footprint)
                viewer.layers['eroded'].data = eroded
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
                peaks = np.rint(centers).astype(int)

                stack_peak_mask = np.zeros(stack.shape, dtype=bool)
                stack_peak_mask[tuple(peaks.T)] = True
                return stack_peak_mask
        
        if self.e.algorithm == "template":
            viewer.window.add_dock_widget(find_peaks_template)
            viewer.layers["find_peaks_template result"].colormap = "magenta"
            viewer.layers["find_peaks_template result"].blending = "additive"
            viewer.layers["find_peaks_template result"].scale = self.e.anisotropy
        else:
            viewer.window.add_dock_widget(find_peaks)
            viewer.layers["find_peaks result"].colormap = "magenta"
            viewer.layers["find_peaks result"].blending = "additive"
            viewer.layers["find_peaks result"].scale = self.e.anisotropy
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
        self.gaussian, self.median, self.skimage = final_params["gaussian"], final_params["median"], final_params["peaks"]
        print("final parameters from sweep: ", final_params)