import copy
from magicgui import magicgui
from magicgui.widgets import FloatSlider, Slider
import napari
from napari.layers import Image
from .segfunctions import *
import skimage.data
import skimage.filters
from skimage.feature import peak_local_max

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
        self.num_peaks = 50
        try:
            self.footprint_z = 3
            self.footprint_x = e.reg_peak_dist
            self.footprint_y = e.reg_peak_dist
        except:
            self.footprint_z = 3,
            self.footprint_x = 20,
            self.footprint_y = 20,


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
        with napari.gui_qt():
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
                footprint_z={"widget_type": Slider, "min": 1, "max": self.e.numz},
                footprint_x={"widget_type": Slider, "min": 1, "max": 200},
                footprint_y={"widget_type": Slider, "min": 1, "max": 200},
            )
            def find_peaks(layer: Image, num_peaks: int = 50, footprint_z: int = self.footprint_z, footprint_x: int = self.footprint_x, footprint_y: int = self.footprint_y) -> napari.types.ImageData:
                if layer:
                    self.num_peaks = num_peaks
                    self.footprint_z = footprint_z
                    self.footprint_x = footprint_x
                    self.footprint_y = footprint_y
                    peaks = []
                    for stack in stacks:
                        peaks.append(peak_local_max(np.array(stack), footprint=np.ones((footprint_z, footprint_x, footprint_y)), num_peaks=num_peaks, indices=False))
                    return np.array(peaks)
            
            viewer.window.add_dock_widget(find_peaks)
            viewer.layers.events.changed.connect(lambda x: gui.refresh_choices("layer"))
            viewer.layers["threshold result"].colormap = "cyan"
            viewer.layers["threshold result"].blending = "additive"
            viewer.layers["threshold result"].scale = self.e.anisotropy
            viewer.layers["threshold result"].visible = False
            viewer.layers["filter result"].blending = "additive"
            viewer.layers["filter result"].scale = self.e.anisotropy
            viewer.layers["find_peaks result"].colormap = "magenta"
            viewer.layers["find_peaks result"].blending = "additive"
            viewer.layers["find_peaks result"].scale = self.e.anisotropy
            viewer.layers["worm"].visible = False
                

        final_params = {"gaussian": (self.width_x_val, self.width_y_val, self.sigma_x, self.sigma_y, self.width_z_val, self.sigma_z), "median": self.median_sizes[self.median_index], "quantile": self.quantile, "peaks": (self.num_peaks, self.footprint_z, self.footprint_x, self.footprint_y)}
        self.gaussian, self.median, self.skimage = final_params["gaussian"], final_params["median"], final_params["peaks"]
        print("final parameters from sweep: ", final_params)