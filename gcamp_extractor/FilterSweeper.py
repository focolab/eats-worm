import copy
from magicgui import magicgui
from magicgui._qt.widgets import QDoubleSlider
from qtpy.QtWidgets import QSlider
import napari
from napari.layers import Image
from PyQt5.QtCore import Qt
from .segfunctions import *
import skimage.data
import skimage.filters

######   THIS SECTION ONLY REQUIRED FOR NAPARI <= 0.2.12   ######

def do_experiment(e, timepoints=1):
    with napari.gui_qt():
        # create a viewer and add some images
        stacks = []
        for i in range(timepoints):
            im1 = e.im.get_t(t=i)
            im1_unfiltered = copy.deepcopy(im1)
            stacks.append(im1_unfiltered)
        viewer = napari.Viewer()
        viewer.add_image(np.array(stacks), name="worm", blending='additive', interpolation='bicubic')

        # turn the gaussian blur function into a magicgui
        # - `auto_call` tells magicgui to call the function whenever a parameter changes
        # - we use `widget_type` to override the default "float" widget on sigma
        # - we provide some Qt-specific parameters
        # - we contstrain the possible choices for `mode`
        @magicgui(
            auto_call=True,
            threshold={"widget_type": QDoubleSlider, "minimum": 0.5, "maximum": 1},
            median_size={"widget_type": QSlider, "minimum": 0, "maximum": 2},
            width_x={"widget_type": QSlider, "minimum": 0, "maximum": 24},
            width_y={"widget_type": QSlider, "minimum": 0, "maximum": 24},
            width_z={"widget_type": QSlider, "minimum": 0, "maximum": 24},
            sigma_x={"widget_type": QDoubleSlider, "maximum": 6},
            sigma_y={"widget_type": QDoubleSlider, "maximum": 6},
            sigma_z={"widget_type": QDoubleSlider, "maximum": 6},
        )
        def filter_and_threshold(layer: Image, threshold: float = 0.9, median_size: int = 0, width_x: int = 1, width_y: int = 1, width_z: int = 1, sigma_x: float = 1., sigma_y: float = 1., sigma_z: float = 1.) -> Image:
            """Apply a gaussian blur to ``layer``."""
            if layer:
                # todo: the order used here does not match the documentation in sefgunctions. change either order or documentation in segfunctions
                filtered_and_thresholded = []
                for stack in stacks:
                    blurred = gaussian3d(copy.deepcopy(stack), (2 * width_x + 1, 2 * width_y + 1, sigma_x, sigma_y, 2 * width_z + 1, sigma_z))
                    filtered = medFilter2d(blurred, [1,3,5][median_size])
                    thresholded = filtered > np.quantile(layer.data, threshold)
                    filtered_and_thresholded.append(thresholded)
                return np.array(filtered_and_thresholded)

        # instantiate the widget
        gui = filter_and_threshold.Gui()
        # add the gui to the viewer as a dock widget
        viewer.window.add_dock_widget(gui)
        # if a layer gets added or removed, refresh the dropdown choices
        viewer.layers.events.changed.connect(lambda x: gui.refresh_choices("layer"))
        viewer.layers["filter_and_threshold result"].colormap = "cyan"
        viewer.layers["filter_and_threshold result"].blending = "additive"
        viewer.layers["filter_and_threshold result"].interpolation = "bicubic"
