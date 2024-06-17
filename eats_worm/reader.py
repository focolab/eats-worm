import os
from pathlib import Path

import napari
from napari.utils.notifications import show_info

from eats_worm import MultiFileTiff


def napari_get_reader(path):
    if os.path.isdir(path):
        return read_multi_tiffs


def read_multi_tiffs(path):
    if len(napari.current_viewer().window._dock_widgets) == 0:
        show_info("The plugin eats-worm is not activated. \nPlease select eats-worm from Plugins in the menu bar.")
    else:
        for k, v in napari.current_viewer().window._dock_widgets.items():
            if "eats-worm" in k:
                loading_param = v.widget().loading_param
                loading_param.numz=200
                loading_param.numc=1
                loading_param.anisotropy = (0.39, 1, 0.55)
                break
        m = MultiFileTiff(path, numz=200, numc=1, anisotropy=(0.39, 1, 0.55))
        return {
            'data':m.get_dask_array(),
            'scale':m.anisotropy,
        }

