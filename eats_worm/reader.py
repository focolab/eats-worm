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
        n_imgs = len(list(Path(path).glob('*.tiff')))
        for k, v in napari.current_viewer().window._dock_widgets.items():
            if "eats_worm" in k:
                loading_param = v.widget()
                loading_param.load_path.setText(str(path))
                loading_param.numz.setText(str(n_imgs))
                loading_param.numc.setText(str(1))
                loading_param.anisotropy.setText(str((0.39, 1, 0.55)))
                break
        m = MultiFileTiff(path, numz=n_imgs, numc=1, anisotropy=(0.39, 1, 0.55))
        return [(m.get_dask_array(), {'colormap': 'inferno', 'scale': m.anisotropy})]

