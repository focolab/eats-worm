#!/usr/bin/env python3
## Example use case of eats-worm package for neuroPAL static image

from eats_worm import *
from npex import npex

data_directory= "/Users/danielsprague/FOCO_lab/data/NP_FOCO_eats/2022-02-12-w01-NP1"
output_directory = "/Users/danielsprague/FOCO_lab/data/NP_FOCO_eats/2022-02-12-w01-NP1"
npex_output = output_directory +'/peakfinding_output'

eats_params = {
    "root": data_directory,
    "numz": 45,
    "frames":[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44],
    "numc": 6,
    # "end_t": 100,
    "offset": 0,
    "gaussian": False,
    "median": 3,
    "quantile": 0.963,
    "anisotropy": [2, 1, 1],
    "blob_merge_dist_thresh": 5,
    "remove_blobs_dist": 3,
    "peakfind_channel":2,
    "algorithm":"tmip_2d_template",
    "algorithm_params": {
        "window_size": 5,
        "min_distance": 11,
        "manage_collisions": "prune",
        "fb_threshold_margin": 50
    },
    "processing_params": {
        "neuroPAL": True,
        "RGBW_channels": [0,2,4,1],
        "resample":False,
        "old_resolution": [0.3208,0.3208,0.75],
        "new_resolution": [0.325, 0.325, 0.75],
        "median_filter":True,
        "median": 3,
        "histogram_match":True,
        "A_max": 4096,
        "ref_max":65536,
        "im_to_match": "/Users/danielsprague/FOCO_lab/data/NP_paper/all/11_YAalR"
    },
    "register_frames": True,
    "output_dir": output_directory
}

npe_params = {
    "f": data_directory+'/processed_data.tif',
    "output_folder": npex_output,
    "rgbw_channels": [0, 1, 2, 3],
    "data_tag": data_directory.split('/')[-1],
}

e = Extractor(**eats_params)
e.process_im()
e.calc_blob_threads()
e.quantify(quant_function=background_subtraction_quant_function)

npe = npex.NPExtractor(**npe_params)
npe.launch_gui(windows_mode=False)
npe.export()

# e = load_extractor(output_dir)

#c = Curator(e=e)