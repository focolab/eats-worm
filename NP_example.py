#!/usr/bin/env python3
## Example use case of eats-worm package

from eats_worm import *
import numpy as np#

data_directory= "/Users/danielsprague/FOCO_lab/data/NP_FOCO_cropped/2022-02-12-w01-NP1"
output_directory = "/Users/danielsprague/FOCO_lab/data/test_eats_NP"

arguments = {
    "root": data_directory,
    "numz": 45,
    "frames":list(range(45)),
    "numc": 4,
    # "end_t": 100,
    "offset": 0,
    "gaussian": False,
    "median": 3,
    "quantile": 0.963,
    "anisotropy": [2, 1, 1],
    "blob_merge_dist_thresh": 5,
    "remove_blobs_dist": 3,
    "algorithm":"tmip_2d_template",
    "algorithm_params": {
        "window_size": 5,
        "min_distance": 11,
        "manage_collisions": "prune",
        "fb_threshold_margin": 50
    },
    "processing_params":{
        "neuroPAL": True,
        "resample": False,
        "new_resolution": [0.375, 0.375, 0.75],
        "old_resolution": [0.3208, 0.3208, 0.75],
        "RGBW_channels": [0,2,4,1], 
        "histogram_match": False,
        "A_max": 4096,
        "ref_max": 655356,
        "im_to_match": '/Users/danielsprague/FOCO_lab/data/NP_paper/all/7_YAaLR.mat'
    },
    "register_frames": True,
    "output_dir": output_directory
}

e = Extractor(**arguments)
processed_im = e.process_im()
e.calc_blob_threads()
e.quantify(quant_function=background_subtraction_quant_function)

# e = load_extractor(output_dir)

c = Curator(e=e)

