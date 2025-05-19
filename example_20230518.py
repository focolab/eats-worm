#!/usr/bin/env python3
## Example use case of eats-worm package

from eats_worm import *

#data_directory = "Z:/muneki/MATLAB/SCAPE/Data/Voleti_et_al_Nat_Meth_2019/Immobilized_actual_properScale/green_actual_properScale"; numZ = 157; endT = 2; pixelSize = [0.75, 0.24, 0.20]
data_directory = "Z:/muneki/MATLAB/SCAPE/Data/Voleti_et_al_Nat_Meth_2019/Moving_actual_properScale/green_actual_properScale"; numZ = 127; endT = 2; pixelSize = [1.42, 0.37, 0.32]
output_directory = data_directory + "/output/"


arguments = {
    "root": data_directory,
    "numz": numZ,
    #"frames":[1,2,3,4,5,6,7,8,9,10,11],
    "numc": 1,
    "end_t": endT,
    "offset": 0,
    "gaussian": False,
    "median": 3,
    "quantile": 0.963,
    "anisotropy": pixelSize,
    "blob_merge_dist_thresh": 5,
    "remove_blobs_dist": 3,
    "algorithm":"tmip_2d_template",
    "algorithm_params": {
        "window_size": 5,
        "min_distance": 11,
        "manage_collisions": "prune",
        "fb_threshold_margin": 50
    },
    "register_frames": True,
    "output_dir": output_directory
}

e = Extractor(**arguments)
e.calc_blob_threads()
e.quantify(quant_function=background_subtraction_quant_function)

#e = load_extractor(output_dir)

c = Curator(e=e)