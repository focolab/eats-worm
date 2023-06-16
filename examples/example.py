from eats_worm import *

data_directory= "/Users/danielysprague/foco_lab/data/2021-05-29-w08-A1-gcamp"
output_directory = "/Users/danielysprague/foco_lab/data/eats_example"

arguments = {
    "root": data_directory,
    "numz": 12,
    "frames":[1,2,3,4,5,6,7,8,9,10,11],
    "numc": 1,
    # "end_t": 100,
    "offset": 0,
    "gaussian": False,
    "median": 3,
    "quantile": 0.963,
    "anisotropy": [7, 1, 1],
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

#e = Extractor(**arguments)
#e.calc_blob_threads()
#e.quantify(quant_function=background_subtraction_quant_function)

e = load_extractor(output_directory)

c = Curator(e=e)