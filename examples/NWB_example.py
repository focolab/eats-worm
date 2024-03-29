from eats_worm import * 

data_directory= '/Users/danielysprague/foco_lab/data/final_nwb/SK1/20230506-15-01-45.nwb' #filepath to any NWB file
output_directory = "/Users/danielysprague/foco_lab/data/eats_worm_test"


arguments = {
    "root": data_directory,
    "numz": 12,
    "numc": 1,
    #"end_t":500,
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
    "output_dir": output_directory,
    "regen_mft": False,
}

e = Extractor(**arguments)
e.calc_blob_threads()
e.quantify(quant_function=background_subtraction_quant_function)

#e = load_extractor(output_directory)
#e.im.io.close()

c = Curator(e=e)
c.save_nwb()

c.tf.io.close() #close the NWB file opened in the extractor step 