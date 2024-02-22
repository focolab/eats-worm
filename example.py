#!/usr/bin/env python3
## Example use case of eats-worm package

from eats_worm import *

#data_dir = "D:/SCAPE/Data/Voleti_et_al_Nat_Meth_2019/Immobilized_actual_properScale/green_actual_properScale"; numz = 157; end_t = 3575; pixel_size = (0.75, 0.24, 0.20)
#data_dir = "D:/SCAPE/Data/Voleti_et_al_Nat_Meth_2019/Moving_actual_properScale/green_actual_properScale"; numz = 127; end_t = 460; pixel_size = (1.42, 0.37, 0.32)
#data_dir = "D:/SCAPE/Data/Zyla_4.2PLUS/165mm/NeuroPAL/tiff_stacks/20230716_OH16290_1_run1"; numz = 127; end_t = 1682; pixel_size = (0.35, 0.75, 0.25)
data_dir = "E:/SCAPE/Data/HiCAM_2000/165mm/NeuroPAL/20230914/tiff_stacks/20230914_OH16290_3_run1/Deskewed_-45"; numz = 200; end_t = 818; pixel_size = (0.55, 1, 0.39)
output_dir = data_dir + "/output/"


arguments = {
    "root": data_dir,
    "numz": numz,
    #"frames":[1,2,3,4,5,6,7,8,9,10,11],
    "numc": 1,
    "end_t": end_t,
    "offset": 0,
    "gaussian": False,
    "median": 3, #5
    "quantile": 0.963, #0.999
    "anisotropy": pixel_size,
    "blob_merge_dist_thresh": 5, #10
    "remove_blobs_dist": 3,
    "algorithm":"tmip_2d_template",
    "algorithm_params": {
        "window_size": 5, #3
        "min_distance": 10, #1
        "manage_collisions": "prune",
        "fb_threshold_margin": 50,
    },
    "register_frames": False, #True
    "output_dir": output_dir
}

#e = Extractor(**arguments)
#e.calc_blob_threads()
#e.quantify(quant_function=background_subtraction_quant_function)

e = load_extractor(output_dir)

c = Curator(e=e)