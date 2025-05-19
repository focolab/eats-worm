#!/usr/bin/env python3
## Example use case of eats-worm package

from eats_worm import *

#data_dir = "Z:/muneki/MATLAB/SCAPE/Data/Voleti_et_al_Nat_Meth_2019/Immobilized_actual_properScale/green_actual_properScale"; numz = 157; end_t = 3575; pixel_size = [0.75, 0.24, 0.20]
#data_dir = "C:/Users/miked/OneDrive - UCSF/Documents/MATLAB/SCAPE/Data/Voleti_et_al_Nat_Meth_2019/Immobilized_actual_properScale/green_actual_properScale"; numz = 157; end_t = 1073; pixel_size = (0.20, 0.20, 0.20); #pixel_size = (0.20, 0.75, 0.24)
#data_dir = "C:/Users/miked/OneDrive - UCSF/Documents/MATLAB/SCAPE/Data/Voleti_et_al_Nat_Meth_2019/Moving_actual_properScale/green_actual_properScale"; numz = 127; end_t = 100; pixel_size = [1.42, 0.37, 0.32]
data_dir = "C:/Users/miked/OneDrive - UCSF/Documents/PythonScripts/wormjam/rawdata/worm02_act1/GCAMP"; numz = 12; end_t = 750; pixel_size = (3, 0.22, 0.22)
#data_dir = "C:/Users/miked/OneDrive - UCSF/Documents/PythonScripts/wormjam/rawdata/worm02_act1/GCAMP/tiff_stacks"; numz = 11; end_t = 5; pixel_size = (3, 0.22, 0.22)
#data_dir = "C:/Users/miked/OneDrive - UCSF/Documents/MATLAB/SCAPE/Data/Zyla/165mm/NeuroPAL/tiff_stacks/20230522_OH16290_1_run2"; numz = 199; end_t = 1039; pixel_size = [0.35, 0.75, 0.29]
#data_dir = "C:/Users/miked/OneDrive - UCSF/Documents/MATLAB/SCAPE/Data/Zyla/165mm/NeuroPAL/tiff_stacks/20230529_OH16290_1_run1"; numz = 127; end_t = 1039; pixel_size = [0.35, 0.75, 0.29]
#data_dir = "C:/Users/miked/OneDrive - UCSF/Documents/MATLAB/SCAPE/Data/Zyla/165mm/NeuroPAL/tiff_stacks/20230529_OH16290_2_run1"; numz = 127; end_t = 985; pixel_size = [0.35, 0.75, 0.29]
#data_dir = "C:/Users/miked/OneDrive - UCSF/Documents/MATLAB/SCAPE/Data/Zyla/165mm/NeuroPAL/tiff_stacks/20230711_OH16290_1_run1"; numz = 127; end_t = 1688; pixel_size = [0.35, 0.75, 0.25]
#data_dir = "C:/Users/miked/OneDrive - UCSF/Documents/MATLAB/SCAPE/Data/Zyla/165mm/NeuroPAL/tiff_stacks/20230714_OH16290_1_run1"; numz = 127; end_t = 1685; pixel_size = [0.35, 0.75, 0.25]
#data_dir = "C:/Users/miked/OneDrive - UCSF/Documents/MATLAB/SCAPE/Data/Zyla/165mm/NeuroPAL/tiff_stacks/20230715_OH16290_1_run1"; numz = 127; end_t = 1684; pixel_size = [0.35, 0.75, 0.25]
#data_dir = "C:/Users/miked/OneDrive - UCSF/Documents/MATLAB/SCAPE/Data/Zyla/165mm/NeuroPAL/20230716_OH16290_1_run1"; numz = 127; end_t = 1030; pixel_size = (0.25, 0.75, 0.35) #end_t = 1682;
#output_dir = data_dir + "/output_window_size_1_min_distance_1_fb_threshold_margin_50_9/"
#data_dir = "C:/Users/miked/OneDrive - UCSF/Documents/MATLAB/SCAPE/Data/Zyla/165mm/NeuroPAL/tiff_stacks/20230720_OH16290_1_run1"; numz = 120; end_t = 1714; pixel_size = [0.35, 0.75, 0.25]
#data_dir = "C:/Users/miked/OneDrive - UCSF/Documents/MATLAB/SCAPE/Data/Zyla/165mm/NeuroPAL/tiff_stacks/20230721_OH16290_1_run1"; numz = 120; end_t = 1713; pixel_size = [0.35, 0.75, 0.25]
#data_dir = "C:/Users/miked/OneDrive - UCSF/Documents/MATLAB/SCAPE/Data/Zyla/165mm/NeuroPAL/tiff_stacks/20230722_OH16290_1_run1"; numz = 120; end_t = 1712; pixel_size = [0.35, 0.75, 0.25]
#data_dir = "C:/Users/miked/OneDrive - UCSF/Documents/MATLAB/SCAPE/Data/Zyla/165mm/NeuroPAL/tiff_stacks/20230723_OH16290_1_run1"; numz = 120; end_t = 1715; pixel_size = [0.35, 0.75, 0.25]
#data_dir = "C:/Users/miked/OneDrive - UCSF/Documents/MATLAB/SCAPE/Data/Zyla/165mm/NeuroPAL/tiff_stacks/20230723_OH16290_1_run2"; numz = 120; end_t = 1715; pixel_size = [0.35, 0.75, 0.25]
#data_dir = "C:/Users/miked/OneDrive - UCSF/Documents/MATLAB/SCAPE/Data/Zyla/165mm/NeuroPAL/tiff_stacks/20230724_OH16290_1_run1"; numz = 120; end_t = 1715; pixel_size = [0.35, 0.75, 0.25]
#data_dir = "C:/Users/miked/OneDrive - UCSF/Documents/MATLAB/SCAPE/Data/Zyla/165mm/NeuroPAL/tiff_stacks/20230726_OH16290_1_run2"; numz = 120; end_t = 1715; pixel_size = [0.35, 0.75, 0.25]
#data_dir = "C:/Users/miked/OneDrive - UCSF/Documents/MATLAB/SCAPE/Data/HiCAM_2000/165mm/NeuroPAL/20230915_OH16290_2_run1/Deskewed_-45"; numz = 200; end_t = 2; pixel_size = (0.55, 1, 0.47)
#data_dir = "C:/Users/miked/OneDrive - UCSF/Documents/MATLAB/SCAPE/Data/HiCAM_2000/165mm/NeuroPAL/20230916_OH16290_2_run1/Deskewed_-45"; numz = 200; end_t = 5; pixel_size = (0.39, 1, 0.55)
#data_dir = "C:/Users/miked/OneDrive - UCSF/Documents/MATLAB/SCAPE/Data/HiCAM_2000/165mm/NeuroPAL/20230928_OH16290_1_run1/Deskewed_-40"; numz = 200; end_t = 824; pixel_size = (0.35, 1, 0.55)
#output_dir = data_dir + "/output_window_size_5_min_distance_10_fb_threshold_margin_50/"
output_dir = data_dir + "/output/"


arguments = {
    "root": data_dir,
    "numz": numz,
    "frames":[1,2,3,4,5,6,7,8,9,10,11],
    "numc": 1,
    "end_t": end_t,
    "offset": 0,
    "gaussian": False,
    "median": 3, #5
    "quantile": 0.963, #0.999
    "anisotropy": pixel_size,
    "blob_merge_dist_thresh": 3, #5
    "remove_blobs_dist": 3,
    "algorithm":"tmip_2d_template",
    "algorithm_params": {
        "window_size": 5, #3
        "min_distance": 2, #10
        "manage_collisions": "prune",
        "fb_threshold_margin": 10
    },
    "register_frames": True,
    "output_dir": output_dir
}

#e = Extractor(**arguments)
#e.calc_blob_threads()
#e.quantify(quant_function=background_subtraction_quant_function)

e = load_extractor(output_dir)
#e.quantify(quant_function=background_subtraction_quant_function)

c = Curator(e=e)