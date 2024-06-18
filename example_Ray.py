#!/usr/bin/env python3
## Example use case of eats-worm package

from eats_worm import *

segmentation_algorithm = "tmip_2d_template"
#data_dir = "D:/SCAPE/Data/Voleti_et_al_Nat_Meth_2019/Immobilized_actual_properScale/green_actual_properScale"; numz = 157; end_t = 3575; pixel_size = (0.20, 0.20, 0.20) #pixel_size = (0.20, 0.75, 0.24)
#data_dir = "D:/SCAPE/Data/Voleti_et_al_Nat_Meth_2019/Moving_actual_properScale/green_actual_properScale"; numz = 127; end_t = 10; pixel_size = (032, 0.32, 0.32) #pixel_size = (0.32, 1.42, 0.37)
#data_dir = "D:/SCAPE/Data/Zyla/165mm/NeuroPAL/tiff_stacks/20230716_OH16290_1_run1"; numz = 127; end_t = 1030; pixel_size = (0.25, 0.75, 0.35) #end_t = 1682
#data_dir = "D:/SCAPE/Data/Zyla/165mm/NeuroPAL/tiff_stacks/20230724_OH16290_1_run1"; numz = 120; end_t = 1030; pixel_size = (0.25, 0.75, 0.35) #end_t = 1682
#data_dir = "D:/SCAPE/Data/Zyla/165mm/NeuroPAL/tiff_stacks/20230726_OH16290_1_run2"; numz = 125; end_t = 1030; pixel_size = (0.25, 0.75, 0.35) #end_t = 1682
#data_dir = "E:/SCAPE/Data/20230911/tiff_stacks/20230911_GreenBeads4um_run3/Deskewed_-60"; numz = 200; end_t = 3; pixel_size = (0.55, 1, 0.47)
#data_dir = "E:/SCAPE/Data/HiCAM_2000/165mm/NeuroPAL/20230924/tiff_stacks/20230924_OH16290_1_run2/Deskewed_-45"; numz = 200; end_t = 818; pixel_size = (0.39, 1, 0.55)
#data_dir = "E:/SCAPE/Data/HiCAM_2000/165mm/NeuroPAL/20230926/tiff_stacks/20230926_OH16290_2_run1/Deskewed_-45"; numz = 200; end_t = 824; pixel_size = (0.39, 0.75, 0.55) #end_t = 827
#data_dir = "E:/SCAPE/Data/HiCAM_2000/165mm/NeuroPAL/20230928/tiff_stacks/20230928_OH16290_1_run1/Deskewed_-40"; numz = 200; end_t = 824; pixel_size = (0.35, 0.75, 0.55)

data_dir = "E:/HiCAM_2000/NeuroPAL/20240525_OH16289_1_run1/Deskewed_-45"; numz = 84; end_t = 683; pixel_size = (0.39, 1, 0.55)

#output_dir = data_dir + "/output/"
output_dir = data_dir + "/output_window_size_5_min_distance_10_fb_threshold_margin_50_3min"


arguments = {
    'root': data_dir,
    "numz": numz,
    "end_t": end_t,
    "numc": 1,
    "anisotropy": pixel_size,
    "gaussian": False,
    'median': 3,
    "remove_blobs_dist": 3,
    "blob_merge_dist_thresh": 5,
    "mip_movie": True,
    "marker_movie": True,
    "infill": True,
    'quantile': 0.953,
    "save_threads": True,
    "save_timeseries": True,
    "suppress_output": False,
    "regen": False,
    "algorithm": segmentation_algorithm,
    "algorithm_params": {
        "peaks": [],
        "labels": [],
        "window_size": 5,
        "min_distance": 10,
        "manage_collisions": "prune",
        "fb_threshold_margin": 50,
        "gaussian_diameter": 5,
        "gaussian_sigma": 3,
        "exclude_border": False,
    },
    "predict": True,
    "register_frames": False,
    "output_dir": output_dir,
    "quant_args": {
        "bleach_correction": None,
        "quant_radius": 3,  # radius of quantification zone
        "other_pos_radius": 5,  # exclusion around adjacent rois during quantification and background subtraction
        "background_radius": 15,  # radius of background for background subtraction
        "quant_z_radius": 3,  # adjacent z's to include in quantification
        "quant_voxels": 20,  # number of pixels/voxels to quantify within mask (average of brightest n)
    },
}

#e = Extractor(**arguments)
#e.calc_blob_threads()
#e.quantify(quant_function=background_subtraction_quant_function)

e = load_extractor(output_dir)

c = Curator(e=e)