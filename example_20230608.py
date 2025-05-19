#!/usr/bin/env python3
## Example use case of eats-worm package

from eats_worm import *

#data_directory = "C:/Users/Muneki/Documents/MATLAB/SCAPE/Data/50mm/NeuroPAL/tiff_stacks/unCorrected_20221127_OH16290_1_run1"; numZ = 50; pixelSize = [1, 1, 1]
#data_directory = "C:/Users/Muneki/Documents/MATLAB/SCAPE/Data/50mm/NeuroPAL/tiff_stacks/20221127_OH16290_1_run1"; numZ = 35; pixelSize = [1, 1, 1]
#data_directory = "C:/Users/Muneki/Documents/MATLAB/SCAPE/Data/50mm/NeuroPAL/tiff_stacks/20221127_OH16290_1_run2; numZ = 34; pixelSize = [1, 1, 1]
#data_directory = "C:/Users/Muneki/Documents/MATLAB/SCAPE/Data/50mm/NeuroPAL/tiff_stacks/unCorrected_20221129_OH16290_1_run1"; numZ = 40; pixelSize = [1, 1, 1]
#data_directory = "C:/Users/Muneki/Documents/MATLAB/SCAPE/Data/50mm/NeuroPAL/tiff_stacks/20221129_OH16290_1_run1"; numZ = 38; pixelSize = [1, 1, 1]
#data_directory = "C:/Users/Muneki/Documents/MATLAB/SCAPE/Data/165mm/NeuroPAL/tiff_stacks/unCorrected_20221201_OH16290_1_run2"; numZ = 30; pixelSize = [1, 1, 1]
#data_directory = "C:/Users/Muneki/Documents/MATLAB/SCAPE/Data/165mm/NeuroPAL/tiff_stacks/20221201_OH16290_1_run2"; numZ = 134; pixelSize = [1, 1, 1]
#data_directory = "C:/Users/Muneki/Documents/MATLAB/SCAPE/Data/165mm/NeuroPAL/tiff_stacks/unCorrected_20230102_OH16290_1_run2"; numZ = 70; pixelSize = [1, 1, 1]
#data_directory = "C:/Users/Muneki/Documents/MATLAB/SCAPE/Data/165mm/NeuroPAL/tiff_stacks/unCorrected_20230102_OH16290_1_run3"; numZ = 70; pixelSize = [1, 1, 1]
#data_directory = "C:/Users/Muneki/Documents/MATLAB/SCAPE/Data/165mm/NeuroPAL/tiff_stacks/unCorrected_20230125_OH16230_1_run3_HR"; numZ = 500; pixelSize = [1, 1, 1]
#data_directory = "C:/Users/miked/Documents/MATLAB/SCAPE/Data/165mm/NeuroPAL/tiff_stacks/20230316_OH16230_1_run2_HR"; numZ = 799; pixelSize = [1, 1, 1]
#data_directory = "C:/Users/miked/OneDrive - UCSF/Documents/MATLAB/SCAPE/Data/Voleti_et_al_Nat_Meth_2019/Immobilized_actual_properScale/green_actual_properScale"; numZ = 157; endT = 1; pixelSize = [0.75, 0.24, 0.20]
#data_directory = "C:/Users/miked/OneDrive - UCSF/Documents/MATLAB/SCAPE/Data/Voleti_et_al_Nat_Meth_2019/Moving_actual_properScale/green_actual_properScale"; numZ = 127; endT = 2; pixelSize = [1.42, 0.37, 0.32]
#data_directory = "C:/Users/miked/OneDrive - UCSF/Documents/MATLAB/SCAPE/Data/165mm/NeuroPAL/tiff_stacks/20230522_OH16290_1_run2"; numZ = 199; endT = 1; pixelSize = [0.35, 0.75, 0.29]
data_directory = "C:/Users/miked/OneDrive - UCSF/Documents/MATLAB/SCAPE/Data/165mm/NeuroPAL/tiff_stacks/20230529_OH16290_1_run1"; numZ = 127; endT = 1; pixelSize = [0.35, 0.75, 0.29]
#data_directory = "C:/Users/miked/OneDrive - UCSF/Documents/MATLAB/SCAPE/Data/165mm/NeuroPAL/tiff_stacks/20230529_OH16290_2_run1"; numZ = 127; endT = 1; pixelSize = [0.35, 0.75, 0.29]
output_directory = data_directory + "/output/"


arguments = {
    "root": data_directory,
    "numz": numZ,
    #"frames":[1,2,3,4,5,6,7,8,9,10,11],
    "numc": 1,
    "end_t": endT,
    "offset": 0,
    "gaussian": False,
    "median": 3, #5
    "quantile": 0.963, #0.999
    "anisotropy": pixelSize,
    "blob_merge_dist_thresh": 5,
    "remove_blobs_dist": 3,
    "algorithm":"tmip_2d_template",
    "algorithm_params": {
        "window_size": 5, #10
        "min_distance": 1, #11
        "manage_collisions": "prune",
        "fb_threshold_margin": 50
    },
    "register_frames": True,
    "output_dir": output_directory
}

e = Extractor(**arguments)

e.calc_blob_threads()
e.quantify(quant_function=background_subtraction_quant_function)

#e = load_extractor(output_directory)

c = Curator(e=e)