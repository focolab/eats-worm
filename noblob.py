#!/usr/bin/env python3
## Example use case of eats-worm package

from eats_worm import *

## pixelSize = [0.35, 0.75, 0.3545×sin(π×Θ°/180°)]
#m = MultiFileTiff('E:/SCAPE/Data/Voleti_et_al_Nat_Meth_2019/Moving_actual_properScale/green_actual_properScale', numz=127, numc=1, anisotropy=(0.32, 0.32, 0.32))
#m = MultiFileTiff('E:/SCAPE/Data/Zyla/165mm/NeuroPAL/tiff_stacks/20230726_OH16290_1_run2', numz=127, numc=1, anisotropy=(0.25, 0.75, 0.35))
#m = MultiFileTiff('E:/SCAPE/Data/HiCAM_2000/165mm/GreenBeads/20230911/tiff_stacks/20230911_GreenBeads4um_run3/Deskewed_-60', numz=200, numc=1, anisotropy=(0.39, 1, 0.55))

#m = MultiFileTiff('D:/HiCAM_2000/GreenBeads/20240615/tiff_stacks/20240615_GreenBeads4um_run2/Deskewed_-41', numz=800, numc=1, anisotropy=(0.35, 1, 0.55))
#m = MultiFileTiff('D:/HiCAM_2000/NeuroPAL/20240525/tiff_stacks/20240525_OH16289_1_run1/Deskewed_-45', numz=84, numc=1, anisotropy=(0.39, 1, 0.55))
m = MultiFileTiff('D:/HiCAM_2000/NeuroPAL/20240616/tiff_stacks/20240616_OH16290_2_SmallArena_v4_run3/Deskewed_-41', numz=200, numc=1, anisotropy=(0.36, 1, 0.55))

#m = MultiFileTiff('E:/HiCAM_2000/GreenBeads/20240526_GreenBeads4um_run6/Deskewed_-45', numz=800, numc=1, anisotropy=(0.39, 1, 0.55))
#m = MultiFileTiff('E:/HiCAM_2000/NeuroPAL/20240525_OH16289_1_run1/Deskewed_-45', numz=84, numc=1, anisotropy=(0.39, 1, 0.55))
#m = MultiFileTiff('E:/HiCAM_2000/NeuroPAL/20240527_OH16290_SmallArenav4_run2/Deskewed_-45', numz=115, numc=1, anisotropy=(0.39, 1, 0.55))

viewer = napari.Viewer(ndisplay=3)
viewer.add_image(m.get_dask_array(), scale=m.anisotropy)
napari.run()