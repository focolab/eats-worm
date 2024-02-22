#!/usr/bin/env python3
## Example use case of eats-worm package

from eats_worm import *

## pixelSize = [0.35, 0.75, 0.3545×sin(π×Θ°/180°)]
#m = MultiFileTiff('D:/SCAPE/Data/Voleti_et_al_Nat_Meth_2019/Moving_actual_properScale/green_actual_properScale', numz=127, numc=1, anisotropy=(0.32, 0.32, 0.32))
#m = MultiFileTiff('D:/SCAPE/Data/Zyla/165mm/NeuroPAL/tiff_stacks/20230726_OH16290_1_run2', numz=127, numc=1, anisotropy=(0.25, 0.75, 0.35))
#m = MultiFileTiff('E:/SCAPE/Data/HiCAM_2000/165mm/GreenBeads/20230911/tiff_stacks/20230911_GreenBeads4um_run3/Deskewed_-60', numz=200, numc=1, anisotropy=(0.39, 1, 0.55))
#m = MultiFileTiff('E:/SCAPE/Data/HiCAM_2000/165mm/GreenBeads/20230912/tiff_stacks/20230912_GreenBeads4um_run1/Deskewed_-45', numz=200, numc=1, anisotropy=(0.39, 1, 0.55))
m = MultiFileTiff('E:/SCAPE/Data/HiCAM_2000/165mm/NeuroPAL/20230917/tiff_stacks/20230917_OH16290_3_run1/Deskewed_-45', numz=200, numc=1, anisotropy=(0.39, 1, 0.55))
#m = MultiFileTiff('E:/SCAPE/Data/HiCAM_2000/165mm/NeuroPAL/20230924/tiff_stacks/20230924_OH16290_5_run1/Deskewed_-45', numz=200, numc=1, anisotropy=(0.39, 1, 0.55))
#m = MultiFileTiff('E:/SCAPE/Data/HiCAM_2000/165mm/NeuroPAL/20230926/tiff_stacks/20230926_OH16290_1_run1/Deskewed_-45', numz=200, numc=1, anisotropy=(0.39, 0.75, 0.55))
#m = MultiFileTiff('E:/SCAPE/Data/HiCAM_2000/165mm/NeuroPAL/20230928/tiff_stacks/20230928_OH16290_1_run1/Deskewed_-40', numz=200, numc=1, anisotropy=(0.35, 0.75, 0.55))

viewer = napari.Viewer(ndisplay=3)
viewer.add_image(m.get_dask_array(), scale=m.anisotropy)
napari.run()