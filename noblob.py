#!/usr/bin/env python3
## Example use case of eats-worm package

from eats_worm import *

## pixelSize = [0.35, 0.75, 0.3545×sin(π×Θ°/180°)]
m = MultiFileTiff('D:/SCAPE/Data/Voleti_et_al_Nat_Meth_2019/Moving_actual_properScale/green_actual_properScale', numz=127, numc=1, anisotropy=(1.42, 0.37, 0.32))
#m = MultiFileTiff('E:/SCAPE/Data/HiCAM_2000/165mm/NeuroPAL/20230914/tiff_stacks/20230914_OH16290_3_run1/Deskewed_-45', numz=200, numc=1, anisotropy=(0.55, 0.5, 0.47))
#m = MultiFileTiff('E:/SCAPE/Data/HiCAM_2000/165mm/NeuroPAL/20230915/tiff_stacks/20230915_OH16290_3_run1/Deskewed_-45', numz=200, numc=1, anisotropy=(0.55, 1, 0.47))
#m = MultiFileTiff('E:/SCAPE/Data/HiCAM_2000/165mm/GreenBeads/20230915/tiff_stacks/20230915_FocalCheck15um_run1/Deskewed_-45', numz=200, numc=1, anisotropy=(0.55, 1, 0.47))

viewer = napari.Viewer(ndisplay=3)
viewer.add_image(m.get_dask_array(), scale=m.anisotropy)
napari.run()