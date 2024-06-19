#!/usr/bin/env python3
## Example use case of eats-worm package

from eats_worm import *
import napari
import numpy as np
from qtpy.QtWidgets import QPushButton, QVBoxLayout, QWidget
from qtpy.QtCore import QTimer

## pixelSize = [0.35, 0.75, 0.3545×sin(π×Θ°/180°)]
#m = MultiFileTiff('E:/SCAPE/Data/Voleti_et_al_Nat_Meth_2019/Moving_actual_properScale/green_actual_properScale', numz=127, numc=1, anisotropy=(0.32, 0.32, 0.32))
#m = MultiFileTiff('E:/SCAPE/Data/Zyla/165mm/NeuroPAL/tiff_stacks/20230726_OH16290_1_run2', numz=127, numc=1, anisotropy=(0.25, 0.75, 0.35))
#m = MultiFileTiff('E:/SCAPE/Data/HiCAM_2000/165mm/GreenBeads/20230911/tiff_stacks/20230911_GreenBeads4um_run3/Deskewed_-60', numz=200, numc=1, anisotropy=(0.39, 1, 0.55))

#m = MultiFileTiff('D:/HiCAM_2000/GreenBeads/20240526/tiff_stacks/20240526_GreenBeads4um_run6/Deskewed_-45', numz=800, numc=1, anisotropy=(0.39, 1, 0.55))
#m = MultiFileTiff('D:/HiCAM_2000/NeuroPAL/20240525/tiff_stacks/20240525_OH16289_1_run1/Deskewed_-45', numz=84, numc=1, anisotropy=(0.39, 1, 0.55))
#m = MultiFileTiff('D:/HiCAM_2000/NeuroPAL/20240618/tiff_stacks/20240618_OH16290_1_SmallArena_v4_run2/Deskewed_-41', end_t = 1365, numz=200, numc=1, anisotropy=(0.36, 1, 0.55))

#m = MultiFileTiff('E:/HiCAM_2000/GreenBeads/20240526_GreenBeads4um_run6/Deskewed_-45', numz=800, numc=1, anisotropy=(0.39, 1, 0.55))
#m = MultiFileTiff('E:/HiCAM_2000/NeuroPAL/20240525_OH16289_1_run1/Deskewed_-45', numz=84, numc=1, anisotropy=(0.39, 1, 0.55))
m = MultiFileTiff('E:/HiCAM_2000/NeuroPAL/20240618_OH16290_1_SmallArena_v4_run4/Deskewed_-41', end_t = 1365, numz=200, numc=1, anisotropy=(0.36, 1, 0.55))

shape = (512, 512)
data = np.random.random((m.end_t, *shape))

viewer = napari.Viewer(ndisplay=3)
layer = viewer.add_image(m.get_dask_array(), scale=m.anisotropy)

# 再生機能の設定
def play_movie():
    current_frame = 0

    def update_frame():
        nonlocal current_frame
        if current_frame < m.end_t:
            viewer.dims.set_current_step(0, current_frame)
            current_frame += 1
        else:
            timer.stop()

    timer = QTimer()
    timer.timeout.connect(update_frame)
    timer.start(85)  # フレーム間の遅延（ミリ秒単位、ここでは100ミリ秒）

# カスタムウィジェットの作成
class PlayWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.button = QPushButton("Play")
        self.button.clicked.connect(play_movie)
        layout = QVBoxLayout()
        layout.addWidget(self.button)
        self.setLayout(layout)

# カスタムウィジェットをドッキング
play_widget = PlayWidget()
viewer.window.add_dock_widget(play_widget, name='Play', area='left')

napari.run()