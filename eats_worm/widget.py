import ast
import os.path

from napari.layers import Image
from napari.utils.notifications import show_warning, show_error
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLineEdit, QLabel, QGridLayout, QComboBox

from eats_worm import MultiFileTiff


class MainWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.initUI()

    def initUI(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)

        loading_param = QWidget()
        param_box_layout = QGridLayout()
        self.load_path = QLineEdit()
        self.numz = QLineEdit()
        self.numc = QLineEdit()
        self.anisotropy = QLineEdit()
        btn_apply_loading_param = QPushButton("Reconstruct")
        btn_apply_loading_param.clicked.connect(self.apply_loading_param)
        param_box_layout.addWidget(QLabel("Path"), 1, 0)
        param_box_layout.addWidget(self.load_path, 1, 1)
        param_box_layout.addWidget(QLabel("Num. Z"), 2, 0)
        param_box_layout.addWidget(self.numz, 2, 1)
        param_box_layout.addWidget(QLabel("Num. Color"), 3, 0)
        param_box_layout.addWidget(self.numc, 3, 1)
        param_box_layout.addWidget(QLabel("Anisotropy"), 4, 0)
        param_box_layout.addWidget(self.anisotropy, 4, 1)
        param_box_layout.addWidget(btn_apply_loading_param, 5, 0, 1, 2)

        self.dim_to_slice = QComboBox()
        btn_slice_data = QPushButton('Slice')
        btn_slice_data.clicked.connect(self.slice_dim)
        param_box_layout.addWidget(QLabel("Dimension"), 6, 0)
        param_box_layout.addWidget(self.dim_to_slice, 6, 1)
        param_box_layout.addWidget(btn_slice_data, 7, 0, 1, 2)
        loading_param.setLayout(param_box_layout)

        main_layout.addWidget(loading_param)
        main_layout.addStretch(1)
        self.setLayout(main_layout)

    def apply_loading_param(self):
        if selected_layer := self.get_current_layer(Image):
            anisotropy = ast.literal_eval(self.anisotropy.text())
            if not isinstance(anisotropy, tuple):
                show_error('Anisotropy must be Tuple.')
                return
            if os.path.isdir(self.load_path.text()):
                m = MultiFileTiff(
                    self.load_path.text(),
                    numz=int(self.numz.text()),
                    numc=int(self.numc.text()),
                    anisotropy=ast.literal_eval(self.anisotropy.text())
                )
                selected_layer.data = m.get_dask_array()
                selected_layer.refresh()
            else:
                show_warning('Path is not a directory.')

    def slice_dim(self):
        if selected_layer := self.get_current_layer(Image):
            dim = int(self.dim_to_slice.currentText())
            idx = self.viewer.dims.current_step[dim]
            slicing = [slice(None)] * selected_layer.data.ndim
            slicing[dim] = idx
            selected_layer.data = selected_layer.data[tuple(slicing)]
            selected_layer.refresh()
            self.update_dimension(selected_layer.data.ndim)

    def get_current_layer(self, target_type=None, trigger_error=True):
        if len(self.viewer.layers.selection) < 1:
            if trigger_error:
                show_error("No layer is selected.")
            return None
        elif len(self.viewer.layers.selection) > 1:
            if trigger_error:
                show_error("Multiple layers are selected. Select only one label layer.")
            return None
        current_layer = next(iter(self.viewer.layers.selection))
        if target_type is None:
            return current_layer
        else:
            if not isinstance(current_layer, target_type):
                if trigger_error:
                    show_error(f"The selected layer must be a {target_type.__name__} layer.")
                return None
            return current_layer

    def update_dimension(self, dims=4):
        self.dim_to_slice.clear()
        self.dim_to_slice.addItems([str(i) for i in (range(dims))])
