import os.path
from napari.utils.notifications import show_warning
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLineEdit, QLabel, QGridLayout

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
        btn_apply_loading_param = QPushButton("Load tiffs")
        btn_apply_loading_param.clicked.connect(self.apply_loading_param)
        param_box_layout.addWidget(QLabel("Path"), 1, 0)
        param_box_layout.addWidget(self.load_path, 1, 1)
        param_box_layout.addWidget(QLabel("Num. Z"), 2, 0)
        param_box_layout.addWidget(self.numz, 2, 1)
        param_box_layout.addWidget(QLabel("Num. Color"), 3, 0)
        param_box_layout.addWidget(self.numc, 3, 1)
        param_box_layout.addWidget(QLabel("Anisotropy"), 4, 0)
        param_box_layout.addWidget(self.anisotropy, 5, 1)
        param_box_layout.addWidget(btn_apply_loading_param, 6, 0, 1, 2)
        loading_param.setLayout(param_box_layout)

        main_layout.addWidget(loading_param)
        main_layout.addStretch(1)
        self.setLayout(main_layout)

    def apply_loading_param(self):
        if os.path.isdir(self.load_path.text()):
            m = MultiFileTiff(
                self.load_path.text(),
                numz=int(self.numz),
                numc=int(self.numc),
                anisotropy=self.anisotropy
            )
            self.viewer.layers[-1].data = m.get_dask_array()
            self.viewer.refresh()
        else:
            show_warning('Path is not a directory.')
