from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTabWidget, QPushButton, QLineEdit, QLabel, QGridLayout


class MainWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.initUI()

    def initUI(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)

        self.loading_param = QWidget()

        param_box_layout = QGridLayout()
        numz = QLineEdit()
        numc = QLineEdit()
        anisotropy = QLineEdit()
        btn_apply_loading_param = QPushButton("Apply param")
        btn_apply_loading_param.clicked.connect(self.apply_loading_param)
        param_box_layout.addWidget(QLabel("Num. Z"), 1, 0)
        param_box_layout.addWidget(numz, 1, 1)
        param_box_layout.addWidget(QLabel("Num. Color"), 2, 0)
        param_box_layout.addWidget(numc, 2, 1)
        param_box_layout.addWidget(QLabel("Anisotropy"), 3, 0)
        param_box_layout.addWidget(anisotropy, 3, 1)
        param_box_layout.addWidget(btn_apply_loading_param, 4, 0, 1, 2)
        self.loading_param.setLayout(param_box_layout)

        main_layout.addWidget(self.loading_param)
        main_layout.addStretch(1)
        self.setLayout(main_layout)

    def apply_loading_param(self):
        pass
