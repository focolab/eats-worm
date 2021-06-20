import atexit
import json
import napari
import numpy as np
import pyqtgraph as pg

from .Extractor import *
from .multifiletiff import *
from .Threads import *
from qtpy.QtWidgets import QAbstractItemView, QAction, QSlider, QButtonGroup, QGridLayout, QLabel, QListWidget, QListWidgetItem, QMenu, QPushButton, QRadioButton, QWidget
from qtpy.QtCore import Qt, QPoint, QSize
from qtpy.QtGui import QBrush, QCursor, QIcon, QImage, QPen, QPixmap

def subaxis(im, position, window = 100):
    """
    returns window around specified center of image, along with an an offset that gives how much the image was offset (and thus yields the center of the image)

    Parameters
    ----------
    im : np.array
        3d numpy array representing volume at a particular time point
    z : int
    x : int
    y : int
        pixel indices of the center
    window : int
        size of image window around center

    Returns
    -------
    im : np.array
        2d image array of size window by window
    offset : int
        integer offset 
    """

    z,y,x = position
    z = int(z)
    xmin,xmax = int(x-window//2),int(x+window//2)
    ymin,ymax = int(y-window//2),int(y+window//2)

    offset = [0,0]
    if xmin < 0:
        xmax -= xmin
        xmin = 0
        offset[0] = -xmin
    if ymin < 0:
        ymax -= ymin
        ymin = 0 
        offset[1] = -ymin
    if ymax >= im.shape[-2]:
        ymin -= ymax-im.shape[-2]+1
        offset[1] = (ymax-im.shape[-2] )
        ymax = im.shape[-2]-1

    if xmax >= im.shape[-1]:
        xmin -= xmax - im.shape[-1]+1
        offset[0] = (xmax - im.shape[-1]+1)
        xmax = im.shape[-1]-1
    #offset[0],offset[1] = offset[1], offset[0]
    return im[ymin:ymax, xmin:xmax], offset

def subaxis_MIP(im, z,x,y, window = 100):
    """
    returns window around specified center of MIP image, along with an an offset that gives how much the image was offset (and thus yields the center of the image). 

    Parameters
    ----------
    im : np.array
        3d numpy array representing volume at a particular time point
    z : int
    x : int
    y : int
        pixel indices of the center
    window : int
        size of image window around center

    Returns
    -------
    im : np.array
        2d image array of size window by window
    offset : int
        integer offset 
    """
    z = z
    xmin,xmax = x-window//2,x+window//2
    ymin,ymax = y-window//2,y+window//2

    offset = [0,0]
    if xmin < 0:
        xmax -= xmin
        xmin = 0
        offset[0] = xmin
    if ymin < 0:
        ymax -= ymin
        ymin = 0 
        offset[1] = ymin
    if ymax > im.shape[-2]:
        ymin -= ymax-im.shape[-2] 
        offset[1] = ymax-im.shape[-2] 
        ymax = im.shape[-2]-1

    if xmax > im.shape[-1]:
        xmin -= xmax - im.shape[-1]
        offset[0] = xmax - im.shape[-1]
        xmax = im.shape[-1]-1


    #print(offset)
    return im[ymin:ymax, xmin:xmax], offset

class Curator:
    """
    matplotlib display of scrolling image data 
    
    Parameters
    ---------
    extractor : extractor
        extractor object containing a full set of infilled threads and time series

    Attributes
    ----------
    ind : int
        thread indexing 

    min : int
        min of image data (for setting ranges)

    max : int
        max of image data (for setting ranges)

    """
    def __init__(self,e,window=100):
        # get info from extractors
        self.s = e.spool
        self.timeseries = e.timeseries
        self.tf = e.im
        self.tf.t = 0
        self.window = window
        ## num neurons
        self.numneurons = len(self.s.threads)
        self.num_frames = len(e.frames)

        self.path = e.root + 'extractor-objects/curate.json'
        self.ind = 0
        try:
            with open(self.path) as f:
                self.curate = json.load(f)
            
            self.ind = int(self.curate['last'])
        except:
            self.curate = {}
            self.ind = 0
            self.curate['0']='seen'


        # array to contain internal state: whether to display single ROI, ROI in Z, or all ROIs
        self.pointstate = 0
        self.show_settings = 0
        self.showmip = 0

        ## index for which time point to display
        self.t = 0

        ### First frame of the first thread
        self.update_ims()

        ## Display range 
        self.min = np.min(np.nonzero([self.im, self.im_plus_one, self.im_minus_one]))
        self.max = np.max([self.im, self.im_plus_one, self.im_minus_one]) # just some arbitrary value
        
        ## maximum t
        self.tmax = e.t

        self.restart()
        atexit.register(self.log_curate)

    def __init__(self, mft=None, spool=None, timeseries=None, window=100):
        # get info from extractors
        self.tf = mft
        if self.tf:
            self.tf.t = 0
        self.s = spool
        self.timeseries = timeseries
        self.window = window
        ## num neurons
        self.numneurons = len(self.s.threads) if self.s else 0
        self.num_frames = len(self.tf.frames) if self.tf else 0

        self.path = self.tf.root + 'extractor-objects/curate.json' if self.tf else None
        self.ind = 0
        try:
            with open(self.path) as f:
                self.curate = json.load(f)
            
            self.ind = int(self.curate['last'])
        except:
            self.curate = {}
            self.ind = 0
            self.curate['0']='seen'


        # array to contain internal state: whether to display single ROI, ROI in Z, or all ROIs
        self.pointstate = 0
        self.show_settings = 0
        self.showmip = 0

        ## index for which time point to display
        self.t = 0

        ### First frame of the first thread
        self.update_ims()

        ## Display range 
        self.min = np.min(np.nonzero([self.im, self.im_plus_one, self.im_minus_one]))
        self.max = np.max([self.im, self.im_plus_one, self.im_minus_one]) # just some arbitrary value
        
        ## maximum t
        self.tmax = (self.tf.numframes-self.tf.offset)//self.tf.numz if self.tf else 0

        self.restart()
        atexit.register(self.log_curate)

    def restart(self):
        ### enable antialiasing in pyqtgraph
        pg.setConfigOption('antialias', True)

        ### initialize napari viewer
        self.viewer = napari.Viewer(ndisplay=3)
        self.scale = [5, 1, 1]
        if self.tf:
            self.viewer.add_image(self.tf.get_t(self.t), name='volume', scale=self.scale)
        if self.s:
            self.viewer.add_points(np.empty((0, 3)), face_color='blue', edge_color='blue', name='other rois', size=1, scale=self.scale)
            self.viewer.add_points(np.empty((0, 3)), face_color='red', edge_color='red', name='roi', size=1, scale=self.scale)

        ### initialize views for images
        self.z_view = self.get_imageview()
        self.z_plus_one_view = self.get_imageview()
        self.z_minus_one_view = self.get_imageview()
        self.z_subim = self.get_imageview()
        self.z_plus_one_subim = self.get_imageview()
        self.z_minus_one_subim = self.get_imageview()
        self.ortho_1_view = self.get_imageview()
        self.ortho_2_view = self.get_imageview()
        self.timeseries_view = pg.PlotWidget()
        self.timeseries_view.setBackground('w')
        if self.timeseries:
            self.timeseries_view.plot((self.timeseries[:,self.ind]-np.min(self.timeseries[:,self.ind]))/(np.max(self.timeseries[:,self.ind])-np.min(self.timeseries[:,self.ind])), pen='b')
            self.timeseries_view.addLine(x=self.t, pen='r')

        ### Series label
        self.series_label = QLabel()
        self.viewer.window.add_dock_widget(self.series_label, area='right')

        ### figure grid
        image_grid_container = QWidget()
        image_grid = QGridLayout(image_grid_container)
        image_grid.addWidget(self.z_plus_one_view, 0, 0)
        image_grid.addWidget(self.z_view, 1, 0)
        image_grid.addWidget(self.z_minus_one_view, 2, 0)
        image_grid.addWidget(self.z_plus_one_subim, 0, 1)
        image_grid.addWidget(self.z_subim, 1, 1)
        image_grid.addWidget(self.z_minus_one_subim, 2, 1)
        image_grid.addWidget(self.timeseries_view, 1, 2)
        image_grid.addWidget(self.ortho_1_view, 0, 2)
        image_grid.addWidget(self.ortho_2_view, 2, 2)
        image_grid.setColumnStretch(0, 2)
        image_grid.setColumnStretch(1, 1)
        image_grid.setColumnStretch(2, 2)
        self.viewer.window.add_dock_widget(image_grid_container, area='bottom', name='image_grid')

        ### initialize figures
        self.update_figures()
        self.update_timeseries()

        ### Axis for setting min/max range
        min_r_slider = QSlider()
        min_r_slider.setMaximum(int(np.max(self.im)))
        min_r_slider.setTickPosition(int(self.min))
        min_r_slider.setValue(int(self.min))
        min_r_slider.setOrientation(Qt.Horizontal)
        min_r_slider.valueChanged.connect(lambda:self.update_mm("min", min_r_slider.value()))
        max_r_slider = QSlider()
        max_r_slider.setMaximum(int(np.max(self.im)*4))
        max_r_slider.setTickPosition(int(self.max))
        max_r_slider.setValue(int(self.max))
        max_r_slider.setOrientation(Qt.Horizontal)
        max_r_slider.valueChanged.connect(lambda:self.update_mm("max", max_r_slider.value()))
        self.viewer.window.add_dock_widget([QLabel('R Min'), min_r_slider, QLabel('R Max'), max_r_slider], area='right')

        ### Axis for scrolling through t
        t_slider = QSlider()
        t_slider.setMaximum(int(self.tmax-1))
        t_slider.setValue(int(self.t))
        t_slider.setOrientation(Qt.Horizontal)
        t_slider.valueChanged.connect(lambda:self.update_t(t_slider.value()))
        self.viewer.window.add_dock_widget([QLabel('Timepoint'), t_slider], area='right')

        #### Axis for button for display
        points_button_group = [QRadioButton('Single'), QRadioButton('Same Z'), QRadioButton('All')]
        points_button_group[0].setChecked(True)
        points_button_group[0].toggled.connect(lambda:self.update_pointstate(points_button_group[0].text()))
        points_button_group[1].toggled.connect(lambda:self.update_pointstate(points_button_group[1].text()))
        points_button_group[2].toggled.connect(lambda:self.update_pointstate(points_button_group[2].text()))
        self.viewer.window.add_dock_widget(points_button_group, area='right')

        #### Axis for whether to display MIP on left
        mip_button_group = [QRadioButton('Single Z'), QRadioButton('MIP')]
        mip_button_group[0].setChecked(True)
        mip_button_group[0].toggled.connect(lambda:self.update_mipstate(mip_button_group[0].text()))
        mip_button_group[1].toggled.connect(lambda:self.update_mipstate(mip_button_group[1].text()))
        self.viewer.window.add_dock_widget(mip_button_group, area='right')

        ### Axis for button to keep
        self.keep_button_group = QButtonGroup()
        self.keep_button = QRadioButton('Keep')
        self.trash_button = QRadioButton('Trash')
        self.keep_button_group.addButton(self.keep_button)
        self.keep_button_group.addButton(self.trash_button)
        self.keep_button_group.buttonClicked.connect(lambda:self.keep_current(self.keep_button_group.checkedButton().text()))
        self.viewer.window.add_dock_widget(self.keep_button_group.buttons(), area='right')

        ### Axis to determine which ones to show
        show_button_group = [QRadioButton('All'), QRadioButton('Unlabelled'), QRadioButton('Kept'), QRadioButton('Trashed')]
        show_button_group[0].setChecked(True)
        show_button_group[0].toggled.connect(lambda:self.show(show_button_group[0].text()))
        show_button_group[1].toggled.connect(lambda:self.show(show_button_group[1].text()))
        show_button_group[2].toggled.connect(lambda:self.show(show_button_group[2].text()))
        show_button_group[3].toggled.connect(lambda:self.show(show_button_group[3].text()))
        self.viewer.window.add_dock_widget(show_button_group, area='right')

        ### Axis for buttons for next/previous time series
        bprev = QPushButton('Previous')
        bprev.clicked.connect(lambda:self.prev())
        bnext = QPushButton('Next')
        bnext.clicked.connect(lambda:self.next())
        self.viewer.window.add_dock_widget([bprev, bnext], area='right')

        ### Grid showing all extracted timeseries
        self.trace_grid = QListWidget()
        self.trace_grid.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.trace_grid.setViewMode(QListWidget.IconMode)
        self.trace_grid.setIconSize(QSize(96, 96))
        self.trace_grid.itemDoubleClicked.connect(lambda:self.go_to_trace(int(self.trace_grid.currentItem().text())))
        self.trace_grid.setContextMenuPolicy(Qt.CustomContextMenu)
        self.trace_grid_context_menu = QMenu(self.trace_grid)
        keep_all_action = QAction('Keep selected', self.trace_grid, triggered = self.keep_all_selected)
        self.trace_grid_context_menu.addAction(keep_all_action)
        trash_all_action = QAction('Trash selected', self.trace_grid, triggered = self.trash_all_selected)
        self.trace_grid_context_menu.addAction(trash_all_action)
        self.trace_grid.customContextMenuRequested[QPoint].connect(self.show_trace_grid_context_menu)
        self.set_trace_icons()
        self.viewer.window.add_dock_widget(self.trace_grid)

        ### Update buttons in case of previous curation
        self.update_buttons()
        napari.run()
    
    ## Attempting to get autosave when instance gets deleted, not working right now TODO     
    def __del__(self):
        self.log_curate()

    def update_ims(self):
        if not self.tf:
            self.im = np.ones((1, 1))
            self.im_plus_one = np.ones((1, 1))
            self.im_minus_one = np.ones((1, 1))
            self.subim, self.offset = np.ones((1, 1)), 0
            self.subim_plus_one = np.ones((1, 1))
            self.subim_minus_one = np.ones((1, 1))
    
        else:
            f = int(self.s.threads[self.ind].get_position_t(self.t)[0])
            if self.showmip:
                self.im = np.max(self.tf.get_t(self.t),axis = 0)
            else:
                self.im = self.tf.get_tbyf(self.t, f)
            if f == self.num_frames - 1:
                self.im_plus_one = np.zeros(self.im.shape)
            else:
                self.im_plus_one = self.tf.get_tbyf(self.t, f + 1)
            if f == 0:
                self.im_minus_one = np.zeros(self.im.shape)
            else:
                self.im_minus_one = self.tf.get_tbyf(self.t, f - 1)
            self.subim, self.offset = subaxis(self.im, self.s.threads[self.ind].get_position_t(self.t), self.window)
            self.subim_plus_one, _ = subaxis(self.im_plus_one, self.s.threads[self.ind].get_position_t(self.t), self.window)
            self.subim_minus_one, _ = subaxis(self.im_minus_one, self.s.threads[self.ind].get_position_t(self.t), self.window)
    
    def get_im_display(self, im):
        return (im - self.min)/(self.max - self.min)
    
    def update_figures(self):
        if self.tf:
            self.viewer.layers['volume'].data = self.tf.get_t(self.t)
            self.update_imageview(self.ortho_1_view, np.max(self.tf.get_t(self.t), axis=1), "Ortho MIP ax 1")
            self.update_imageview(self.ortho_2_view, np.max(self.tf.get_t(self.t), axis=2), "Ortho MIP ax 2")
        if self.s:
            self.viewer.layers['roi'].data = np.array([self.s.threads[self.ind].get_position_t(self.t)])
            if self.pointstate==0:
                self.viewer.layers.data = np.empty((0, 3))
            elif self.pointstate==1:
                self.viewer.layers['other rois'].data = self.s.get_positions_t_z(self.t, self.s.threads[self.ind].get_position_t(self.t)[0])
            elif self.pointstate==2:
                self.viewer.layers['other rois'].data = self.s.get_positions_t(self.t)

        self.update_imageview(self.z_view, self.get_im_display(self.im), "Parent Z")
        self.update_imageview(self.z_plus_one_view, self.get_im_display(self.im_plus_one), "Z + 1")
        self.update_imageview(self.z_minus_one_view, self.get_im_display(self.im_minus_one), "Z - 1")

        if self.s:
            # plotting for multiple points
            if self.pointstate==0:
                pass
            elif self.pointstate==1:
                self.plot_on_imageview(self.z_view, self.s.get_positions_t_z(self.t, self.s.threads[self.ind].get_position_t(self.t)[0])[:,2], self.s.get_positions_t_z(self.t,self.s.threads[self.ind].get_position_t(self.t)[0])[:,1], Qt.blue)
            elif self.pointstate==2:
                self.plot_on_imageview(self.z_view, self.s.get_positions_t(self.t)[:,2], self.s.get_positions_t(self.t)[:,1], Qt.blue)
                self.plot_on_imageview(self.z_plus_one_view, self.s.get_positions_t(self.t)[:,2], self.s.get_positions_t(self.t)[:,1], Qt.blue)
                self.plot_on_imageview(self.z_minus_one_view, self.s.get_positions_t(self.t)[:,2], self.s.get_positions_t(self.t)[:,1], Qt.blue)
            self.plot_on_imageview(self.z_view, [self.s.threads[self.ind].get_position_t(self.t)[2]], [self.s.threads[self.ind].get_position_t(self.t)[1]], Qt.red)

            #plotting for single point
            self.update_imageview(self.z_subim, self.get_im_display(self.subim), "Parent Z")
            self.update_imageview(self.z_plus_one_subim, self.get_im_display(self.subim_plus_one), "Z + 1")
            self.update_imageview(self.z_minus_one_subim, self.get_im_display(self.subim_minus_one), "Z - 1")

            self.plot_on_imageview(self.z_subim, [self.window/2+self.offset[0]], [self.window/2+self.offset[1]], Qt.red)

        if self.timeseries:
            self.series_label.setText('Series=' + str(self.ind) + ', Z=' + str(int(self.s.threads[self.ind].get_position_t(self.t)[0])))

    def update_timeseries(self):
        self.timeseries_view.clear()
        if self.timeseries:
            self.timeseries_view.plot((self.timeseries[:,self.ind]-np.min(self.timeseries[:,self.ind]))/(np.max(self.timeseries[:,self.ind])-np.min(self.timeseries[:,self.ind])), pen=pg.mkPen(color=(31, 119, 180), width=3))
            self.timeseries_view.addLine(x=self.t, pen='r')
            self.timeseries_view.setTitle('Series=' + str(self.ind) + ', Z=' + str(int(self.s.threads[self.ind].get_position_t(self.t)[0])), color='#000')

    def update_t(self, val):
        # Update index for t
        self.t = val
        # update image for t
        self.update_ims()
        self.update_figures()
        self.update_timeseries()

    def update_mm(self, button, val):
        if 'min' == button:
            self.min = val
        elif 'max' == button:
            self.max = val
        self.update_figures()

    def next(self):
        self.set_index_next()
        self.update_ims()
        self.update_figures()
        self.update_timeseries()
        self.update_buttons()
        self.update_curate()
        self.update_trace_icons()

    def prev(self):
        self.set_index_prev()
        self.update_ims()
        self.update_figures()
        self.update_timeseries()
        self.update_buttons()
        self.update_curate()
        self.update_trace_icons()

    def log_curate(self):
        self.curate['last'] = self.ind
        with open(self.path, 'w') as fp:
            json.dump(self.curate, fp)

    def keep_current(self, label):
        d = {
            'Keep':0,
            'Trash':1
        }
        status = d[label]
        if status == 0:
            self.curate[str(self.ind)]='keep'
        elif status == 1:
            self.curate[str(self.ind)]='trash'
        else:
            pass

    def keep_all_selected(self, label):
        selected_trace_indices = [icon.text() for icon in self.trace_grid.selectedItems()]
        for index in selected_trace_indices:
            self.curate[str(index)]='keep'
        self.update_buttons()
        self.update_trace_icons()


    def trash_all_selected(self, label):
        selected_trace_indices = [icon.text() for icon in self.trace_grid.selectedItems()]
        for index in selected_trace_indices:
            self.curate[str(index)]='trash'
        self.update_buttons()
        self.update_trace_icons()

    def update_buttons(self):
        if self.curate.get(str(self.ind))=='keep':
            self.keep_button.setChecked(True)
        elif self.curate.get(str(self.ind))=='trash':
            self.trash_button.setChecked(True)
        else:
            self.keep_button_group.setExclusive(False)
            self.keep_button.setChecked(False)
            self.trash_button.setChecked(False)
            self.keep_button_group.setExclusive(True)

    def show(self,label):
        d = {
            'All':0,
            'Unlabelled':1,
            'Kept':2,
            'Trashed':3
        }
        self.show_settings = d[label]
        self.update_trace_icons()

    def set_index_prev(self):
        if self.show_settings == 0:
            self.ind -= 1
            self.ind = self.ind % self.numneurons

        elif self.show_settings == 1:
            self.ind -= 1
            counter = 0
            while self.curate.get(str(self.ind)) in ['keep','trash'] and counter != self.numneurons:
                self.ind -= 1
                self.ind = self.ind % self.numneurons
                counter += 1
            self.ind = self.ind % self.numneurons
        elif self.show_settings == 2:
            self.ind -= 1
            counter = 0
            while self.curate.get(str(self.ind)) not in ['keep'] and counter != self.numneurons:
                self.ind -= 1
                counter += 1
            self.ind = self.ind % self.numneurons
        else:
            self.ind -= 1
            counter = 0
            while self.curate.get(str(self.ind)) not in ['trash'] and counter != self.numneurons:
                self.ind -= 1
                counter += 1
            self.ind = self.ind % self.numneurons

    def set_index_next(self):
        if self.show_settings == 0:
            self.ind += 1
            self.ind = self.ind % self.numneurons

        elif self.show_settings == 1:
            self.ind += 1
            counter = 0
            while self.curate.get(str(self.ind)) in ['keep','trash'] and counter != self.numneurons:
                self.ind += 1
                self.ind = self.ind % self.numneurons
                counter += 1
            self.ind = self.ind % self.numneurons
        elif self.show_settings == 2:
            self.ind += 1
            counter = 0
            while self.curate.get(str(self.ind)) not in ['keep'] and counter != self.numneurons:
                self.ind += 1
                counter += 1
            self.ind = self.ind % self.numneurons
        else:
            self.ind += 1
            counter = 0
            while self.curate.get(str(self.ind)) not in ['trash'] and counter != self.numneurons:
                self.ind += 1
                counter += 1
            self.ind = self.ind % self.numneurons

    def set_index(self, index):
        self.ind = index % self.numneurons

    def update_curate(self):
        if self.curate.get(str(self.ind)) in ['keep','seen','trash']:
            pass
        else:
            self.curate[str(self.ind)] = 'seen'

    def update_pointstate(self, label):
        d = {
            'Single':0,
            'Same Z':1,
            'All':2,
        }
        self.pointstate = d[label]
        self.update_figures()

    def update_mipstate(self, label):
        d = {
            'Single Z':0,
            'MIP':1,
        }
        self.showmip = d[label]

        self.update_ims()
        self.update_figures()

    def set_trace_icons(self):
        for ind in range(self.tmax):
            timeseries_view = pg.PlotWidget()
            timeseries_view.setBackground('w')
            timeseries_view.plot((self.timeseries[:,ind]-np.min(self.timeseries[:,ind]))/(np.max(self.timeseries[:,ind])-np.min(self.timeseries[:,ind])), pen=pg.mkPen(color=(31, 119, 180), width=5))
            icon = QIcon(timeseries_view.grab())
            item = QListWidgetItem(icon, str(ind))
            self.trace_grid.addItem(item)
    
    def update_trace_icons(self):
        for trace_icon in [self.trace_grid.item(index) for index in range(self.trace_grid.count())]:
            if self.show_settings == 0:
                trace_icon.setHidden(False)

            elif self.show_settings == 1:
                trace_icon.setHidden(self.curate.get(trace_icon.text()) in ['keep','trash'])

            elif self.show_settings == 2:
                trace_icon.setHidden(self.curate.get(trace_icon.text()) != 'keep')

            else:
                trace_icon.setHidden(self.curate.get(trace_icon.text()) != 'trash')

    def go_to_trace(self, index):
        self.set_index(index)
        self.update_ims()
        self.update_figures()
        self.update_timeseries()
        self.update_buttons()
        self.update_curate()

    def show_trace_grid_context_menu(self):
        self.trace_grid_context_menu.exec_(QCursor.pos())

    def get_imageview(self):
        plot_item = pg.PlotItem()
        image_view = pg.ImageView(view=plot_item)
        image_view.setPredefinedGradient('viridis')
        image_view.ui.histogram.hide()
        image_view.ui.roiBtn.hide()
        image_view.ui.menuBtn.hide()
        image_view.show()
        return image_view

    def update_imageview(self, image_view, im, title):
        plot_item = image_view.getView()
        for data_item in plot_item.listDataItems():
            plot_item.removeItem(data_item)
        plot_item.setTitle(title)
        image_view.getImageItem().setImage(im.T)

    def plot_on_imageview(self, image_view, x, y, color):
        plot_item = image_view.getView()
        plot_item.scatterPlot(x, y, symbolSize=10, pen=QPen(color, .1), brush=QBrush(color))