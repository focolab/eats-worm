import atexit
import json
import napari
import os
import numpy as np
import pyqtgraph as pg

from .Extractor import *
from .multifiletiff import *
from napari.layers.points._points_constants import Mode
from .Threads import *
from qtpy.QtWidgets import QAbstractItemView, QAction, QCheckBox, QSlider, QButtonGroup, QFileDialog, QGridLayout, QLabel, QListView, QListWidget, QListWidgetItem, QMenu, QPushButton, QRadioButton, QWidget
from qtpy.QtCore import Qt, QPoint, QSize
from qtpy.QtGui import QBrush, QCursor, QIcon, QImage, QPen, QPixmap

viewer_settings = {
    1: [{'colormap': 'gray', 'visible': True}],
    4: [{'colormap': 'red', 'visible': True}, {'colormap': 'gray', 'visible': True}, {'colormap': 'green', 'visible': True}, {'colormap': 'blue', 'visible': True}],
    5: [{'colormap': 'red', 'visible': True}, {'colormap': 'gray', 'visible': True}, {'colormap': 'green', 'visible': True}, {'colormap': 'gray', 'visible': False}, {'colormap': 'blue', 'visible': True}],
}

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
    z = round(z)
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
    def __init__(self, mft=None, spool=None, timeseries=None, e=None, window=100, labels={}, new_curation=False):
        if e:
            self.s = e.spool
            self.timeseries = e.timeseries
            self.tf = e.im
            self.tmax = e.end_t - e.start_t
            self.t_offset = e.start_t
            self.scale = e.anisotropy
            try:
                self.curator_layers = e.curator_layers
                for layer in self.curator_layers:
                    self.curator_layers[layer]['data'] = np.array(self.curator_layers[layer]['data'])
            except:
                self.curator_layers = None
        else:
            self.tf = mft
            self.s = spool
            self.timeseries = timeseries
            self.tmax = None
            self.t_offset = 0
            self.scale = (15, 1, 1)
            if mft:
                self.scale = mft.anisotropy
        if self.tf:
            self.tf.t = 0
        self.e = e
        self.window = window
        ## num neurons
        self.num_neurons = len(self.s.threads) if self.s else 0
        self.num_frames = len(self.tf.frames) if self.tf else 0

        self.path = os.path.join(self.tf.output_dir, 'curate.json') if self.tf else None
        self.ind = 0
        curate_loaded = False
        if not new_curation:
            try:
                with open(self.path) as f:
                    self.curate = json.load(f)
                self.ind = int(self.curate['last'])
                curate_loaded = True
            except:
                print("No curate.json in output folder. Creating new curation.")
        if not curate_loaded:
            self.curate = {}
            self.ind = 0
            self.curate['0'] = 'seen'
            self.curate['labels'] = {}
            for i in range(len(self.s.threads)):
                if self.s.threads[i].label:
                    self.curate['labels'][str(i)] = self.s.threads[i].label
        if labels != {}:
            self.curate['labels'] = labels


        # array to contain internal state: whether to display single ROI, ROI in Z, or all ROIs
        self.pointstate = 2
        self.zoom_to_roi = False
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
        if not self.tmax:
            self.tmax = (self.tf.numframes-self.tf.offset)//self.tf.numz if self.tf else 0
        
        self.neurs_to_add = 0

        self.restart()
        atexit.register(self.log_curate)

    def restart(self):
        ### enable antialiasing in pyqtgraph
        pg.setConfigOption('antialias', True)

        ### initialize napari viewer
        self.viewer = napari.Viewer(ndisplay=3)
        if self.tf:
            for c in range(self.tf.numc):
                self.viewer.add_image(self.tf.get_t(self.t + self.t_offset, channel=c), name='channel {}'.format(c), scale=self.scale, blending='additive', **viewer_settings[self.tf.numc][c])
        if self.s:
            self.viewer.add_points(np.empty((0, 3)), symbol='ring', face_color='red', edge_color='red', name='roi', size=2, scale=self.scale)

            self.other_rois = self.viewer.add_points(np.empty((0, 3)), symbol='ring', face_color='blue', edge_color='blue', name='other rois', size=1, scale=self.scale)

            if self.curator_layers:
                for layer in self.curator_layers.keys():
                    if self.curator_layers[layer]['type'] == 'image':
                        self.viewer.add_image(self.curator_layers[layer]['data'][self.t], name=layer, scale=self.scale, blending='additive', visible=False)

            self.last_selected = set()
            def handle_select(event):
                if self.other_rois.mode == 'select':
                    selected = self.other_rois.selected_data
                    if selected != self.last_selected:
                        self.last_selected = selected
                        if selected != set():
                            for trace_icon in self.trace_grid.selectedItems():
                                trace_icon.setSelected(False)
                            # napari indices may not match thread indices depending on display mode
                            thread_index = -1
                            visible_threads = 0
                            for napari_point_index in sorted(selected):
                                while visible_threads <= napari_point_index:
                                    thread_index += 1
                                    if not self.trace_grid.item(thread_index).isHidden():
                                        visible_threads += 1
                                self.trace_grid.item(thread_index).setSelected(True)
                            if len(selected) == 1:
                                self.go_to_trace(thread_index)

            self.other_rois.events.highlight.connect(handle_select)

            def handle_add(event):
                if self.other_rois.mode == 'add':
                    data = self.other_rois.data
                    if data.size:
                        if not (self.s.get_positions_t(self.t) == data[-1]).all(axis=1).any():
                            self.add_roi(data[-1], self.t)
            self.other_rois.events.data.connect(handle_add)

        # initialize load buttons
        self.load_image_button = QPushButton("Load image folder")
        self.load_image_button.clicked.connect(self.load_image_folder)

        ### initialize views for images
        self.z_view = self.get_imageview()
        self.z_plus_one_view = self.get_imageview()
        self.z_plus_one_view.view.setXLink(self.z_view.view)
        self.z_plus_one_view.view.setYLink(self.z_view.view)
        self.z_minus_one_view = self.get_imageview()
        self.z_minus_one_view.view.setXLink(self.z_view.view)
        self.z_minus_one_view.view.setYLink(self.z_view.view)
        self.ortho_1_view = self.get_imageview()
        self.ortho_2_view = self.get_imageview()
        self.timeseries_view = pg.PlotWidget()
        self.timeseries_view.setBackground('w')
        if self.timeseries is not None:
            self.timeseries_view.plot((self.timeseries[:,self.ind]-np.min(self.timeseries[:,self.ind]))/(np.max(self.timeseries[:,self.ind])-np.min(self.timeseries[:,self.ind])), pen='b')
            self.timeseries_view.addLine(x=self.t, pen='r')
        
        ### initialize montage view
        self.montage_view = self.get_imageview()
        self.montage_view.setVisible(False)

        ### Series label
        self.series_label = QLabel()
        self.viewer.window.add_dock_widget(self.series_label, area='right')

        ### Grid showing all extracted timeseries
        self.trace_grid = QListWidget()
        self.trace_grid.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.trace_grid.setViewMode(QListWidget.IconMode)
        self.trace_grid.setResizeMode(QListView.Adjust)
        self.trace_grid.setDragDropMode(QAbstractItemView.NoDragDrop)
        self.trace_grid.setIconSize(QSize(288, 96))
        self.trace_grid.itemDoubleClicked.connect(lambda:self.go_to_trace(self.trace_grid.currentRow()))
        self.trace_grid.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.trace_grid.itemChanged.connect(self.update_label)
        self.trace_grid.setContextMenuPolicy(Qt.CustomContextMenu)
        self.trace_grid_context_menu = QMenu(self.trace_grid)
        keep_all_action = QAction('Keep selected', self.trace_grid, triggered = self.keep_all_selected)
        self.trace_grid_context_menu.addAction(keep_all_action)
        trash_all_action = QAction('Trash selected', self.trace_grid, triggered = self.trash_all_selected)
        self.trace_grid_context_menu.addAction(trash_all_action)
        add_label_action = QAction('Add label', self.trace_grid, triggered = self.label_selected)
        self.trace_grid_context_menu.addAction(add_label_action)
        self.trace_grid.customContextMenuRequested[QPoint].connect(self.show_trace_grid_context_menu)
        self.set_trace_icons()

        ### figure grid
        image_grid_container = QWidget()
        image_grid = QGridLayout(image_grid_container)
        self.image_grid = image_grid
        self.add_figures_to_image_grid()
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
        points_button_group[2].setChecked(True)
        points_button_group[0].toggled.connect(lambda:self.update_pointstate(points_button_group[0].text()))
        points_button_group[1].toggled.connect(lambda:self.update_pointstate(points_button_group[1].text()))
        points_button_group[2].toggled.connect(lambda:self.update_pointstate(points_button_group[2].text()))
        self.viewer.window.add_dock_widget(points_button_group, area='right')

        #### Axis for button for display
        zoom_checkbox = QCheckBox("Zoom to ROI?")
        zoom_checkbox.setChecked(False)
        zoom_checkbox.stateChanged.connect(lambda:self.update_zoomstate(zoom_checkbox))
        self.viewer.window.add_dock_widget(zoom_checkbox, area='right')

        #### Axis for whether to display MIP on left
        mip_button_group = [QRadioButton('Single Z'), QRadioButton('MIP'), QRadioButton('Montage')]
        mip_button_group[0].setChecked(True)
        mip_button_group[0].toggled.connect(lambda:self.update_mipstate(mip_button_group[0].text()))
        mip_button_group[1].toggled.connect(lambda:self.update_mipstate(mip_button_group[1].text()))
        mip_button_group[2].toggled.connect(lambda:self.update_mipstate(mip_button_group[2].text()))
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

        ### Update buttons in case of previous curation
        self.update_buttons()
        napari.run()
    
    ## Attempting to get autosave when instance gets deleted, not working right now TODO     
    def __del__(self):
        self.log_curate()

    def update_ims(self):
        if not self.tf or not self.s:
            self.im = np.ones((1, 1))
            self.im_plus_one = np.ones((1, 1))
            self.im_minus_one = np.ones((1, 1))
    
        else:
            f = int(self.s.threads[self.ind].get_position_t(self.t)[0])
            if 1 == self.showmip:
                self.im = np.max(self.tf.get_t(self.t + self.t_offset),axis = 0)
            else:
                self.im = self.tf.get_tbyf(self.t + self.t_offset, f)
            if f == self.num_frames - 1:
                self.im_plus_one = np.zeros(self.im.shape)
            else:
                self.im_plus_one = self.tf.get_tbyf(self.t + self.t_offset, f + 1)
            if f == 0:
                self.im_minus_one = np.zeros(self.im.shape)
            else:
                self.im_minus_one = self.tf.get_tbyf(self.t + self.t_offset, f - 1)
    
    def get_im_display(self, im):
        return (im - self.min)/(self.max - self.min)
    
    def add_figures_to_image_grid(self):
        self.image_grid.addWidget(self.z_plus_one_view, 0, 0)
        if self.tf:
            self.image_grid.addWidget(self.z_view, 1, 0)
            self.load_image_button.setVisible(False)
        else:
            self.image_grid.addWidget(self.load_image_button, 1, 0)
            self.z_view.setVisible(False)
        self.image_grid.addWidget(self.z_minus_one_view, 2, 0)
        self.image_grid.addWidget(self.timeseries_view, 0, 1)
        self.image_grid.addWidget(self.ortho_1_view, 1, 1)
        self.image_grid.addWidget(self.ortho_2_view, 2, 1)
        self.image_grid.addWidget(self.trace_grid, 0, 2, 3, 1)
        self.image_grid.setColumnStretch(0, 2)
        self.image_grid.setColumnStretch(1, 1)
        self.image_grid.setColumnStretch(2, 1)
    
    def update_figures(self):
        if self.tf:
            for c in range(self.tf.numc):
                self.viewer.layers['channel {}'.format(c)].data = self.tf.get_t(self.t + self.t_offset, channel=c)
            self.update_imageview(self.ortho_1_view, np.max(self.tf.get_t(self.t + self.t_offset), axis=1), "Ortho MIP ax 1")
            self.update_imageview(self.ortho_2_view, np.max(self.tf.get_t(self.t + self.t_offset), axis=2), "Ortho MIP ax 2")
            self.update_imageview(self.montage_view, np.rot90(np.vstack(self.tf.get_t(self.t + self.t_offset))), "Montage View")
        if self.s:
            # swap to PAN_ZOOM to avoid highlighting points on recreation
            last_roi_mode = self.viewer.layers['roi'].mode
            last_other_rois_mode = self.viewer.layers['other rois'].mode
            last_other_rois_selected_data = self.viewer.layers['other rois'].selected_data
            self.viewer.layers['roi'].mode = Mode.PAN_ZOOM
            self.viewer.layers['other rois'].mode = Mode.PAN_ZOOM

            self.viewer.layers['roi'].data = np.array([self.s.threads[self.ind].get_position_t(self.t)])
            if self.pointstate==0:
                self.viewer.layers['other rois'].data = np.empty((0, 3))
            elif self.pointstate==1:
                self.viewer.layers['other rois'].data = self.s.get_positions_t_z(self.t, self.s.threads[self.ind].get_position_t(self.t)[0])
            elif self.pointstate==2:
                self.viewer.layers['other rois'].data = self.s.get_positions_t(self.t)
                if self.show_settings != 0:
                    other_rois = []
                    for i in range(self.viewer.layers['other rois'].data.shape[0]):
                        roi = self.viewer.layers['other rois'].data[i]
                        if self.curate.get(str(i))=='keep':
                            if self.show_settings == 2:
                                other_rois.append(roi)
                        elif self.curate.get(str(i))=='trash':
                            if self.show_settings == 3:
                                other_rois.append(roi)
                        elif self.show_settings ==1:
                            other_rois.append(roi)
                    self.viewer.layers['other rois'].data = np.array(other_rois)

            self.viewer.layers['roi'].selected_data = {}
            self.viewer.layers['other rois'].selected_data = last_other_rois_selected_data
            self.viewer.layers['roi'].mode = last_roi_mode
            self.viewer.layers['other rois'].mode = last_other_rois_mode

            if self.curator_layers:
                for layer in self.curator_layers:
                    if self.curator_layers[layer]['type'] == 'image':
                        self.viewer.layers[layer].data = self.curator_layers[layer]['data'][self.t]

            if self.zoom_to_roi:
                roi_pos = self.s.threads[self.ind].get_position_t(self.t)
                z_lower_bound, z_upper_bound, x_lower_bound, x_upper_bound, y_lower_bound, y_upper_bound = self.get_zoom_bounds(roi_pos)
                zoom_mask = np.zeros(self.viewer.layers['channel 0'].data.shape, dtype=bool)
                zoom_mask[z_lower_bound:z_upper_bound, x_lower_bound:x_upper_bound, y_lower_bound:y_upper_bound] = True
                self.viewer.layers['channel 0'].data *= zoom_mask
                self.viewer.camera.center = roi_pos
                self.viewer.camera.zoom = 20

        self.update_imageview(self.z_view, self.get_im_display(self.im), "Parent Z")
        self.update_imageview(self.z_plus_one_view, self.get_im_display(self.im_plus_one), "Z + 1")
        self.update_imageview(self.z_minus_one_view, self.get_im_display(self.im_minus_one), "Z - 1")

        if self.s:
            # plotting for multiple points
            if self.pointstate==0:
                pass
            elif self.pointstate==1:
                self.plot_on_imageview(self.z_view, self.s.get_positions_t_z(self.t, self.s.threads[self.ind].get_position_t(self.t)[0])[:,2], self.s.get_positions_t_z(self.t,self.s.threads[self.ind].get_position_t(self.t)[0])[:,1], Qt.blue)
                self.plot_on_montageview(self.s.get_positions_t_z(self.t, self.s.threads[self.ind].get_position_t(self.t)[0]), Qt.blue)
            elif self.pointstate==2:
                self.plot_on_imageview(self.z_view, self.s.get_positions_t(self.t)[:,2], self.s.get_positions_t(self.t)[:,1], Qt.blue)
                self.plot_on_imageview(self.z_plus_one_view, self.s.get_positions_t(self.t)[:,2], self.s.get_positions_t(self.t)[:,1], Qt.blue)
                self.plot_on_imageview(self.z_minus_one_view, self.s.get_positions_t(self.t)[:,2], self.s.get_positions_t(self.t)[:,1], Qt.blue)
                self.plot_on_montageview(self.s.get_positions_t(self.t), Qt.blue)
            self.plot_on_imageview(self.z_view, [self.s.threads[self.ind].get_position_t(self.t)[2]], [self.s.threads[self.ind].get_position_t(self.t)[1]], Qt.red)
            self.plot_on_montageview(np.array([self.s.threads[self.ind].get_position_t(self.t)]), Qt.red)

        if self.timeseries is not None:
            self.series_label.setText('Series=' + str(self.ind) + ', Z=' + str(round(self.s.threads[self.ind].get_position_t(self.t)[0])))

    def update_timeseries(self):
        self.timeseries_view.clear()
        if self.timeseries is not None:
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
        self.update()

    def prev(self):
        self.set_index_prev()
        self.update()
    
    def update(self):
        self.update_ims()
        self.update_figures()
        self.update_timeseries()
        self.update_buttons()
        self.update_curate()
        self.update_trace_icons()

    def log_curate(self):
        self.do_hacks()
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
        selected_trace_indices = [self.trace_grid.indexFromItem(icon).row() for icon in self.trace_grid.selectedItems()]
        for index in selected_trace_indices:
            self.curate[str(index)]='keep'
        if self.show_settings != 0:
            self.viewer.layers['other rois'].selected_data = {}
        self.update()


    def trash_all_selected(self, label):
        selected_trace_indices = [self.trace_grid.indexFromItem(icon).row() for icon in self.trace_grid.selectedItems()]
        for index in selected_trace_indices:
            self.curate[str(index)]='trash'
        if self.show_settings != 0:
            self.viewer.layers['other rois'].selected_data = {}
        self.update()

    def label_selected(self, label):
        selected_icons = self.trace_grid.selectedItems()
        if len(selected_icons) == 1:
            self.trace_grid.editItem(selected_icons[0])

    def update_label(self, labeled_item):
        index = self.trace_grid.indexFromItem(labeled_item).row()
        if 'labels' not in self.curate:
            self.curate['labels'] = {}
        self.curate['labels'][str(index)] = labeled_item.text()

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
        if self.show_settings != 0:
            if str(self.ind) in self.curate:
                current_label = self.curate[str(self.ind)]
                if (current_label != 'seen' and self.show_settings == 1) or (current_label != 'keep' and self.show_settings == 2) or (current_label != 'trash' and self.show_settings == 3):
                    self.next()
        self.update_trace_icons()
        self.update_figures()

    def set_index_prev(self):
        if self.show_settings == 0:
            self.ind -= 1
            self.ind = self.ind % self.num_neurons

        elif self.show_settings == 1:
            self.ind -= 1
            counter = 0
            while self.curate.get(str(self.ind)) in ['keep','trash'] and counter != self.num_neurons:
                self.ind -= 1
                self.ind = self.ind % self.num_neurons
                counter += 1
            self.ind = self.ind % self.num_neurons
        elif self.show_settings == 2:
            self.ind -= 1
            counter = 0
            while self.curate.get(str(self.ind)) not in ['keep'] and counter != self.num_neurons:
                self.ind -= 1
                counter += 1
            self.ind = self.ind % self.num_neurons
        else:
            self.ind -= 1
            counter = 0
            while self.curate.get(str(self.ind)) not in ['trash'] and counter != self.num_neurons:
                self.ind -= 1
                counter += 1
            self.ind = self.ind % self.num_neurons

    def set_index_next(self):
        if self.show_settings == 0:
            self.ind += 1
            self.ind = self.ind % self.num_neurons

        elif self.show_settings == 1:
            self.ind += 1
            counter = 0
            while self.curate.get(str(self.ind)) in ['keep','trash'] and counter != self.num_neurons:
                self.ind += 1
                self.ind = self.ind % self.num_neurons
                counter += 1
            self.ind = self.ind % self.num_neurons
        elif self.show_settings == 2:
            self.ind += 1
            counter = 0
            while self.curate.get(str(self.ind)) not in ['keep'] and counter != self.num_neurons:
                self.ind += 1
                counter += 1
            self.ind = self.ind % self.num_neurons
        else:
            self.ind += 1
            counter = 0
            while self.curate.get(str(self.ind)) not in ['trash'] and counter != self.num_neurons:
                self.ind += 1
                counter += 1
            self.ind = self.ind % self.num_neurons

    def set_index(self, index):
        self.ind = index % self.num_neurons

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
        self.update()

    def update_zoomstate(self, checkbox):
        self.zoom_to_roi = checkbox.isChecked()
        self.update()

    def update_mipstate(self, label):
        last_setting = self.showmip
        d = {
            'Single Z':0,
            'MIP':1,
            'Montage':2,
        }
        self.showmip = d[label]

        ## clear image grid and reassign
        if 2 == self.showmip:
            i = self.image_grid.count() - 2
            while(i >= 0):
                grid_item = self.image_grid.itemAt(i).widget()
                grid_item.setParent(None)
                i -=1
            self.image_grid.addWidget(self.montage_view, 0, 0, 3, 2)
            self.montage_view.setVisible(True)

        elif 2 == last_setting:
            self.montage_view.setVisible(False)
            self.montage_view.setParent(None)
            self.add_figures_to_image_grid()

        self.update_ims()
        self.update_figures()

    def set_trace_icons(self):
        if self.timeseries is not None:
            for index in range(self.timeseries.shape[1]):
                self.plot_timeseries_to_trace_grid(index)

    def plot_timeseries_to_trace_grid(self, index):
        timeseries_view = pg.PlotWidget()
        timeseries_view.setBackground('w')
        timeseries_view.plot((self.timeseries[:,index]-np.min(self.timeseries[:,index]))/(np.max(self.timeseries[:,index])-np.min(self.timeseries[:,index])), pen=pg.mkPen(color=(31, 119, 180), width=5))
        icon = QIcon(timeseries_view.grab())
        label = self.curate.get('labels', {}).get(str(index), str(index))
        item = QListWidgetItem(icon, label)
        item.setFlags(item.flags() | Qt.ItemIsEditable)
        self.trace_grid.addItem(item)
    
    def update_trace_icons(self):
        for index in range(self.trace_grid.count()):
            trace_icon = self.trace_grid.item(index)

            if self.show_settings == 0:
                trace_icon.setHidden(False)

            elif self.show_settings == 1:
                trace_icon.setHidden(self.curate.get(str(index)) in ['keep','trash'])

            elif self.show_settings == 2:
                trace_icon.setHidden(self.curate.get(str(index)) != 'keep')

            else:
                trace_icon.setHidden(self.curate.get(str(index)) != 'trash')

    def go_to_trace(self, index):
        self.set_index(index)
        self.update()

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
        plot_item.scatterPlot(x, y, symbolSize=3, pen=QPen(color, .1), brush=QBrush(color))
    
    def plot_on_montageview(self, positions, color):
        x_size, y_size = self.tf.get_t(self.t + self.t_offset).shape[-2:]
        z = positions[:,0]
        x = positions[:,1]
        y = positions[:,2]
        self.plot_on_imageview(self.montage_view, z * x_size + x, -y + y_size, color)
        for z in range(1, self.num_frames):
            self.montage_view.getView().plot([z * x_size] * y_size, range(y_size), pen='y')

    def get_zoom_bounds(self, position):
        window_radius = 15
        z_lower_bound = max(0, round(position[0] - (window_radius // self.scale[0])))
        z_upper_bound = min(self.viewer.layers['channel 0'].data.shape[0], round(position[0] + (window_radius // self.scale[0])))
        x_lower_bound = max(0, round(position[1] - window_radius))
        x_upper_bound = min(self.viewer.layers['channel 0'].data.shape[1], round(position[1] + window_radius))
        y_lower_bound = max(0, round(position[2] - window_radius))
        y_upper_bound = min(self.viewer.layers['channel 0'].data.shape[2], round(position[2] + window_radius))
        return z_lower_bound, z_upper_bound, x_lower_bound, x_upper_bound, y_lower_bound, y_upper_bound

    def load_image_folder(self):
        folder_path = QFileDialog.getExistingDirectory()
        mft = MultiFileTiff(folder_path)
        self.viewer.window.close()
        Curator(mft=mft, spool = self.s, timeseries = self.timeseries, window = self.window)

    def add_roi(self, position, t):
        excluded_threads = []
        for thread in range(len(self.s.threads)):
            if self.curate[str(thread)] == 'trash':
                excluded_threads.append(thread)
        roi_added = self.s.add_thread_post_hoc(position, t, self.scale, excluded_threads=excluded_threads)
        if not roi_added:
            self.other_rois.data = np.delete(self.other_rois.data, -1, axis=0)
        else:
            print('Saving blob timeseries as numpy object...')
            self.e.spool.export(f=os.path.join(self.e.output_dir, 'threads.obj'))

            self.neurs_to_add += 1

    # handle any rois added manually
    def do_hacks(self):
        if self.neurs_to_add:
            self.e.quantify()
            self.timeseries = self.e.timeseries
            self.e.save_timeseries()

            self.num_neurons += self.neurs_to_add
