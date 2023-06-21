import atexit
import json
import napari
import os
import numpy as np
import pyqtgraph as pg

from .Extractor import *
from .multifiletiff import *
from napari.layers.points._points_constants import Mode
from napari.utils.notifications import show_info
from .Threads import *
from qtpy.QtWidgets import QAbstractItemView, QAction, QCheckBox, QSlider, QButtonGroup, QFileDialog, QGridLayout, QLabel, QListView, QListWidget, QListWidgetItem, QMenu, QPushButton, QRadioButton, QWidget, QFileDialog
from qtpy.QtCore import Qt, QPoint, QSize
from qtpy.QtGui import QBrush, QCursor, QIcon, QImage, QPen, QPixmap

def show_select_extractor_dialog():
    viewer = next(iter(napari.Viewer._instances))
    run_button = None
    for widget_name, widget in viewer.window._dock_widgets.items():
        if widget_name == 'Load Extractor (eats-worm)':
            run_button = widget
            break
    directory = str(QFileDialog.getExistingDirectory(None, "Select extractor-objects directory to load"))
    if directory.endswith('extractor-objects'):
        e = load_extractor(directory)
        run_button.setVisible(False)
        c = Curator(e=e, viewer=viewer)
    else:
        show_info("Invalid extractor directory selection. Please select an existing extractor-objects directory.")

viewer_settings = {
    1: [{'colormap': 'gray', 'visible': True}],
    4: [{'colormap': 'red', 'visible': True}, {'colormap': 'gray', 'visible': True}, {'colormap': 'green', 'visible': True}, {'colormap': 'blue', 'visible': True}],
    5: [{'colormap': 'red', 'visible': True}, {'colormap': 'gray', 'visible': True}, {'colormap': 'green', 'visible': True}, {'colormap': 'gray', 'visible': False}, {'colormap': 'blue', 'visible': True}],
}

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
    def __init__(self, mft=None, spool=None, timeseries=None, e=None, window=100, labels={}, new_curation=False, viewer=None):
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
        self.min, self.max = None, None
        curate_loaded = False
        if not new_curation:
            try:
                with open(self.path) as f:
                    self.curate = json.load(f)
                self.ind = int(self.curate['last'])
                if 'contrast_min' in self.curate and 'contrast_max' in self.curate:
                    self.min, self.max = self.curate['contrast_min'], self.curate['contrast_max']
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
        self.panel_mode = 0

        ## index for which time point to display
        self.t = 0
        
        ## maximum t
        if not self.tmax:
            self.tmax = self.tf.end_t if self.tf else 0
        
        self.threads_edited = False

        self.viewer = viewer

        self.restart()
        atexit.register(self.log_curate)

    def restart(self):
        ### enable antialiasing in pyqtgraph
        pg.setConfigOption('antialias', True)

        ### initialize napari viewer
        new_viewer = False
        if self.viewer is None:
            new_viewer = True
            self.viewer = napari.Viewer(ndisplay=3)
        if self.tf:
            for c in range(self.tf.numc):
                self.viewer.add_image(self.tf.get_t(self.t + self.t_offset, channel=c), name='channel {}'.format(c), scale=self.scale, blending='additive', **viewer_settings[self.tf.numc][c])
            if self.min is not None and self.max is not None:
                self.viewer.layers['channel 0'].contrast_limits = (self.min, self.max)
            self.min, self.max = self.viewer.layers['channel 0'].contrast_limits
            self.viewer.layers['channel 0'].events.contrast_limits.connect(lambda:self.update_mm(self.viewer.layers['channel 0'].contrast_limits))
        if self.s:
            point_size=10
            edge_width=1
            edge_width_is_relative=False
            point_symbol = 'disc'
            face_color = np.array([0,0,0,0])
            self.viewer.add_points(np.empty((0, 3)), symbol=point_symbol, face_color=face_color, edge_color='red', name='roi', size=point_size+1, scale=self.scale, edge_width=edge_width*1.25, edge_width_is_relative=edge_width_is_relative)
            self.other_rois = self.viewer.add_points(np.empty((0, 3)), symbol=point_symbol, face_color=face_color, edge_color='green', name='other rois', size=point_size, scale=self.scale, edge_width=edge_width, edge_width_is_relative=edge_width_is_relative)

            if self.curator_layers:
                for layer in self.curator_layers.keys():
                    if self.curator_layers[layer]['type'] == 'image':
                        self.viewer.add_image(self.curator_layers[layer]['data'][self.t], name=layer, scale=self.scale, blending='additive', visible=False)
                    elif self.curator_layers[layer]['type'] == 'points':
                        self.viewer.add_points(np.empty((0, 3)), name=layer, scale=self.scale, size=1, visible=False)

            self.last_selected = set()
            self.last_selected_t = None
            def handle_select(event):
                if self.other_rois.mode == 'select':
                    self.last_selected_coords = None
                    self.last_selected_t = self.t
                    selected = self.other_rois.selected_data
                    if selected != self.last_selected:
                        self.last_selected = selected
                        if selected != set():
                            for trace_icon in self.trace_grid.selectedItems():
                                trace_icon.setSelected(False)
                            thread_indices = self.napari_indices_to_spool_indices(list(selected))
                            for thread_index in thread_indices:
                                self.trace_grid.item(thread_index).setSelected(True)
                            if len(selected) == 1:
                                self.go_to_trace(thread_index)
                                self.last_selected_coords = self.other_rois.data[list(selected)[0]]

            self.other_rois.events.highlight.connect(handle_select)

            def handle_add(event):
                if self.other_rois.mode == 'add':
                    data = self.other_rois.data
                    if data.size:
                        if not (self.s.get_positions_t(self.t) == data[-1]).all(axis=1).any():
                            self.add_roi(data[-1], self.t)
            self.other_rois.events.data.connect(handle_add)

            self.drag_beginning = None
            def handle_move(event):
                if self.other_rois.mode == 'select':
                    selected = self.other_rois.selected_data
                    if len(selected) == 1 and selected == self.last_selected and (self.other_rois.data[list(selected)[0]] != self.last_selected_coords).all() and self.last_selected_t == self.t:
                        new_drag = True
                        point_index = list(selected)[0]
                        thread_index = self.napari_indices_to_spool_indices([point_index])[0]
                        if self.drag_beginning:
                            if self.drag_beginning['thread'] == thread_index and self.drag_beginning['t'] != self.t:
                                new_drag = False
                                self.alter_roi_positions(thread_index, self.drag_beginning['pos'], self.other_rois.data[point_index], self.drag_beginning['t'], self.t)
                            else:
                                print("Moved same ROI without changing timepoint or moved different ROI; discarding drag.")
                            self.drag_beginning = None
                        if new_drag:
                            print("Starting new drag for thread", thread_index, "at timepoint", self.t)
                            self.drag_beginning = {'thread': thread_index, 't': self.t, 'pos': self.other_rois.data[point_index]}
            self.other_rois.events.data.connect(handle_move)

        # initialize load buttons
        self.load_image_button = QPushButton("Load image folder")
        self.load_image_button.clicked.connect(self.load_image_folder)

        ### initialize views for images
        self.ortho_1_view = self.get_imageview()
        self.ortho_2_view = self.get_imageview()
        self.timeseries_view = pg.PlotWidget()
        self.timeseries_view_time_line = None
        self.timeseries_view_ind = None
        self.timeseries_view.setBackground('w')

        ### initialize montage view
        self.montage_view = self.get_imageview()
        self.montage_view.setVisible(False)

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
        mip_button_group = [QRadioButton('Timeseries'), QRadioButton('Montage')]
        mip_button_group[0].setChecked(True)
        mip_button_group[0].toggled.connect(lambda:self.update_panel_mode(mip_button_group[0].text()))
        mip_button_group[1].toggled.connect(lambda:self.update_panel_mode(mip_button_group[1].text()))
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
        if new_viewer:
            napari.run()
    
    ## Attempting to get autosave when instance gets deleted, not working right now TODO     
    def __del__(self):
        self.log_curate()
    
    def get_im_display(self, im):
        return np.clip(im, self.min, self.max) / (self.max - self.min)
    
    def add_figures_to_image_grid(self):
        if self.tf:
            self.load_image_button.setVisible(False)
        else:
            self.image_grid.addWidget(self.load_image_button, 1, 0)
        self.image_grid.addWidget(self.timeseries_view, 0, 0, 2, 1)
        self.image_grid.addWidget(self.ortho_1_view, 0, 1)
        self.image_grid.addWidget(self.ortho_2_view, 1, 1)
        self.image_grid.addWidget(self.trace_grid, 0, 2, 2, 1)
        self.image_grid.setColumnStretch(0, 2)
        self.image_grid.setColumnStretch(1, 1)
        self.image_grid.setColumnStretch(2, 1)
    
    def update_figures(self):
        if self.tf:
            for c in range(self.tf.numc):
                self.viewer.layers['channel {}'.format(c)].data = self.tf.get_t(self.t + self.t_offset, channel=c)
            if self.panel_mode == 0:
                self.update_imageview(self.ortho_1_view, self.get_im_display(np.max(self.tf.get_t(self.t + self.t_offset), axis=1)), "Ortho MIP ax 1")
                self.update_imageview(self.ortho_2_view, self.get_im_display(np.max(self.tf.get_t(self.t + self.t_offset), axis=2)), "Ortho MIP ax 2")
            else:
                self.update_imageview(self.montage_view, self.get_im_display(np.rot90(np.vstack(self.tf.get_t(self.t + self.t_offset)))), "Montage View")
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
                    elif self.curator_layers[layer]['type'] == 'points':
                        self.viewer.layers[layer].data = self.curator_layers[layer]['data'].item()[self.ind].get(self.t, np.empty((0, 3)))

            if self.zoom_to_roi:
                roi_pos = self.s.threads[self.ind].get_position_t(self.t)
                if self.viewer.dims.ndisplay == 2:
                    z_index = self.viewer.dims.order.index(0)
                    self.viewer.dims.current_step = self.viewer.dims.current_step[:z_index] + (round(roi_pos[0]),) + self.viewer.dims.current_step[z_index+1:]
                else:
                    z_lower_bound, z_upper_bound, x_lower_bound, x_upper_bound, y_lower_bound, y_upper_bound = self.get_zoom_bounds(roi_pos)
                    zoom_mask = np.zeros(self.viewer.layers['channel 0'].data.shape, dtype=bool)
                    zoom_mask[z_lower_bound:z_upper_bound, x_lower_bound:x_upper_bound, y_lower_bound:y_upper_bound] = True
                    self.viewer.layers['channel 0'].data *= zoom_mask
                    # centering produces buggy behavior in 2d; for now, less broken to only center in 3d
                    self.viewer.camera.center = roi_pos
                self.viewer.camera.zoom = 20

        if self.s and self.panel_mode == 1:
            # plotting for multiple points
            if self.pointstate==0:
                pass
            elif self.pointstate==1:
                self.plot_on_montageview(self.s.get_positions_t_z(self.t, self.s.threads[self.ind].get_position_t(self.t)[0]), Qt.blue)
            elif self.pointstate==2:
                self.plot_on_montageview(self.s.get_positions_t(self.t), Qt.blue)
            self.plot_on_montageview(np.array([self.s.threads[self.ind].get_position_t(self.t)]), Qt.red)

    def update_timeseries(self):
        if self.timeseries is not None:
            if not self.timeseries_view_ind or self.timeseries_view_ind != self.ind:
                self.timeseries_view.clear()
                self.timeseries_view.plot((self.timeseries[:,self.ind]-np.min(self.timeseries[:,self.ind]))/(np.max(self.timeseries[:,self.ind])-np.min(self.timeseries[:,self.ind])), pen=pg.mkPen(color=(31, 119, 180), width=3))
                self.timeseries_view_ind = self.ind
                self.timeseries_view_time_line = self.timeseries_view.addLine(x=self.t, pen='r')
            else:
                self.timeseries_view.removeItem(self.timeseries_view_time_line)
                self.timeseries_view_time_line = self.timeseries_view.addLine(x=self.t, pen='r')
            self.timeseries_view.setTitle('Series=' + str(self.ind) + ', Z=' + str(int(self.s.threads[self.ind].get_position_t(self.t)[0])) + ", T=" + str(self.t), color='#000')

    def update_t(self, val):
        # Update index for t
        self.t = val
        # update images for t
        self.update_figures()
        self.update_timeseries()

    def update_mm(self, contrast_limits):
        self.min = contrast_limits[0]
        self.max = contrast_limits[1]
        self.update_figures()

    def next(self):
        self.set_index_next()
        self.update()

    def prev(self):
        self.set_index_prev()
        self.update()
    
    def update(self):
        self.update_figures()
        self.update_timeseries()
        self.update_buttons()
        self.update_curate()
        self.update_trace_icons()

    def log_curate(self):
        print('Logging curation...')
        self.do_hacks()
        self.curate['last'] = self.ind
        self.curate["contrast_min"], self.curate["contrast_max"] = self.min, self.max
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
        if self.ind in selected_trace_indices and self.show_settings != 0:
            self.next()
        else:
            self.update()


    def trash_all_selected(self, label):
        selected_trace_indices = [self.trace_grid.indexFromItem(icon).row() for icon in self.trace_grid.selectedItems()]
        for index in selected_trace_indices:
            self.curate[str(index)]='trash'
        if self.show_settings != 0:
            self.viewer.layers['other rois'].selected_data = {}
        if self.ind in selected_trace_indices and self.show_settings != 0:
            self.next()
        else:
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
        self.other_rois.selected_data = {}
        if self.show_settings != 0:
            if str(self.ind) in self.curate:
                current_label = self.curate[str(self.ind)]
                if (current_label != 'seen' and self.show_settings == 1) or (current_label != 'keep' and self.show_settings == 2) or (current_label != 'trash' and self.show_settings == 3):
                    self.next()
            elif self.show_settings != 1:
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
        self.other_rois.selected_data = {}
        self.update()

    def update_zoomstate(self, checkbox):
        self.zoom_to_roi = checkbox.isChecked()
        self.update()

    def update_panel_mode(self, label):
        last_setting = self.panel_mode
        d = {
            'Timeseries':0,
            'Montage':1,
        }
        self.panel_mode = d[label]

        ## clear image grid and reassign
        if 1 == self.panel_mode:
            i = self.image_grid.count() - 2
            while(i >= 0):
                grid_item = self.image_grid.itemAt(i).widget()
                grid_item.setParent(None)
                i -=1
            self.image_grid.addWidget(self.montage_view, 0, 0, 2, 2)
            self.montage_view.setVisible(True)

        elif 1 == last_setting:
            self.montage_view.setVisible(False)
            self.montage_view.setParent(None)
            self.add_figures_to_image_grid()

        self.update_figures()

    def set_trace_icons(self, indices=None):
        if indices is None:
            indices = range(self.timeseries.shape[1])
        if self.timeseries is not None:
            for index in indices:
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
            if str(thread) in self.curate.keys():
                if self.curate[str(thread)] == 'trash':
                    excluded_threads.append(thread)
        roi_added = self.s.add_thread_post_hoc(position, t, self.scale, excluded_threads=excluded_threads)
        if not roi_added:
            self.other_rois.data = np.delete(self.other_rois.data, -1, axis=0)
        else:
            self.num_neurons += 1
            print('Saving blob timeseries as numpy object...')
            self.e.timeseries = np.hstack((self.e.timeseries, np.empty((self.e.timeseries.shape[0], 1))))
            self.e.timeseries[:,-1] = np.NaN
            self.e.spool.export(f=os.path.join(self.e.output_dir, 'threads.obj'))
            self.e.save_timeseries()
            self.e.save_dataframe()
            self.other_rois.selected_data = {}
            self.timeseries = self.e.timeseries
            self.set_trace_icons([self.timeseries.shape[1] - 1])
            self.update_figures()
            self.update_timeseries()

    def alter_roi_positions(self, thread, position_0, position_1, time_0, time_1):
        self.s.alter_thread_post_hoc(thread, position_0, position_1, time_0, time_1)
        self.threads_edited = True
        self.update_figures()

    def napari_indices_to_spool_indices(self, indices):
        spool_indices = []
        thread_index = -1
        visible_threads = 0
        for napari_point_index in sorted(indices):
            while visible_threads <= napari_point_index:
                thread_index += 1
                if not self.trace_grid.item(thread_index).isHidden():
                    visible_threads += 1
            spool_indices.append(thread_index)
        return spool_indices

    # handle any rois added manually
    def do_hacks(self):
        if self.threads_edited:
            self.s.export(f=os.path.join(self.e.output_dir, 'threads.obj'))
            self.e.save_dataframe()
