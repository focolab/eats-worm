from .Extractor import *
from .Threads import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider, CheckButtons, TextBox, RadioButtons
from .multifiletiff import *
import json
import atexit

from magicgui import magicgui
from magicgui._qt.widgets import QDoubleSlider
from qtpy.QtWidgets import QSlider, QButtonGroup, QLabel, QListWidget, QListWidgetItem, QPushButton, QRadioButton
from qtpy.QtCore import Qt, QSize
from qtpy.QtGui import QPixmap, QImage, QIcon
import napari
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure

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
        self.update_im()

        ## Display range 
        self.min = np.min(self.im)
        self.max = np.max(self.im) # just some arbitrary value
        
        ## maximum t
        self.tmax = e.t

        self.restart()
        atexit.register(self.log_curate)

    def restart(self):
        with napari.gui_qt():
            ## Figures to display
            self.static_canvas_1 = FigureCanvas(Figure())
            self.ax1 = self.static_canvas_1.figure.subplots()
            self.static_canvas_2 = FigureCanvas(Figure())
            self.ax2 = self.static_canvas_2.figure.subplots()
            self.static_canvas_3 = FigureCanvas(Figure())
            self.timeax = self.static_canvas_3.figure.subplots()


            ### First subplot: whole image with red dot over ROI
            self.img1 = self.ax1.imshow(self.get_im_display(),cmap='gray',vmin = 0, vmax = 1)
            
            # plotting for multiple points
            if self.pointstate==0:
                pass
            elif self.pointstate==1:
                self.point1 = self.ax1.scatter(self.s.get_positions_t_z(self.t, self.s.threads[self.ind].get_position_t(self.t)[0])[:,2], self.s.get_positions_t_z(self.t,self.s.threads[self.ind].get_position_t(self.t)[0])[:,1],c='b', s=10)
            elif self.pointstate==2:
                self.point1 = self.ax1.scatter(self.s.get_positions_t(self.t)[:,2], self.s.get_positions_t(self.t)[:,1],c='b', s=10)
            self.thispoint = self.ax1.scatter(self.s.threads[self.ind].get_position_t(self.t)[2], self.s.threads[self.ind].get_position_t(self.t)[1],c='r', s=10)

            ### Second subplot: some window around the ROI
            self.subim,self.offset = subaxis(self.im, self.s.threads[self.ind].get_position_t(self.t), self.window)

            self.img2 = self.ax2.imshow(self.get_subim_display(),cmap='gray',vmin = 0, vmax =1)
            self.point2 = self.ax2.scatter(self.window/2+self.offset[0], self.window/2+self.offset[1],c='r', s=40)

            ### Third subplot: plotting the timeseries
            self.timeplot, = self.timeax.plot((self.timeseries[:,self.ind]-np.min(self.timeseries[:,self.ind]))/(np.max(self.timeseries[:,self.ind])-np.min(self.timeseries[:,self.ind])))
        
            ### new
            self.viewer = napari.Viewer(ndisplay=3)
            scale = [5, 1, 1]
            self.viewer.add_image(self.tf.get_t(self.t), name='volume', scale=scale)
            self.viewer.add_points([self.s.threads[self.ind].get_position_t(self.t)], face_color='red', name='roi', size=1, scale=scale)
            self.viewer.window.add_dock_widget(self.static_canvas_1, area='bottom', name='img1')
            self.viewer.window.add_dock_widget(self.static_canvas_2, area='bottom', name='img2')
            self.viewer.window.add_dock_widget(self.static_canvas_3, area='bottom', name='timeplot')

            ### Series label
            self.series_label = QLabel('Series=' + str(self.ind) + ', Z=' + str(int(self.s.threads[self.ind].get_position_t(self.t)[0])))
            self.viewer.window.add_dock_widget(self.series_label, area='right')

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
            self.keep_button_group.buttonClicked.connect(lambda:self.keep(self.keep_button_group.checkedButton().text()))
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
            #where the buttons are, and their locations
            bprev = QPushButton('Previous')
            bprev.clicked.connect(lambda:self.prev())
            bnext = QPushButton('Next')
            bnext.clicked.connect(lambda:self.next())
            self.viewer.window.add_dock_widget([bprev, bnext], area='right')

            self.trace_list = QListWidget()
            self.trace_list.setViewMode(QListWidget.IconMode)
            self.trace_list.setIconSize(QSize(96, 96))
            self.trace_list.itemDoubleClicked.connect(lambda:self.go_to_trace(int(self.trace_list.currentItem().text())))
            self.update_trace_icons()
            self.viewer.window.add_dock_widget(self.trace_list)
    
    ## Attempting to get autosave when instance gets deleted, not working right now TODO     
    def __del__(self):
        self.log_curate()

    def update_im(self):
        if self.showmip:
            self.im = np.max(self.tf.get_t(self.t),axis = 0)
        else:
            self.im = self.tf.get_tbyf(self.t,int(self.s.threads[self.ind].get_position_t(self.t)[0]))
    
    def get_im_display(self):

        return (self.im - self.min)/(self.max - self.min)
    
    def get_subim_display(self):
        return (self.subim - self.min)/(self.max - self.min)
    
    def update_figures(self):
        self.viewer.layers['volume'].data = self.tf.get_t(self.t)
        self.viewer.layers['roi'].data = np.array([self.s.threads[self.ind].get_position_t(self.t)])

        self.subim,self.offset = subaxis(self.im, self.s.threads[self.ind].get_position_t(self.t), self.window)
        self.ax1.clear()
        self.img1 = self.ax1.imshow(self.get_im_display(),cmap='gray',vmin = 0, vmax = 1)

        # plotting for multiple points
        if self.pointstate==0:
            pass
        elif self.pointstate==1:
            self.point1 = self.ax1.scatter(self.s.get_positions_t_z(self.t, self.s.threads[self.ind].get_position_t(self.t)[0])[:,2], self.s.get_positions_t_z(self.t,self.s.threads[self.ind].get_position_t(self.t)[0])[:,1],c='b', s=10)
        elif self.pointstate==2:
            self.point1 = self.ax1.scatter(self.s.get_positions_t(self.t)[:,2], self.s.get_positions_t(self.t)[:,1],c='b', s=10)
        self.thispoint = self.ax1.scatter(self.s.threads[self.ind].get_position_t(self.t)[2], self.s.threads[self.ind].get_position_t(self.t)[1],c='r', s=10)
        
        if self.pointstate==0:
            pass
        elif self.pointstate==1:
            self.point1.set_offsets(np.array([self.s.get_positions_t_z(self.t, self.s.threads[self.ind].get_position_t(self.t)[0])[:,2], self.s.get_positions_t_z(self.t,self.s.threads[self.ind].get_position_t(self.t)[0])[:,1]]).T)
        elif self.pointstate == 2:
            self.point1.set_offsets(np.array([self.s.get_positions_t(self.t)[:,2], self.s.get_positions_t(self.t)[:,1]]).T)
        self.thispoint.set_offsets([self.s.threads[self.ind].get_position_t(self.t)[2], self.s.threads[self.ind].get_position_t(self.t)[1]])

        self.static_canvas_1.draw()

        #plotting for single point
        self.ax2.clear()
        self.ax2.imshow(self.get_subim_display(),cmap='gray',vmin = 0, vmax =1)

        self.point2 = self.ax2.scatter(self.window/2+self.offset[0], self.window/2+self.offset[1],c='r', s=40)
        self.point2.set_offsets([self.window/2+self.offset[0], self.window/2+self.offset[1]])
        
        self.static_canvas_2.draw()
        self.series_label.setText('Series=' + str(self.ind) + ', Z=' + str(int(self.s.threads[self.ind].get_position_t(self.t)[0])))
        self.static_canvas_3.draw()

    def update_timeseries(self):
        self.timeax.clear()
        self.timeplot, = self.timeax.plot((self.timeseries[:,self.ind]-np.min(self.timeseries[:,self.ind]))/(np.max(self.timeseries[:,self.ind])-np.min(self.timeseries[:,self.ind])))
        self.static_canvas_3.draw()

    def update_t(self, val):
        # Update index for t
        self.t = val
        # update image for t
        self.update_im()
        self.update_figures()

    def update_mm(self, button, val):
        if 'min' == button:
            self.min = val
        elif 'max' == button:
            self.max = val
        self.update_figures()

    def next(self):
        self.set_index_next()
        self.update_im()
        self.update_figures()
        self.update_timeseries()
        self.update_buttons()
        self.update_curate()
    def prev(self):
        self.set_index_prev()
        self.update_im()
        self.update_figures()
        self.update_timeseries()
        self.update_buttons()
        self.update_curate()

    def log_curate(self):
        self.curate['last'] = self.ind
        with open(self.path, 'w') as fp:
            json.dump(self.curate, fp)

    def keep(self, label):
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
        self.update_point1()
        self.update_figures()

    def update_point1(self):
        self.ax1.clear()
        self.img1 = self.ax1.imshow(self.get_im_display(),cmap='gray',vmin = 0, vmax = 1)
        if self.pointstate==0:
            self.point1 = None
        elif self.pointstate==1:
            self.point1 = self.ax1.scatter(self.s.get_positions_t_z(self.t, self.s.threads[self.ind].get_position_t(self.t)[0])[:,2], self.s.get_positions_t_z(self.t,self.s.threads[self.ind].get_position_t(self.t)[0])[:,1],c='b', s=10)
        elif self.pointstate==2:
            self.point1 = self.ax1.scatter(self.s.get_positions_t(self.t)[:,2], self.s.get_positions_t(self.t)[:,1],c='b', s=10)
        self.thispoint = self.ax1.scatter(self.s.threads[self.ind].get_position_t(self.t)[2], self.s.threads[self.ind].get_position_t(self.t)[1],c='r', s=10)
        self.static_canvas_1.draw()

    def update_mipstate(self, label):
        d = {
            'Single Z':0,
            'MIP':1,
        }
        self.showmip = d[label]

        self.update_im()
        self.update_figures()

    def update_trace_icons(self):
        static_trace_canvas = FigureCanvas(Figure())
        trace_ax = static_trace_canvas.figure.subplots()
        for ind in range(self.tmax):
            trace_ax.clear()
            timeplot = trace_ax.plot((self.timeseries[:,ind]-np.min(self.timeseries[:,ind]))/(np.max(self.timeseries[:,ind])-np.min(self.timeseries[:,ind])))
            static_trace_canvas.draw()
            np_img = np.asarray(static_trace_canvas.buffer_rgba())[:,:,:3]
            h,w,d = np_img.shape
            q_img = QImage(np_img.tobytes(), h, w, QImage.Format_RGBA8888)
            icon = QIcon(QPixmap.fromImage(q_img))
            item = QListWidgetItem(icon, str(ind))
            self.trace_list.addItem(item)

    def go_to_trace(self, index):
        self.set_index(index)
        self.update_im()
        self.update_figures()
        self.update_timeseries()
        self.update_buttons()
        self.update_curate()
