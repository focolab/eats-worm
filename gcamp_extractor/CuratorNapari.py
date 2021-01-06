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
from qtpy.QtWidgets import QSlider, QButtonGroup, QLabel, QPushButton, QRadioButton
from qtpy.QtCore import Qt
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
        ## Figure to display
        self.fig = plt.figure()

        ## grid object for complicated subplot handing
        self.grid = plt.GridSpec(4, 2, wspace=0.1, hspace=0.2)


        ### First subplot: whole image with red dot over ROI
        self.ax1 = plt.subplot(self.grid[:3,0])
        plt.subplots_adjust(bottom=0.4)
        self.img1 = self.ax1.imshow(self.get_im_display(),cmap='gray',vmin = 0, vmax = 1)
        
        # plotting for multiple points
        
        if self.pointstate==0:
            pass
        elif self.pointstate==1:
            self.point1 = self.ax1.scatter(self.s.get_positions_t_z(self.t, self.s.threads[self.ind].get_position_t(self.t)[0])[:,2], self.s.get_positions_t_z(self.t,self.s.threads[self.ind].get_position_t(self.t)[0])[:,1],c='b', s=10)
        elif self.pointstate==2:
            self.point1 = self.ax1.scatter(self.s.get_positions_t(self.t)[:,2], self.s.get_positions_t(self.t)[:,1],c='b', s=10)
        self.thispoint = self.ax1.scatter(self.s.threads[self.ind].get_position_t(self.t)[2], self.s.threads[self.ind].get_position_t(self.t)[1],c='r', s=10)
        plt.axis('off')

        ### Second subplot: some window around the ROI
        plt.subplot(self.grid[:3,1])
        plt.subplots_adjust(bottom=0.4)

        self.subim,self.offset = subaxis(self.im, self.s.threads[self.ind].get_position_t(self.t), self.window)

        self.img2 = plt.imshow(self.get_subim_display(),cmap='gray',vmin = 0, vmax =1)
        self.point2 = plt.scatter(self.window/2+self.offset[0], self.window/2+self.offset[1],c='r', s=40)

        self.title = self.fig.suptitle('Series=' + str(self.ind) + ', Z=' + str(int(self.s.threads[self.ind].get_position_t(self.t)[0])))
        plt.axis("off")


        ### Third subplot: plotting the timeseries
        self.timeax = plt.subplot(self.grid[3,:])
        plt.subplots_adjust(bottom=0.4)
        self.timeplot, = self.timeax.plot((self.timeseries[:,self.ind]-np.min(self.timeseries[:,self.ind]))/(np.max(self.timeseries[:,self.ind])-np.min(self.timeseries[:,self.ind])))
        plt.axis("off")

        ### Axis for scrolling through t
        self.tr = plt.axes([0.2, 0.15, 0.3, 0.03], facecolor='lightgoldenrodyellow')
        self.s_tr = Slider(self.tr, 'Timepoint', 0, self.tmax-1, valinit=0, valstep = 1)
        self.s_tr.on_changed(self.update_t)

        ### Axis for setting min/max range
        self.minr = plt.axes([0.2, 0.2, 0.3, 0.03], facecolor='lightgoldenrodyellow')
        self.sminr = Slider(self.minr, 'R Min', 0, np.max(self.im), valinit=self.min, valstep = 1)
        self.maxr = plt.axes([0.2, 0.25, 0.3, 0.03], facecolor='lightgoldenrodyellow')
        self.smaxr = Slider(self.maxr, 'R Max', 0, np.max(self.im)*4, valinit=self.max, valstep = 1)
        self.sminr.on_changed(self.update_mm)
        self.smaxr.on_changed(self.update_mm)


        ### Axis for buttons for next/previous time series
        #where the buttons are, and their locations 
        self.axprev = plt.axes([0.62, 0.20, 0.1, 0.075])
        self.axnext = plt.axes([0.75, 0.20, 0.1, 0.075])
        self.bnext = Button(self.axnext, 'Next')
        self.bnext.on_clicked(self.next)
        self.bprev = Button(self.axprev, 'Previous')
        self.bprev.on_clicked(self.prev)


        #### Axis for button for display
        self.pointsax = plt.axes([0.75, 0.10, 0.1, 0.075])
        self.pointsbutton = RadioButtons(self.pointsax, ('Single','Same Z','All'))
        self.pointsbutton.set_active(self.pointstate)
        self.pointsbutton.on_clicked(self.update_pointstate)

        #### Axis for whether to display MIP on left
        self.mipax = plt.axes([0.62, 0.10, 0.1, 0.075])
        self.mipbutton = RadioButtons(self.mipax, ('Single Z','MIP'))
        self.mipbutton.set_active(self.showmip)
        self.mipbutton.on_clicked(self.update_mipstate)


        ### Axis for button to keep
        self.keepax = plt.axes([0.87, 0.20, 0.075, 0.075])
        self.keep_button = CheckButtons(self.keepax, ['Keep','Trash'], [False,False])
        self.keep_button.on_clicked(self.keep)


        ### Axis to determine which ones to show
        self.showax = plt.axes([0.87, 0.10, 0.075, 0.075])
        self.showbutton = RadioButtons(self.showax, ('All','Unlabelled','Kept','Trashed'))
        self.showbutton.set_active(self.show_settings)
        self.showbutton.on_clicked(self.show)
    
        ### new
        with napari.gui_qt():
            viewer = napari.Viewer(ndisplay=3)
            viewer.add_image(self.tf.get_t(self.t))
            viewer.add_points([self.s.threads[self.ind].get_position_t(self.t)], face_color='red')
            static_canvas_1 = FigureCanvas(Figure())
            axes_1 = static_canvas_1.figure.subplots()
            axes_1.imshow(self.get_im_display(),cmap='gray',vmin = 0, vmax = 1)
            axes_1.scatter(self.s.threads[self.ind].get_position_t(self.t)[2], self.s.threads[self.ind].get_position_t(self.t)[1],c='r', s=10)
            static_canvas_2 = FigureCanvas(Figure())
            axes_2 = static_canvas_2.figure.subplots()
            axes_2.imshow(self.get_subim_display(),cmap='gray',vmin = 0, vmax =1)
            axes_2.scatter(self.window/2+self.offset[0], self.window/2+self.offset[1],c='r', s=40)
            static_canvas_3 = FigureCanvas(Figure())
            axes_3 = static_canvas_3.figure.subplots()
            axes_3.plot((self.timeseries[:,self.ind]-np.min(self.timeseries[:,self.ind]))/(np.max(self.timeseries[:,self.ind])-np.min(self.timeseries[:,self.ind])))
            viewer.window.add_dock_widget(static_canvas_1, area='bottom', name='matplotlib figure')
            viewer.window.add_dock_widget(static_canvas_2, area='bottom', name='matplotlib figure')
            viewer.window.add_dock_widget(static_canvas_3, area='bottom', name='matplotlib figure')

            

            ### Axis for buttons for next/previous time series
            #where the buttons are, and their locations
            bprev = QPushButton('Previous')
            bprev.clicked.connect(lambda:self.prev())
            bnext = QPushButton('Next')
            bnext.clicked.connect(lambda:self.next())
            viewer.window.add_dock_widget([bprev, bnext])

            min_r_slider = QSlider()
            min_r_slider.setMaximum(int(np.max(self.im)))
            min_r_slider.setTickPosition(int(self.min))
            min_r_slider.setValue(int(self.min))
            min_r_slider.setOrientation(Qt.Horizontal)
            min_r_slider.valueChanged.connect(lambda:self.update_mm("min", max_r_slider.getValue()))
            max_r_slider = QSlider()
            max_r_slider.setMaximum(int(np.max(self.im)*4))
            max_r_slider.setTickPosition(int(self.max))
            max_r_slider.setValue(int(self.max))
            max_r_slider.setOrientation(Qt.Horizontal)
            max_r_slider.valueChanged.connect(lambda:self.update_mm("max", max_r_slider.getValue()))
            viewer.window.add_dock_widget([QLabel('R Min'), min_r_slider, QLabel('R Max'), max_r_slider], area='right')

            points_button_group = [QRadioButton('Single'), QRadioButton('Same Z'), QRadioButton('All')]
            points_button_group[0].setChecked(True)
            for button in points_button_group:
                button.toggled.connect(lambda:self.update_pointstate(button.text()))
            viewer.window.add_dock_widget(points_button_group, area='right')
            mip_button_group = [QRadioButton('Single Z'), QRadioButton('MIP')]
            mip_button_group[0].setChecked(True)
            for button in mip_button_group:
                button.toggled.connect(lambda:self.update_mipstate(button.text()))
            viewer.window.add_dock_widget(mip_button_group, area='right')
            keep_button_group = [QRadioButton('Keep'), QRadioButton('Trash')]
            if str(self.ind) in self.curate:
                keep_status = self.curate[str(self.ind)]
                if 'keep' == keep_status:
                    keep_button_group[0].setChecked(True)
                elif 'trash' == keep_status:
                    keep_button_group[1].setChecked(True)
            for button in keep_button_group:
                button.toggled.connect(lambda:self.keep(button.text()))
            viewer.window.add_dock_widget(keep_button_group, area='right')
            show_button_group = [QRadioButton('All'), QRadioButton('Unlabelled'), QRadioButton('Kept'), QRadioButton('Trashed')]
            show_button_group[0].setChecked(True)
            for button in show_button_group:
                button.toggled.connect(lambda:self.show(button.text()))
            viewer.window.add_dock_widget(show_button_group, area='right')

        plt.show()
    
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
        self.subim,self.offset = subaxis(self.im, self.s.threads[self.ind].get_position_t(self.t), self.window)
        self.img1.set_data(self.get_im_display())
        

        


        if self.pointstate==0:
            pass
        elif self.pointstate==1:
            self.point1.set_offsets(np.array([self.s.get_positions_t_z(self.t, self.s.threads[self.ind].get_position_t(self.t)[0])[:,2], self.s.get_positions_t_z(self.t,self.s.threads[self.ind].get_position_t(self.t)[0])[:,1]]).T)
        elif self.pointstate == 2:
            self.point1.set_offsets(np.array([self.s.get_positions_t(self.t)[:,2], self.s.get_positions_t(self.t)[:,1]]).T)
        self.thispoint.set_offsets([self.s.threads[self.ind].get_position_t(self.t)[2], self.s.threads[self.ind].get_position_t(self.t)[1]])
        plt.axis('off')
        #plotting for single point

        self.img2.set_data(self.get_subim_display())
        self.point2.set_offsets([self.window/2+self.offset[0], self.window/2+self.offset[1]])
        self.title.set_text('Series=' + str(self.ind) + ', Z=' + str(int(self.s.threads[self.ind].get_position_t(self.t)[0])))
        plt.draw()

    def update_timeseries(self):
        self.timeplot.set_ydata((self.timeseries[:,self.ind]-np.min(self.timeseries[:,self.ind]))/(np.max(self.timeseries[:,self.ind])-np.min(self.timeseries[:,self.ind])))
        plt.draw()
    def update_t(self, val):
        # Update index for t
        self.t = val
        # update image for t
        self.update_im()
        self.update_figures()

    def update_mm(self,val):
        self.min = self.sminr.val
        self.max = self.smaxr.val
        #self.update_im()
        self.update_figures()

    def next(self,event):
        self.set_index_next()
        self.update_im()
        self.update_figures()
        self.update_timeseries()
        self.update_buttons()
        self.update_curate()
    def prev(self,event):
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

        curr = self.keep_button.get_status()
        future = [False for i in range(len(curr))]
        if self.curate.get(str(self.ind))=='seen':
            pass
        elif self.curate.get(str(self.ind))=='keep':
            future[0] = True
        elif self.curate.get(str(self.ind))=='trash':
            future[1] = True
        else:
            pass

        for i in range(len(curr)):
            if curr[i] != future[i]:
                self.keep_button.set_active(i)

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
        plt.axis('off')
        if self.pointstate==0:
            self.point1 = None
        elif self.pointstate==1:
            self.point1 = self.ax1.scatter(self.s.get_positions_t_z(self.t, self.s.threads[self.ind].get_position_t(self.t)[0])[:,2], self.s.get_positions_t_z(self.t,self.s.threads[self.ind].get_position_t(self.t)[0])[:,1],c='b', s=10)
        elif self.pointstate==2:
            self.point1 = self.ax1.scatter(self.s.get_positions_t(self.t)[:,2], self.s.get_positions_t(self.t)[:,1],c='b', s=10)
        self.thispoint = self.ax1.scatter(self.s.threads[self.ind].get_position_t(self.t)[2], self.s.threads[self.ind].get_position_t(self.t)[1],c='r', s=10)
        plt.axis('off')

    def update_mipstate(self, label):
        d = {
            'Single Z':0,
            'MIP':1,
        }
        self.showmip = d[label]

        self.update_im()
        self.update_figures()




