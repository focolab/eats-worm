"""
An interactive GUI for viewing rainbow worms and picking/curating cell centers
using PyQtGraph. It has two linked ortho views, YX and ZX, on top and 
a row of data plot panels (being filled in) at the bottom. 

credits:
    https://www.youtube.com/watch?v=RHmTgapLu4s
    https://stackoverflow.com/a/49049204/6474403
    (layout scaling) https://groups.google.com/forum/#!topic/pyqtgraph/RJjtJea9KFc
"""
import os
import sys
import pdb
import json
import datetime
import random
import string
import time

from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph.opengl as gl
import pyqtgraph as pg
import pyqtgraph.console
import pyqtgraph.exporters
import pyqtgraph.opengl as gl
import numpy as np
import pandas as pd
import napari

from .data.tiffreader import TiffReader
from .blob import SegmentedBlob

def getRGBA(rgbw, levels):
    """convert RGBW data array to RGBA

    parameters
    ----------
    rgbw : (array) Color should be leading dimesion e.g. shape = (4, nx, ny)
        and should be in RGBW order
    levels : (list of tuples) for RGBW channels.

    returns
    -------
    out : (array) RGBA array (same shape as input) and set to np.uint16
    new_levels : (list of tuples) RGBW display limits

    pyqtgraph can only plot RGBA data, so this does an additive blend of RGB
    and W channels (and sets alpha=1 in the whole image) to create an RGBA
    array that can be plotted.
    """
    assert rgbw.shape[0] == 4

    def clip(x, levels):
        """applying display limits to map an array into [0, 1]"""
        return (np.clip(x, *levels) - levels[0]) / (levels[1]-levels[0])

    ## fixed scaling
    scalefactor = 30000   # just under half the uint16 max so we stay in bounds
    out = (rgbw*0).astype(np.uint16)
    W = clip(rgbw[3], levels[3])
    out[0] = (clip(rgbw[0], levels[0]) + W)*scalefactor
    out[1] = (clip(rgbw[1], levels[1]) + W)*scalefactor
    out[2] = (clip(rgbw[2], levels[2]) + W)*scalefactor
    out[3] = rgbw[3]*0 + scalefactor + 5
    new_levels = [(0, scalefactor)]*4

    return out, new_levels

class KeyPressWindow(pg.GraphicsWindow):
    """a graphics window that registers keyPressEvent
    
    sauce: https://stackoverflow.com/a/49049204/6474403
           https://stackoverflow.com/a/27477021/6474403
           https://stackoverflow.com/a/10568394/6474403

    TODO: GraphicsWindow is deprecated, need to redo this for
    GraphicsLayoutWidget. Better yet a LayoutWidget (which seems more general)
    """
    sigKeyPress = QtCore.pyqtSignal(object)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def keyPressEvent(self, ev):
        self.scene().keyPressEvent(ev)
        self.sigKeyPress.emit(ev)

class NPEXCurator(object):
    """PyQtGraph rainbow worm viewer and cell curator
    
    Although this is rather untamed, the view state and data state are the
    core elements of this GUI, and the rest is bells and whistles.

    VIEW STATE: The "view state" comprises a compact set of attributes
    ```self.view_xxx``` that dictate what is displayed. Most of the gui
    elements alter these view state attributes. Other attributes are derived
    from the view and/or data state.

    view_ax: (dict) Map between display (i,j,k) and data (X,Y,Z) dimensions
    view_axflip: (dict) dict(i=0, j=0) whether display axes are flipped
    view_blob_index: (int) -1 is reserved for the provisional "clicked" blob
    view_blob_index_prev: (int) Allows toggle between blob -1 and another blob
    view_layer: (int) Which layer to display
    view_k_index: (int) Which k-slice to display
    view_mip: (bool) display MIP (otherwise slice)
    view_centers: (str) in ['one', 'slice', 'all']
    view_slab: (bool) view a MIP slab instead of a single slice
    view_slab_pad: (int) the slab thickness is 1+2*pad

    DATA STATE: The data state comprises essentially what is passed to the
    constructor; raw image chunks (layers, immutible), the list of blobs
    (MUTIBLE), and a blob checklist (MUTIBLE). The central purpose of this tool
    is to create and edit the list of blobs, so the list of blobs is the core
    data state. Other data attributes -- mips, qrects, dum_ims, histos,
    etc. -- are derived, and used as helpers.

    """
    def __init__(self, layers=None, blobs=None, dest='viewer-output', checklist=None, meta=None,
                windows_mode=False):
        """
        input
        ------
        layers (list): a list of layer dicts, each having the keys:
            'name' : str
            'chunk' : DataChunk
        blobs ():
            if (str): TODO: treat as a josn file and import
            if (list): interpreted as list of SegmentedBlobs
            if None: empty list
        dest (str): destination folder
        checklist (dict): checklist of neurons that are expected to be found..
        meta (dict): metadata trojan horse.. used to load worm/date info
        windows_mode (Bool): If true, the 3D chunk widget (GL errors) is not built
        """

        pg.setConfigOptions(antialias=True)

        self.windows_mode = windows_mode
        self.user_ID = None

        # VIEW STATE
        self.view_ax = dict(i='Y', j='X', k='Z')
        self.view_axflip = dict(i=0, j=0)
        self.view_blob_index = -1
        self.view_blob_index_prev = -1
        self.view_layer = 0
        self.view_k_index = -1
        self.view_mip = True
        self.view_centers = 'one'

        self.view_slab = False
        self.view_slab_pad = 4

        self.view_bbox_pad = dict(X=8, Y=8, Z=3)

        self.dest = dest
        os.makedirs(dest, exist_ok=True)
        self.layers = layers
        self.meta = meta if meta is not None else {}
        self.napari = {}
        self.blobs = [SegmentedBlob(**b.to_jdict()) for b in blobs] if blobs is not None else []

        # precompute mips, crops, offsets, data shape+limits, etc...
        self.setup_mips()
        self.setup_qrects()
        self.setup_view_k_index()

        # parameters
        self.bbox_pad = [25, 20, 5]

        # starting blob
        if len(self.blobs) == 0:
            self.clickblob = SegmentedBlob(
                pos=[self.midbox['X'], self.midbox['Y'], self.midbox['Z']],
                dims=['X', 'Y', 'Z'],
                pad=[self.view_bbox_pad[k] for k in ['X', 'Y', 'Z']],
                prov='none'
                )
        else:
            self.view_blob_index = 0
            self.view_blob_index_prev = 0

        # Holds mouse pointer location in plotted coordinates, when hovering
        # over a panel.
        self.mouse_pos = dict()

        # checklist of blobs we expect to find
        if checklist is None:
            self.checklist = dict(list_items=[])
        else:
            self.checklist = checklist

        # the app and head widget
        self.app = QtCore.QCoreApplication.instance()
        if self.app is None:
            self.app = QtGui.QApplication(sys.argv)
        self.kpw = KeyPressWindow(title='npex curator')

        # setup the plots
        self.make_button_styles()
        self.setup_plots()

        # connect mouse click and key press (put this in setup?)
        #self.kpw.scene().sigMouseClicked.connect(lambda x: self.mouse_clicked(x, click_type='general'))
        self.kpw.sigKeyPress.connect(self.key_pressed)


    def get_zoom_bbox_border(self):
        """helper. compute the lines for the chunk bbox border"""
        #blob = self.get_current_blob()
        bl = self.get_current_blob()
        pad = self.view_bbox_pad

        # getting the bounding box for the blob
        getbox = lambda x,y: np.asarray([ [x[0], x[1], x[1], x[0], x[0]], [y[0],y[0], y[1], y[1], y[0]]])

        chreqX = bl.chreq(rounding='nearest', pad=pad)['X']
        chreqY = bl.chreq(rounding='nearest', pad=pad)['Y']
        chreqZ = bl.chreq(rounding='nearest', pad=pad)['Z']
        bbox_xy = getbox(chreqX, chreqY)
        bbox_xz = getbox(chreqX, chreqZ)

        chreqI = bl.chreq(rounding='nearest', pad=pad)[self.view_ax['i']]
        chreqJ = bl.chreq(rounding='nearest', pad=pad)[self.view_ax['j']]
        chreqK = bl.chreq(rounding='nearest', pad=pad)[self.view_ax['k']]
        bbox_ji = getbox(chreqJ, chreqI)
        bbox_jk = getbox(chreqJ, chreqK)

        return dict(
            XY=bbox_xy,
            XZ=bbox_xz,
            JI=bbox_ji,
            JK=bbox_jk
            )

    def get_zoom_chunk(self, pad=None):
        """helper. Given view state, get the zoom box DataChunk

        NOTE: something funky with negative z indexed chunk requests here?
        """
        blob = self.get_current_blob()
        if pad is None:
            pad = self.view_bbox_pad

        # get the zoomed chunk (bbox around blob center)
        # The offset handling is rather hacky here, could be tracked in DataChunk?
        layer = self.layers[self.view_layer]    # layer has an offset!
        req = blob.chreq(offset=layer['chunk'].meta.get('offset', None), pad=pad, rounding='nearest')
        ch_zoom = layer['chunk'].subchunk(req=req).squeeze()
        ch_zoom.meta['offset'] = blob.lower_corner()    # offset in raw image coords (is this used?)
        return ch_zoom

    def get_current_blob(self):
        """helper"""
        if self.view_blob_index == -1:
            blob = self.clickblob
        else:
            blob = self.blobs[self.view_blob_index]
        return blob

    def get_table(self):
        """table of blob info"""
        x = []
        for b in self.blobs:
            d = dict(
                blob=b.index,
                X=b.posd['X'],
                Y=b.posd['Y'],
                Z=b.posd['Z'],
                status=b.status,
                ID=b.ID,
                prov=b.prov
            )
            x.append(d)
        if len(self.blobs) == 0:
            x = [dict(blob=0, X=0, Y=0, Z=0, status=0, ID='')]

        return x

    def setup_view_k_index(self):
        """the initial view_k_index should be somewhere in the valid range
        but we do not know that range until self.axlim is computed. This fn
        just makes the decision explicit

        (should only be run once in the constructor, after self.axlim and
        self.view_ax are set)
        """
        try:
            self.view_k_index = self.axlim[self.view_ax['k']][0]
        except:
            self.view_k_index = 0

    def setup_mips(self):
        """precompute MIPS"""
        ans = []
        for layer in self.layers:
            if 'C' in layer['chunk'].dims:
                mipIJ = layer['chunk'].max_ip(dim=self.view_ax['k']).reorder_dims(['C', self.view_ax['i'], self.view_ax['j']]).squeeze()
                mipKJ = layer['chunk'].max_ip(dim=self.view_ax['i']).reorder_dims(['C', self.view_ax['k'], self.view_ax['j']]).squeeze()
            else:
                mipIJ = layer['chunk'].max_ip(dim=self.view_ax['k']).reorder_dims([self.view_ax['i'], self.view_ax['j']])
                mipKJ = layer['chunk'].max_ip(dim=self.view_ax['i']).reorder_dims([self.view_ax['k'], self.view_ax['j']])
            ans.append(
                dict(IJ=mipIJ, KJ=mipKJ)
                )
        self.mips = ans

    def setup_qrects(self):
        """setup qrects that are bounding boxes for each layer

        setup offsets and dimension ranges

        (need this to account for crop box offsets)
        (need to rerun this each time IJK are switched?)
        """
        offset = self.layers[0]['chunk'].meta['offset']

        dx = offset['X']
        dy = offset['Y']
        dz = offset['Z']
        lx = self.layers[0]['chunk'].dim_len['X']
        ly = self.layers[0]['chunk'].dim_len['Y']
        lz = self.layers[0]['chunk'].dim_len['Z']

        self.axlim = dict(
            X=[dx, dx+lx],
            Y=[dy, dy+ly],
            Z=[dz, dz+lz]
            )
        self.axlen = dict(
            X=lx,
            Y=ly,
            Z=lz
        )

        di = offset[self.view_ax['i']]
        dj = offset[self.view_ax['j']]
        dk = offset[self.view_ax['k']]
        li = self.layers[0]['chunk'].dim_len[self.view_ax['i']]
        lj = self.layers[0]['chunk'].dim_len[self.view_ax['j']]
        lk = self.layers[0]['chunk'].dim_len[self.view_ax['k']]

        # data midpoint, approximate
        self.midbox = dict(
            i=int(di+li/2),
            j=int(dj+lj/2),
            k=int(dk+lk/2),
            X=int(dx+lx/2),
            Y=int(dy+ly/2),
            Z=int(dz+lz/2),
        )

        # left top width height
        self.qr_ji = QtCore.QRectF(dj-0.5, di-0.5, lj, li)
        self.qr_jk = QtCore.QRectF(dj-0.5, dk-0.5, lj, lk)

    def mouseMoved(self, evt):
        """position tracking inside ONE image's viewbox

        TODO: can we avoid making one of these for every panel :D
        """
        p1 = self.im1
        vb = p1.getViewBox()
        pos = evt[0]  ## using signal proxy turns original arguments into a tuple
        if p1.sceneBoundingRect().contains(pos):
            mousePoint = vb.mapSceneToView(pos)
            self.mouse_pos['topXY'] = [mousePoint.x(), mousePoint.y()]
            self.mouse_pos['topJI'] = [mousePoint.x(), mousePoint.y()]
            #print(self.mouse_pos)
        else:
            self.mouse_pos['topXY'] = None
            self.mouse_pos['topJI'] = None

    def setup_hud(self):
        """heads up display of metadata"""
        self.hud1 = pg.TextItem(text='hud') #, anchor=(xmin, ymax))
        self.vb1.addItem(self.hud1)
        self.hud1.setZValue(10)

    def update_hud(self):
        """heads up display of metadata"""
        ##### update position
        xmin = self.axlim[self.view_ax['j']][self.view_axflip['j']]
        ymax = self.axlim[self.view_ax['i']][1-self.view_axflip['i']]
        self.hud1.setPos(xmin, ymax)

        #### update text
        # current datetime
        timestamp = datetime.datetime.now().replace(microsecond=0).isoformat().replace('T',' ')
        line1 = 'now  : %s\n' % timestamp

        # data tag
        line2 = 'data : %s\n' % self.meta.get('tag', 'data_tag')

        # display info
        name = self.layers[self.view_layer]['name']
        line3 = 'layer: %i [%s]\n' % (self.view_layer, name)
        if self.view_mip:
            line4 = 'view : MIP\n'
        else:
            if self.view_slab == 1:
                ki = self.view_k_index
                pd = self.view_slab_pad
                line4 = 'view : slab z=[%i, %i]\n' % (ki-pd, ki+pd)
            else:
                line4 = 'view : slice z=%i\n' % (self.view_k_index)

        txt = line1+line2+line3+line4
        self.hud1.setText(txt)
        self.hud1.setFont(QtGui.QFont('Monospace', 12))

    def setup_checklist(self):
        """checklist of ID'd neurons

        """
        for i, p in enumerate(self.checklist['list_items']):
            if len(p['ID']) == 1:
                p['TextItem'] = pg.TextItem(text=p['ID'][0], angle=65)
            elif len(p['ID']) == 2:
                p['TextItem'] = pg.TextItem(text=p['ID'][0]+' '+p['ID'][1], angle=65)
            self.vb1.addItem(p['TextItem'], ignoreBounds=True)
            p['TextItem'].setZValue(10)

    def update_checklist(self):
        # reposition, in case any axes were flipped
        for i, p in enumerate(self.checklist['list_items']):
            xmin = self.axlim[self.view_ax['j']][0]
            xmax = self.axlim[self.view_ax['j']][1]
            if self.view_axflip['j'] == 1:
                xmin, xmax = xmax, xmin
            xloc = xmin + i*(xmax-xmin)/len(self.checklist['list_items'])
            if self.view_axflip['i'] == 1:
                yloc = self.axlim[self.view_ax['i']][0]
            else:
                yloc = self.axlim[self.view_ax['i']][1]
            p['TextItem'].setPos(xloc, yloc)


        # use blob_IDs to update the checklist
        blob_IDs = list(set([b.ID for b in self.blobs]))

        cdic = {0:'red', 1:'yellow', 2:'green'}
        for item in self.checklist['list_items']:
            for i, ID in enumerate(item['ID']):
                if ID in blob_IDs:
                    item['status'][i] = 2
                elif ID+'?' in blob_IDs:
                    item['status'][i] = 1
                else:
                    item['status'][i] = 0

            colors = [cdic[j] for j in item['status']]
            if len(item['status']) == 2:
                html = '<font color="%s">%s</font> <font color="%s">(R)</font>' % (colors[0], item['ID'][0], colors[1])
            elif len(item['status']) == 1:
                html = '<font color="%s">%s</font>' % (colors[0], item['ID'][0])
            item['TextItem'].setHtml(html)



    def setup_plots(self):
        """a hot mess right now, just shoveling things in"""

        # This flag is used so that, during setup, update_plots() does not try
        # to update things that do not exit (throwing errors)
        self.is_setup = True

        self.main_layout = QtGui.QVBoxLayout()
        self.kpw.setLayout(self.main_layout)

        self.topp = pg.GraphicsLayoutWidget()
        self.topp.scene().sigMouseClicked.connect(lambda x: self.mouse_clicked(x, click_type='general'))
        self.main_layout.addWidget(self.topp)

        self.bottW = QtGui.QWidget()
        self.main_layout.addWidget(self.bottW)
        self.bott = QtGui.QHBoxLayout()
        self.bottW.setLayout(self.bott)
        self.bottW.setFixedHeight(350)

        #----------------------------------------------
        # TOP: the YX and YZ worm images plus the histogram LUT widget
        # self.vb1 = self.kpw.addPlot(row=0, col=0, colspan=4, title=self.make_big_title())    # returns PlotItem
        # self.vb2 = self.kpw.addPlot(row=1, col=0, colspan=4)
        self.vb1 = self.topp.addPlot(row=0, col=0, colspan=4) #, title=self.make_big_title())    # returns PlotItem
        self.vb2 = self.topp.addPlot(row=1, col=0, colspan=4)

        # these ImageItems are updatable :)
        self.im1 = pg.ImageItem(self.mips[self.view_layer]['IJ'].data.T)
        self.im1.setRect(self.qr_ji)
        #self.sc1 = pg.PlotDataItem()
        self.sc1 = pg.ScatterPlotItem()
        self.sc1box = pg.PlotDataItem()
        self.im2 = pg.ImageItem(self.mips[self.view_layer]['KJ'].data.T)
        self.im2.setRect(self.qr_jk)
        # TODO make sc2 a PlotDataItem (a la sc1)
        #self.sc2 = pg.PlotDataItem()
        self.sc2 = pg.ScatterPlotItem()
        self.sc2box = pg.PlotDataItem()
        self.ln2 = pg.PlotDataItem(x=self.axlim[self.view_ax['j']], y=[self.view_k_index]*2, symbol='o', pen='r')

        # TODO: trying to set click signals for sc1 (scatter plot)
        # WWRW: (hovering over blob)+click -> select blob, (not hovering)+click -> add blob
        # but hover signal not being emitted?!?!
        #self.sc1.sigPointsClicked.connect(lambda x: (self.mouse_clicked(x, click_type='scatter_point')))
        #self.sc1.sigClicked.connect(lambda x: (self.mouse_clicked(x, click_type='scatter_sigClicked')))
        #self.sc1.sigPointsHovered.connect(self.scatter_hover_status)

        # NOTE: set cache to False as workaround of bug
        # (https://github.com/NeuralEnsemble/ephyviewer/issues/132)
        self.sc1.opts['useCache'] = False
        self.sc2.opts['useCache'] = False

        # set up the checklist
        self.setup_checklist()

        # set up HUD (heads up display)
        self.setup_hud()

        # add each ImageItem to the respective PlotItem
        self.vb1.addItem(self.im1)
        self.vb1.addItem(self.sc1)
        self.vb1.addItem(self.sc1box)
        self.vb2.addItem(self.im2)
        self.vb2.addItem(self.sc2)
        self.vb2.addItem(self.sc2box)
        self.vb2.addItem(self.ln2)

        # mouse position tracking for im1 (full XY view)
        self.vb1proxy = pg.SignalProxy(self.im1.scene().sigMouseMoved, rateLimit=60, slot=self.mouseMoved)


        #----------------------------------------------
        # - Make one histogramLUTItem for each layer
        # - Connect each hLUTitem to a dummy image (one per layer, not displayed)
        # - The goal is for this to be an adjustable widget and we can use its
        #   LUT to update *all* images associated with one layer (whereas
        #   normally this item can only connect to one image)
        #   TODO: add a radio/toggle button at bottom to switch mono/rgba
        self.dum_ims = []
        self.histos = []
        for i, lay in enumerate(self.layers):
            if lay['chunk'].dim_len.get('C', 0) >= 3:
                levelMode = 'rgba'
                data = self.mips[i]['IJ'].reorder_dims([self.view_ax['i'],self.view_ax['j'],'C']).data
            else:
                levelMode = 'mono'
                data = self.mips[i]['IJ'].data
            im = pg.ImageItem(data)
            hi = pg.HistogramLUTItem(image=im)
            hi.setLevelMode(levelMode)
            self.dum_ims.append(im)
            self.histos.append(hi)
            self.histos[-1].setLevelMode(levelMode)

            if levelMode == 'mono':
                # fixes annoying overflow warning
                self.histos[-1].setLevels(min=float(np.min(data.ravel())), max=float(np.max(data.ravel())))


            self.histos[-1].sigLookupTableChanged.connect(self.update_plots)
            self.histos[-1].sigLevelsChanged.connect(self.update_plots)
            #self.histos[-1].sigLevelsChanged.connect(self.update_plots)

        self.vb3 = self.topp.addItem(self.histos[0], row=0, col=4, rowspan=1, colspan=1)
        self.histo_now = self.histos[self.view_layer]
        # this works, but introduces jitter
        # self.kpw.ci.layout.setColumnFixedWidth(7, 130)


        # Wide button prevents the sliders above from resizing everything else
        # Fix to make the button edges truly round (below):
        # https://falsinsoft.blogspot.com/2015/11/qt-snippet-rounded-corners-qpushbutton.html
        self.proxyX = QtGui.QGraphicsProxyWidget()
        self.buttonX = QtGui.QPushButton('face tune')
        self.buttonX.setStyleSheet(self.bstyle2)
        self.buttonX.setFixedWidth(150)
        self.buttonX.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)     # WOOO!
        self.buttonX.clicked.connect(self.auto_levels)
        self.proxyX.setWidget(self.buttonX)
        self.proxyX.setFocusPolicy(QtCore.Qt.NoFocus)
        self.vb4 = self.topp.addItem(self.proxyX, row=1, col=4, rowspan=1, colspan=1)

        # formatting, linking X-axes together
        self.vb1.getAxis('bottom').showLabel(False)
        self.vb1.getAxis('bottom').setStyle(showValues=False)
        #self.vb1.getAxis('bottom').setWidth(w=0)
        self.vb1.getAxis('bottom').setPen((20,20,20))

        self.vb1.setAspectLocked(lock=True)
        self.vb2.setAspectLocked(lock=True)
        self.vb1.setXLink(self.vb2)
        self.vb1.autoRange(padding=0.005)    # fix crazy initial scaling
        self.vb2.autoRange(padding=0.005)

        # vertical stretch of the ij and ik views
        stretch = self.axlen[self.view_ax['k']]/self.axlen[self.view_ax['i']]*1.1
        # print('stretch =', stretch)
        self.topp.ci.layout.setRowStretchFactor(0, 5)
        self.topp.ci.layout.setRowStretchFactor(1, max(np.round(5*stretch), 2))





        #----------------------------------------------
        # bottom row

        #---------------------------
        # CONSOLE :)
        namespace = {'pg': pg, 'np': np, 'gui':self}

        greycolor = "#c0c0c0"
        text = 'welcome\n'
        self.console = pyqtgraph.console.ConsoleWidget(namespace=namespace, text=text)
        self.console.setStyleSheet("background-color: black")
        self.console.output.setStyleSheet("background-color: %s" % (greycolor))
        self.console.input.setStyleSheet("background-color: %s" % (greycolor))
        self.console.ui.historyBtn.setStyleSheet("background-color: %s" % (greycolor))
        self.console.ui.exceptionBtn.setStyleSheet("background-color: %s" % (greycolor))
        self.bott.addWidget(self.console)
        self.console.setMinimumWidth(340)

        #---------------------------
        # 3D box
        v = np.array([[1,1,1], [-1,-1,1], [1,-1,-1], [-1,1,-1]])
        f = np.array([[0,1,2], [0,1,3], [1,2,3], [0,2,3]])
        c = np.array([[1,0,0,1], [0,1,0,1], [0,0,1,1], [1,1,1,1]])

        self.v3D = gl.GLViewWidget()
        self.v3D.setFocusPolicy(QtCore.Qt.NoFocus)
        self.bott.addWidget(self.v3D)
        self.v3D.setMinimumSize(200, 200)
        
        #---------------------------
        # zoom MIP box
        zch = self.get_zoom_chunk()
        #self.p7 = self.kpw.addLayout(row=2, col=2, colspan=1)
        #self.p7 = self.bott.addLayout()

        self.p7 = pg.GraphicsLayoutWidget()
        self.bott.addWidget(self.p7)

        self.vb1Z = self.p7.addPlot(row=0, col=0)
        self.vb2Z = self.p7.addPlot(row=1, col=0)

        # reorder dims?
        self.im1Z = pg.ImageItem(zch.max_ip(dim=self.view_ax['k']).data.T)
        self.im2Z = pg.ImageItem(zch.max_ip(dim=self.view_ax['i']).data.T)

        self.vb1Z.addItem(self.im1Z)
        self.vb2Z.addItem(self.im2Z)

        # formatting
        self.vb1Z.getAxis('bottom').showLabel(False)
        self.vb1Z.getAxis('bottom').setStyle(showValues=False)
        self.vb1Z.getAxis('bottom').setPen((20,20,20))
        self.vb1Z.setAspectLocked(lock=True)
        self.vb2Z.setAspectLocked(lock=True)
        self.vb1Z.setXLink(self.vb2Z)
        self.vb1Z.autoRange(padding=0.01)    # fix crazy initial scaling
        self.vb2Z.autoRange(padding=0.01)

        #self.p7.layout.setRowStretchFactor(0, 5)
        #self.p7.layout.setRowStretchFactor(1, 2)

        self.p7.sizeHint = lambda: pg.QtCore.QSize(350, 350)

        self.v3D.setSizePolicy(self.p7.sizePolicy())


        stretch = self.axlen[self.view_ax['k']]/self.axlen[self.view_ax['i']]
        #print(self.axlen)
        #print(self.view_ax)
        #print('stretch =', stretch)
        self.p7.ci.layout.setRowStretchFactor(0, 5)
        self.p7.ci.layout.setRowStretchFactor(1, max(np.round(5*stretch), 2))

        #---------------------------
        # add table of data
        self.tw = pg.TableWidget(sortable=False)
        self.tw.setData(self.get_table())
        # TODO make it look less bright and obnoxious
        # self.tw.setStyleSheet("alternate-background-color: yellow; border-radius: 25px; border-width: 2px;")
        self.tw.setSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.MinimumExpanding)
        self.tw.setMinimumWidth(350)
        self.bott.addWidget(self.tw)

        #---------------------------
        #### BUTTON BANK (lower right)
        def add_button(txt='text', fn=None, style=''):
            """helper for adding a button"""
            button = QtGui.QPushButton(txt)
            button.clicked.connect(fn)
            button.setStyleSheet(style)
            button.setFocusPolicy(QtCore.Qt.NoFocus)
            return button
        # widget/layout setup
        self.button_bank_LR = QtGui.QWidget()
        self.bott.addWidget(self.button_bank_LR)
        self.bbLR_layout = QtGui.QGridLayout()
        self.button_bank_LR.setLayout(self.bbLR_layout)
        # button styles (experimental)

        bstyle = self.bstyle2
        # left column
        self.buttonL1 = add_button(txt='one/slice/all (d)', fn=self.toggle_scatter, style=bstyle)
        self.buttonL2 = add_button(txt='slice/slab', fn=self.toggle_slab, style=bstyle)
        self.buttonL3 = add_button(txt='slab-', fn=lambda x: self.change_slab_size(-1), style=bstyle)
        self.buttonL4 = add_button(txt='slab+', fn=lambda x: self.change_slab_size(1), style=bstyle)
        self.buttonL5 = add_button(txt='sort_LR', fn=self.sort_blobs, style=bstyle)
        self.buttonL6 = add_button(txt='flip_IJ', fn=self.flip_IJ, style=bstyle)
        self.buttonL7 = add_button(txt='flip_LR', fn=self.flip_LR, style=bstyle)
        self.buttonL8 = add_button(txt='flip_UD', fn=self.flip_UD, style=bstyle)
        self.buttonL9 = add_button(txt='flip_reset', fn=self.reset_view, style=bstyle)
        self.bbLR_layout.addWidget(self.buttonL1, 0, 0)
        self.bbLR_layout.addWidget(self.buttonL2, 1, 0)
        self.bbLR_layout.addWidget(self.buttonL3, 2, 0)
        self.bbLR_layout.addWidget(self.buttonL4, 3, 0)
        self.bbLR_layout.addItem(QtGui.QSpacerItem(25, 25), 4, 0) #, 1, -1)
        self.bbLR_layout.addWidget(self.buttonL5, 5, 0)
        self.bbLR_layout.addWidget(self.buttonL6, 6, 0)
        self.bbLR_layout.addWidget(self.buttonL7, 7, 0)
        self.bbLR_layout.addWidget(self.buttonL8, 8, 0)
        self.bbLR_layout.addWidget(self.buttonL9, 9, 0)
        # middle column
        self.buttonM1 = add_button(txt='add blob to list', fn=self.add_blob, style=bstyle)
        self.buttonM2 = add_button(txt='cycle blob status', fn=self.cycle_blob_status, style=bstyle)
        self.buttonM3 = QtGui.QLineEdit(placeholderText='cell ID')
        #self.buttonM3.editingFinished.connect(self.update_ID)
        self.buttonM3.returnPressed.connect(self.update_blob_ID)
        #self.buttonM3.setStyleSheet('border-radius: 3px;')
        self.buttonM6 = add_button(txt='delete -1 blobs', fn=self.clear_blobs, style=self.bstyleR)
        self.buttonM7 = add_button(txt='HELP', fn=self.print_help, style=bstyle)
        self.buttonM8 = add_button(txt='PIKA', fn=self.pika_pika, style=self.bstyleY)
        self.bbLR_layout.addWidget(self.buttonM1, 0, 1)
        self.bbLR_layout.addWidget(self.buttonM2, 1, 1)
        self.bbLR_layout.addWidget(self.buttonM3, 2, 1)
        self.bbLR_layout.addWidget(self.buttonM6, 5, 1)
        self.bbLR_layout.addWidget(self.buttonM7, 8, 1)
        self.bbLR_layout.addWidget(self.buttonM8, 9, 1)
        # rightmost column
        self.buttonR1 = add_button(txt='screenshot', fn=self.export_screenshot, style=bstyle)
        self.buttonR2 = add_button(txt='sweep dump', fn=self.export_sweep, style=bstyle)
        self.buttonR3 = add_button(txt='to napari', fn=self.to_napari, style=bstyle)
        self.buttonR6 = add_button(txt='import blobs', fn=self._import, style=bstyle)
        self.buttonR7 = add_button(txt='export blobs', fn=self.export, style=bstyle)
        self.buttonR8 = QtGui.QLineEdit(placeholderText='user ID')
        self.buttonR8.returnPressed.connect(self.update_user_ID)
        self.bbLR_layout.addWidget(self.buttonR1, 0, 2)
        self.bbLR_layout.addWidget(self.buttonR2, 1, 2)
        self.bbLR_layout.addWidget(self.buttonR3, 2, 2)
        self.bbLR_layout.addWidget(self.buttonR6, 5, 2)
        self.bbLR_layout.addWidget(self.buttonR7, 6, 2)
        self.bbLR_layout.addWidget(self.buttonR8, 7, 2)

        # layout tuning
        self.bbLR_layout.setColumnMinimumWidth(0, 125)
        self.bbLR_layout.setColumnMinimumWidth(1, 125)
        self.bbLR_layout.setColumnMinimumWidth(2, 125)
        self.button_bank_LR.setSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.MinimumExpanding)
        self.bbLR_layout.setAlignment(QtCore.Qt.AlignTop)
        self.bbLR_layout.setHorizontalSpacing(15)

        #### final layout tuning
        self.kpw.resize(1600, 800)
        self.update_hud()
        self.update_checklist()
        self.update_v3D()
        self.is_setup = False

    def _import(self):
        """import blobs
        input is either of these:
        1. json file: list of serialized SegmentedBlob s
        2. json file: dict D, with D['blobs'] a list of serialized SegmentedBlob s
        """
        name = QtGui.QFileDialog.getOpenFileName(self.kpw, 'Open File', self.dest)

        with open(name[0]) as jfopen:
            data = json.load(jfopen)

        if isinstance(data, list):
            # list case
            self.blobs = [SegmentedBlob(**x) for x in data]
        else:
            # preferred, dict case
            d = data.get('blobs', [])
            self.blobs = [SegmentedBlob(**x) for x in d]

        self.console.write('imported: %s \n' % (os.path.relpath(name[0])))

        self.view_blob_index = 0
        self.tw.setData(self.get_table())
        self.update_checklist()
        self.update_plots()
        # print('---- import done ----')

    def make_button_styles(self):
        bstyle_base = "height: 25px; font-size: 14px; border-radius: 6px; border-style: outset; border-width: 2px;"
        bstyle1 = bstyle_base+"background-color: #a0a0a0; border-color: #b0b0b0; color: #000000"
        bstyle2 = bstyle_base+"background-color: #202020; border-color: #202030; color: #c0c0c0;"
        bstyleR = bstyle_base+"background-color: #f09090; border-color: red; color: #000000"
        bstyleY = bstyle_base+"background-color: #f0f090; border-color: yellow; color: #000000"
        self.bstyle1 = "QPushButton{%s} QPushButton::pressed{background-color: #808080}" % bstyle1
        self.bstyle2 = "QPushButton{%s} QPushButton::pressed{background-color: #404040}" % bstyle2
        self.bstyleR = "QPushButton{%s} QPushButton::pressed{background-color: #f05050}" % bstyleR
        self.bstyleY = "QPushButton{%s} QPushButton::pressed{background-color: #f0f010}" % bstyleY

    def pika_pika(self):
        text=''
        text+="⠸⣷⣦⠤⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⣠⣤⠀⠀⠀\n"
        text+="⠀⠙⣿⡄⠈⠑⢄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⠔⠊⠉⣿⡿⠁⠀⠀⠀\n"
        text+="⠀⠀⠈⠣⡀⠀⠀⠑⢄⠀⠀⠀⠀⠀⠀⠀⠀⠀⡠⠊⠁⠀⠀⣰⠟⠀⠀⠀⣀⣀\n"
        text+="⠀⠀⠀⠀⠈⠢⣄⠀⡈⠒⠊⠉⠁⠀⠈⠉⠑⠚⠀⠀⣀⠔⢊⣠⠤⠒⠊⠉⠀⡜\n"
        text+="⠀⠀⠀⠀⠀⠀⠀⡽⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠩⡔⠊⠁⠀⠀⠀⠀⠀⠀⠇\n"
        text+="⠀⠀⠀⠀⠀⠀⠀⡇⢠⡤⢄⠀⠀⠀⠀⠀⡠⢤⣄⠀⡇⠀⠀⠀⠀⠀⠀⠀⢰⠀\n"
        text+="⠀⠀⠀⠀⠀⠀⢀⠇⠹⠿⠟⠀⠀⠤⠀⠀⠻⠿⠟⠀⣇⠀⠀⡀⠠⠄⠒⠊⠁⠀\n"
        text+="⠀⠀⠀⠀⠀⠀⢸⣿⣿⡆⠀⠰⠤⠖⠦⠴⠀⢀⣶⣿⣿⠀⠙⢄⠀⠀⠀⠀⠀⠀\n"
        text+="⠀⠀⠀⠀⠀⠀⠀⢻⣿⠃⠀⠀⠀⠀⠀⠀⠀⠈⠿⡿⠛⢄⠀⠀⠱⣄⠀⠀⠀⠀\n"
        text+="⠀⠀⠀⠀⠀⠀⠀⢸⠈⠓⠦⠀⣀⣀⣀⠀⡠⠴⠊⠹⡞⣁⠤⠒⠉⠀⠀⠀⠀⠀\n"
        text+="⠀⠀⠀⠀⠀⠀⣠⠃⠀⠀⠀⠀⡌⠉⠉⡤⠀⠀⠀⠀⢻⠿⠆⠀⠀⠀⠀⠀⠀⠀\n"
        text+="⠀⠀⠀⠀⠀⠰⠁⡀⠀⠀⠀⠀⢸⠀⢰⠃⠀⠀⠀⢠⠀⢣⠀⠀⠀⠀⠀⠀⠀⠀\n"
        text+="⠀⠀⠀⢶⣗⠧⡀⢳⠀⠀⠀⠀⢸⣀⣸⠀⠀⠀⢀⡜⠀⣸⢤⣶⠀⠀⠀⠀⠀⠀\n"
        text+="⠀⠀⠀⠈⠻⣿⣦⣈⣧⡀⠀⠀⢸⣿⣿⠀⠀⢀⣼⡀⣨⣿⡿⠁⠀⠀⠀⠀⠀⠀\n"
        text+="⠀⠀⠀⠀⠀⠈⠻⠿⠿⠓⠄⠤⠘⠉⠙⠤⢀⠾⠿⣿⠟⠋          \n"
        text+="Pika Pika!!!\n"
        self.console.write(text)

    def get_datetime(self):
        """for timestamps"""
        timestamp = datetime.datetime.now().replace(microsecond=0).isoformat().replace(':','_')
        return timestamp

    def export_sweep(self):
        """dump png frames for a k-sweep"""
        self.console.write('export_sweep start ...\n')

        timestamp = self.get_datetime()
        suffix = ''.join(random.choices(string.ascii_lowercase, k=3))
        frame_dir = os.path.join(self.dest, 'checkpoints', 'k-sweep-%s-%s' % (timestamp, suffix))
        os.makedirs(frame_dir, exist_ok=True)

        klim = self.axlim[self.view_ax['k']]
        for k in range(klim[0], klim[1]):
            self.view_k_index = k
            self.update_plots()
            png = os.path.join(frame_dir, 'frame-k-%03i.png' % k)
            self.topp.grab().save(png)
        self.console.write('export_sweep complete\n')

    def export_screenshot(self):
        """"""
        cpdir = os.path.join(self.dest, 'checkpoints')
        os.makedirs(cpdir, exist_ok=True)
        timestamp = self.get_datetime()
        self.console.write('screenshot (%s)\n' % timestamp)
        png = os.path.join(cpdir, 'plot-screenshot-%s.png' % timestamp)
        self.kpw.grab().save(png)

    def export(self):
        """export blobs

        exports the following files into the directory self.dest
        ---
        blobs.json: includes a list of serialized SegmentedBlobs (used for
        import)
        blobs.csv: (partial) blob information in a spreadsheet (easy viewing)
        plot-gui.png: screenshot of the gui at the time when export is called

        Time stamped copies of blobs.json and blobs.csv are also written
        """
        if self.user_ID is None:
            self.console.write('must set user ID before exporting\n' )
            return

        timestamp = self.get_datetime()

        blobs = [b.to_jdict() for b in self.blobs]
        outdict = dict(
            user_ID=self.user_ID,
            timestamp=timestamp,
            blobs=blobs
            )

        cpdir = os.path.join(self.dest, 'checkpoints')
        os.makedirs(cpdir, exist_ok=True)

        out = os.path.join(cpdir, 'blobs-%s-%s.json' % (timestamp, self.user_ID))
        with open(out, 'w') as f:
            json.dump(outdict, f, indent=2)
            f.write('\n')

        dfs = pd.DataFrame(self.get_table()).sort_values(by=self.view_ax['j'])
        csv = os.path.join(cpdir, 'blobs-%s-%s.csv' %  (timestamp, self.user_ID))
        dfs.to_csv(csv, float_format='%6g')

        # without timestamp
        out = os.path.join(self.dest, 'blobs.json')
        with open(out, 'w') as f:
            json.dump(outdict, f, indent=2)
            f.write('\n')

        csv = os.path.join(self.dest, 'blobs.csv')
        dfs.to_csv(csv, float_format='%6g')

        png = os.path.join(self.dest, 'plot-gui.png')
        self.kpw.grab().save(png)

        self.console.write('------------------------\n')
        self.console.write('export timestamp: %s \n' % timestamp)
        self.console.write('export success  : %s \n' % out)
        self.console.write('export success  : %s \n' % csv)
        self.console.write('export success  : %s \n' % png)
        self.console.write('------------------------\n')

    def cycle_blob_status(self):
        """crude right now"""
        s = [-1, 0, 1]
        sdic = {x:i for i,x in enumerate(s)}

        b = self.get_current_blob()
        new_index = int(np.mod(sdic[b.status]+1, len(s))) 
        b.status = s[new_index]
        self.tw.setData(self.get_table())
        self.update_plots()

    def update_user_ID(self):
        """qlineEdit text is used to hold user ID"""
        #self.console.write('setting user ID\n')
        txt = self.buttonR8.text()
        if not txt.isalpha():
            self.console.write('user ID must be letters only\n')
            self.buttonR8.setText('')
        else:
            self.console.write('user ID set to %s\n' % txt)
            self.user_ID = txt
        self.buttonR8.clearFocus()

    def update_blob_ID(self):
        """qlineEdit text is used to update cell ID"""
        if self.view_blob_index >-1:
            txt = self.buttonM3.text()
            b = self.get_current_blob()
            old_ID = b.ID
            b.ID = txt
            if b.status != 1:
                b.status = 1
            # the text that gets displayed
            b.stash['gui_vb1_label'].setPlainText(b.ID)
            self.console.write('update ID: %s -> %s\n' % (old_ID, txt))
            self.tw.setData(self.get_table())
            self.update_checklist()
            self.update_plots()
        else:
            self.console.write('cannot ID provisional peak (add it first)\n')
        self.buttonM3.clearFocus()
        # TODO: cannot get the cursor to stop blinking using clearFocus or anything
        #focused_widget = QtGui.QApplication.focusWidget()
        #print('focused_widget:', focused_widget)


    def change_slab_size(self, step):
        if step<0:
            new = max(self.view_slab_pad+step, 1)
            self.console.write('decrease view_slab_pad -> %i \n' % new)
            self.view_slab_pad = new
            self.update_plots()
        elif step>0:
            new = min(self.view_slab_pad+step, 10)
            self.console.write('increase view_slab_pad -> %i \n' % new)
            self.view_slab_pad = new
            self.update_plots()
        else:
            new = self.view_slab_pad

    def auto_levels(self):
        """auto adjust color levels"""

        minZ = -1
        maxZ = 10
        numC = self.layers[self.view_layer]['chunk'].dim_len.get('C', 0)

        if numC == 4:
            data = self.mips[self.view_layer]['IJ'].reorder_dims(['C', self.view_ax['i'],self.view_ax['j']]).data
            data = data.reshape(4, -1)
            rgba = []
            for x in data:
                #xmin = np.mean(x) + minZ*np.std(x)
                #xmin = np.min(x.ravel())
                xmin = np.median(x.ravel())
                xmax = np.mean(x) + maxZ*np.std(x)
                rgba.append((xmin, xmax))
            levels = dict(rgba=rgba)
            txt = 'rgba'
        elif numC == 3:
            data = self.mips[self.view_layer]['IJ'].reorder_dims(['C', self.view_ax['i'],self.view_ax['j']]).data
            data = data.reshape(3, -1)
            rgba = []
            for x in data:
                #xmin = np.mean(x) + minZ*np.std(x)
                #xmin = np.min(x.ravel())
                xmin = np.median(x.ravel())
                xmax = np.mean(x) + maxZ*np.std(x)
                rgba.append((xmin, xmax))
            levels = dict(rgba=rgba)
            txt = 'rgb'
        else:
            data = self.mips[self.view_layer]['IJ'].data.ravel()
            levels = []
            x = data
            #xmin = np.mean(x) + minZ*np.std(x)
            #xmin = np.min(x.ravel())
            xmin = np.median(x.ravel())
            xmax = np.mean(x) + maxZ*np.std(x)
            levels = dict(min=xmin, max=xmax)
            txt = 'single channel'

        #print(txt)
        #print(data.shape)
        #print(levels)

        self.histos[self.view_layer].setLevels(**levels)

    def reindex_blobs(self):
        """"""
        for i, bl in enumerate(self.blobs):
            bl.index = i

    def add_blob(self):
        """clickblob gets promoted!"""
        if self.view_blob_index != -1:
            self.console.write('cannot add existing blob\n')
            return

        self.reindex_blobs()
        ndx = len(self.blobs)
        self.clickblob.index = ndx
        self.clickblob.status = 1
        self.blobs.append(self.clickblob)
        self.view_blob_index = ndx
        self.reindex_blobs()
        self.tw.setData(self.get_table())
        self.update_checklist()
        self.update_plots()
        self.console.write('blob added\n')

    def clear_blobs(self):
        """blobs with status -1 are dropped from the list"""
        junk = [b for b in self.blobs if b.status == -1]
        for j in junk:
            self.console.write('adios: %s %s \n' % (str(j.index), j.ID))
            self.vb1.removeItem(j.stash['gui_vb1_label'])
            _ = j.stash.pop('gui_vb1_label', None)
        self.blobs = [b for b in self.blobs if b.status != -1]
        self.view_blob_index = len(self.blobs)-1
        self.tw.setData(self.get_table())
        self.update_checklist()
        self.update_plots()

    def sort_blobs(self):
        """spatial J (horizontal axis) sort"""
        dfs = pd.DataFrame(self.get_table()).sort_values(by=self.view_ax['j'])
        ix = dfs.index.values
        if self.view_axflip['j'] == 1:
            ix = reversed(ix)

        b = [self.blobs[i] for i in ix]
        self.blobs = b
        self.tw.setData(self.get_table())
        self.update_plots()

    def print_help(self):
        msg = []
        msg.append('------------------------\n')
        msg.append('KEYBOARD SHORTCUTS:\n')
        msg.append('--------\n')
        msg.append('left/right: step through blob list\n')
        msg.append('up/down   : step through z-slices\n')
        msg.append('a         : toggle MIP and slice view\n')
        msg.append('d         : show one/slice/all points\n')
        msg.append('c         : step through layers\n')
        msg.append('--------\n')
        for x in msg:
            self.console.write(x)


    def mouse_clicked(self, event, click_type=None):
        """mouse click events are directed here

        TODO: make this just apply to the upper image panel, deprecate the
            GUI-wide mouse click event handling
        NOTE: event is a pyqtgraph.GraphicsScene.mouseEvents.MouseClickEvent
        TODO: factor out hardcoded pad
        """
        #self.console.write('CLICK\n')
        # if hovering over a scatter point, switch to the point
        if True in self.sc1.data['hovered']:
            # determine the new blob index given what is displayed
            ctr = self.get_centers()
            df_ix = np.where(self.sc1.data['hovered'] == True)[0][0]
            bl_ix = ctr['df_disp']['index'].values[df_ix]
            k_ix = ctr['df_disp']['kslice'].values[df_ix]

            self.view_k_index = k_ix
            self.view_blob_index = bl_ix
            self.update_plots()

        # otherwise, add a new point?
        else:
            #print('clicked plot 0x{:x}, event: {}'.format(id(self.kpw), event))
            if self.mouse_pos.get('topXY', None) is not None:
                if self.view_mip is False and event.button() == 1:
                    pos = np.rint(np.asarray(self.mouse_pos.get('topJI', None)))
                    dims = [
                        self.view_ax['j'],
                        self.view_ax['i'],
                        self.view_ax['k'],
                    ]
                    b = SegmentedBlob(
                        pos=[pos[0], pos[1], self.view_k_index],
                        dims=dims,
                        pad=self.bbox_pad,
                        #pad=self.view_bbox_pad,    # TODO make this work
                        prov='curated'
                        )
                    self.clickblob = b
                    if self.view_blob_index != -1:
                        self.view_blob_index_prev = self.view_blob_index
                    self.view_blob_index = -1
                    self.update_plots()
                    #print('clicked in the top panel! pos=', pos)

    def to_napari(self):
        """spawns a napari viewer with points layer

        Updates points when called again and the viewer still exists.
        Very helpful for picking out dim/tricky peaks.
        Currently only works for the RGBW layer.
        """
        # dig out metadata
        print('get voxel size from layer (or chunk)')
        voxel_size = self.meta.get('voxel_size', dict(X=1, Y=1, Z=1))
        scale = [voxel_size[k] for k in ['Z', 'Y', 'X']]
        chunk = self.layers[self.view_layer]['chunk']
        offset = chunk.meta.get('offset', dict(X=0, Y=0, Z=0))
        dims = ['Z', 'Y', 'X']

        ## trying to feed text to napari
        IDVEC = [b.ID for b in self.blobs]
        properties = {'ID': IDVEC}
        text_parameters = dict(
            text='{ID}',
            anchor='upper_left'
            )

        # if chunk.dim_len.get('C', 0)<4:
        #     self.console.write('womp womp.. to napari only works for RGBW so far\n')
        #     return

        try:
            ## this will throw an error if the napari window was closed
            self.napari['points_layer'].data = [[1,2,3]]
            exists = True
        except:
            self.napari['viewer'] = napari.Viewer()
            exists = False

        if exists:
            ## just update the points
            dd = self.get_centers()
            points = list(zip(np.asarray(dd['z'])-offset['Z'], np.asarray(dd['y'])-offset['Y'], np.asarray(dd['x'])-offset['X']))
            self.napari['points_layer'].data = points

            ## trying to update points and properties but hitting errors
            #self.napari['points_layer'].properties = properties
            #self.napari['points_layer']._TextManager.values = IDVEC
            #self.napari['points_layer']._TextManager.refresh_text(properties)

            ## update layers
            if chunk.dim_len.get('C', 0)==4:
                for i, colormap in enumerate(['red', 'green', 'blue', 'gray']):
                    if self.view_slab == 0:
                        channel = chunk.subchunk(req=dict(C=i)).squeeze().reorder_dims(dims).data
                        self.napari['viewer'].layers[colormap].data = channel
                        self.napari['viewer'].layers[colormap].translate = [0, 0, 0]
                    else:
                        req = self.get_slab_req()
                        req['C'] = i
                        channel = chunk.subchunk(req=req).squeeze().reorder_dims(dims).data
                        self.napari['viewer'].layers[colormap].data = channel
                        self.napari['viewer'].layers[colormap].translate = [req.get(k, [0])[0]*voxel_size[k] for k in dims]

            ## update contrast_limits
            if chunk.dim_len.get('C', 0)==4:
                lvl = self.histos[self.view_layer].getLevels()
                for i, colormap in enumerate(['red', 'green', 'blue', 'gray']):
                    self.napari['viewer'].layers[colormap].contrast_limits = lvl[i]

        else:
            ## initialize image and points layers
            if chunk.dim_len.get('C', 0)==4:
                lvl = self.histos[self.view_layer].getLevels()
                ## add image and points layers
                for i, colormap in enumerate(['red', 'green', 'blue', 'gray']):
                    if self.view_slab == 0:
                        channel = chunk.subchunk(req=dict(C=i)).squeeze().reorder_dims(dims).data
                        self.napari['viewer'].add_image(channel, scale=scale, colormap=colormap, name=colormap, blending='additive', interpolation='bicubic', gamma=0.6, contrast_limits=lvl[i])
                    else:
                        req = self.get_slab_req()
                        req['C'] = i
                        channel = chunk.subchunk(req=req).squeeze().reorder_dims(dims).data
                        self.napari['viewer'].add_image(channel, scale=scale, colormap=colormap, name=colormap, blending='additive', interpolation='bicubic', gamma=0.6, contrast_limits=lvl[i])

            elif chunk.dim_len.get('C', 0)==0:
                    channel = chunk.reorder_dims(dims).data
                    self.napari['viewer'].add_image(channel, scale=scale,  blending='additive', interpolation='bicubic', gamma=0.6)


            points = [[b.posd[k]-offset[k] for k in dims] for b in self.blobs]
            dd = dict(properties=properties, text=text_parameters)
            self.napari['points_layer'] = self.napari['viewer'].add_points(points, scale=scale, size=1.5) #, **dd)

    def get_slab_req(self):
        """"""
        kmid = self.view_k_index-self.axlim[self.view_ax['k']][0]
        kmin = max(kmid - self.view_slab_pad, 0)
        kmax = min(kmid + self.view_slab_pad+1, self.axlen[self.view_ax['k']])
        req = {self.view_ax['k']:(kmin, kmax)}
        return req

    def get_centers(self):
        """get blob center coordinates, for GUI display

        self.view_centers cases
        ------
        all: show everything
        slice: only in this k-slice
        one: only the current blob

        yes, this is hideous
        """
        color_map = {
            99:QtGui.QColor(0,0,220),
            0:QtGui.QColor(220,20,20),
            1:QtGui.QColor(220,220,20),
            2:QtGui.QColor(20,220,20)
        }

        # make a dataframe of all blob centers
        jj = [b.posd[self.view_ax['j']] for b in self.blobs]
        ii = [b.posd[self.view_ax['i']] for b in self.blobs]
        kk = [b.posd[self.view_ax['k']] for b in self.blobs]
        status = [b.status for b in self.blobs]
        ID = [b.ID for b in self.blobs]
        ID_status = []
        index = list(range(len(ID)))
        for b in self.blobs:
            if b.ID == '':
                ID_status.append(0)
            elif b.ID.find('?')>0:
                ID_status.append(1)
            else:
                ID_status.append(2)
        if self.view_blob_index == -1:
            # provisional blob
            jj.append(self.clickblob.posd[self.view_ax['j']])
            ii.append(self.clickblob.posd[self.view_ax['i']])
            kk.append(self.clickblob.posd[self.view_ax['k']])
            ID.append('')
            ID_status.append(99)
            index.append(-1)
            status.append(0)

        data = list(zip(jj, ii, kk, ID, ID_status, index, status))
        cols = ['j', 'i', 'k', 'ID', 'ID_status', 'index', 'status']
        df_ctr = pd.DataFrame(data=data, columns=cols)
        # df_ctr['kslice'] = np.floor(df_ctr['k']).astype(int)
        df_ctr['kslice'] = np.rint(df_ctr['k']).astype(int)
        size_vec = [8]*len(df_ctr)
        size_vec[self.view_blob_index] = 12
        df_ctr['size'] = size_vec

        # display column
        if self.view_centers == 'one':
            disp = [False]*len(df_ctr)
            disp[self.view_blob_index] = True
            df_ctr['display'] = disp
        if self.view_centers == 'slice':
            if self.view_slab == 0:
                df_ctr['display'] = df_ctr['kslice'] == self.view_k_index
            elif self.view_slab == 1:
                kk = df_ctr['kslice']
                kmin = self.view_k_index - self.view_slab_pad
                kmax = self.view_k_index + self.view_slab_pad
                col = (kk >= kmin) & (kk<= kmax)
                df_ctr['display'] = col
        if self.view_centers == 'all':
            df_ctr['display'] = [True]*len(df_ctr)

        # add display status to the blobs themselves
        for i,b in enumerate(self.blobs):
            b.stash['visible'] = df_ctr['display'][i]

        # cull down to those currently being displayed
        df_disp = df_ctr[df_ctr['display'] == True]
        
        # set symbols and colors
        #symb = ['o']*len(df_disp)

        symbol_map = {-1:'x', 0:'d', 1:'o', 2:'o'}
        symb = [symbol_map[i] for i in df_disp['status']]
        #print(symb)
        brush = [color_map[i] for i in df_disp['ID_status']]

        dd = dict(
            x=list(df_disp['j'].values),
            y=list(df_disp['i'].values),
            z=list(df_disp['k'].values),
            size=df_disp['size'].values,
            symb=symb,
            brush=brush,
            ID=df_disp['ID'].values,
            df_ctr=df_ctr,
            df_disp=df_disp
        )
        return dd

    def update_v3D(self, sliceDensity=20):
        """update the 3D chunk view

        Nothing here takes long, suspect that slow 3D rendering is causing lag
        TODO if key is held down, use low sliceDensity for speeeed :)
        """
        times = []
        alpha = 3

        times.append(time.time())

        self.v3D.clear()

        times.append(time.time())

        # pad this chunk to be 2x the size of the view_bbox
        pad3d = {k:2*v for k, v in self.view_bbox_pad.items()}
        ch3d = self.get_zoom_chunk(pad=pad3d)

        times.append(time.time())

        # center marker point
        p = np.array(list(pad3d.values()))+0.5
        c = np.array([200,0,0,1])
        s = np.array([6])

        times.append(time.time())

        orderXYZ = ['Z', 'Y', 'X']
        orderXYZC = ['C', 'Z', 'Y', 'X']
        if self.is_setup:
            ctr = np.flip(np.asarray([ch3d.dim_len[k] for k in orderXYZ]))//2+0.5
            self.v3D.opts['center'] = pg.Vector(*ctr)
            self.v3D.update()
        times.append(time.time())

        if 'C' not in ch3d.dims:
            xyzc = np.asarray([ch3d.reorder_dims(orderXYZ).data]*4).T

            # apply histo LUT levels
            lvl = self.histo_now.getLevels()
            xyzc = (np.clip(xyzc, lvl[0], lvl[1])-lvl[0])/(lvl[1]-lvl[0])*254
            # skewers
            pad = list(pad3d.values())
            xyzc[pad[0], pad[1], :] = 254
            xyzc[pad[0], :, pad[2]] = 254
            xyzc[:, pad[1], pad[2]] = 254
            # alpha (hackyy)
            xyzc[:,:,:,3]*=0
            xyzc[:,:,:,3]+=alpha
            self.vol3d = gl.GLVolumeItem(xyzc, smooth=False, sliceDensity=sliceDensity, glOptions='additive')
            self.v3D.addItem(self.vol3d)
        else:
            if ch3d.dim_len['C'] == 4:
                # applies histo LUT levels
                czyx = ch3d.reorder_dims(orderXYZC).data
                czyx, new_levels = getRGBA(czyx, self.histo_now.getLevels())
                xyzc = czyx.T
                for i, lvl in enumerate(new_levels):
                    arr = np.clip(xyzc[:,:,:,i], lvl[0], lvl[1])
                    xyzc[:,:,:,i] = (arr-lvl[0])/(lvl[1]-lvl[0])*254
                    # skewers
                    pad = list(pad3d.values())
                    xyzc[pad[0], pad[1], :, i] = 254
                    xyzc[pad[0], :, pad[2], i] = 254
                    xyzc[:, pad[1], pad[2], i] = 254

                # alpha (hacky)
                xyzc[:,:,:,3]*=0
                xyzc[:,:,:,3]+=alpha
                self.vol3d = gl.GLVolumeItem(xyzc, smooth=False, sliceDensity=sliceDensity, glOptions='additive')
                self.v3D.addItem(self.vol3d)
            elif ch3d.dim_len['C'] == 3:
                pass

        times.append(time.time())

        # print('timing')
        # for i in range(1, len(times)):
        #     print('time %i %i: %7.2f' % (i, i-1, (times[i]-times[i-1])*1000))


    def update_plots(self):
        """yup"""

        if self.is_setup:
            return

        # QLineEdit that holds Cell ID string
        self.buttonM3.setText(self.get_current_blob().ID)
        self.buttonM3.clearFocus()

        #item(row, column).setText("Put here whatever you want!")
        # self.tw.setData(self.get_table())
        # scroll table to the current blob
        if self.view_blob_index >= 0:
            self.tw.setCurrentCell(self.view_blob_index, 0)
            self.tw.scrollToItem(self.tw.item(self.view_blob_index, 0))

        # table/cell formatting:
        # https://stackoverflow.com/a/18905408/6474403

        # Refresh the LUT slider widget on the upper right.
        # This is important to get the lut and levels for the shown images!
        try:
            self.topp.removeItem(self.histo_now)
        except:
            pass
        self.histo_now = self.histos[self.view_layer]
        #self.vb3 = self.kpw.addItem(self.histo_now, row=0, col=4, rowspan=2, colspan=1)
        self.vb3 = self.topp.addItem(self.histo_now, row=0, col=4, rowspan=1, colspan=1)
        lut = self.histo_now.getLookupTable(img=self.dum_ims[self.view_layer], n=50)
        lvl = self.histo_now.getLevels()

        # this works, but introduces annoying jitter
        # self.kpw.ci.layout.setColumnFixedWidth(7, 130)


        #--------------------------------
        # TOP FULL CHUNK VIEW
        # get data for top full stack
        if self.view_mip:
            d1 = self.mips[self.view_layer]['IJ']
        else:
            # need to reorder_dims([self.view_ax['i'], self.view_ax['j']]).data.T
            if 'C' in self.layers[self.view_layer]['chunk'].dims:
                dims = ['C', self.view_ax['i'], self.view_ax['j']]
            else:
                dims = [self.view_ax['i'], self.view_ax['j']]

            if self.view_slab == 1:
                kmid = self.view_k_index-self.axlim[self.view_ax['k']][0]
                kmin = max(kmid - self.view_slab_pad, 0)
                kmax = min(kmid + self.view_slab_pad+1, self.axlen[self.view_ax['k']])
                req = {self.view_ax['k']:(kmin, kmax)}
                d1 = self.layers[self.view_layer]['chunk'].subchunk(req=req).max_ip(dim=self.view_ax['k']).squeeze().reorder_dims(dims)
            else:
                req = {self.view_ax['k']:self.view_k_index-self.axlim[self.view_ax['k']][0]}
                d1 = self.layers[self.view_layer]['chunk'].subchunk(req=req).squeeze().reorder_dims(dims)
        d2 = self.mips[self.view_layer]['KJ']

        # set the top full stack images (w appropriate lut/levels)
        if self.layers[self.view_layer]['chunk'].dim_len.get('C', 0) ==4:
            #RGBW to RGBA conversion
            out1, new_levels = getRGBA(d1.data, lvl)
            out2, new_levels = getRGBA(d2.data, lvl)
            self.im1.setImage(out1.T, lut=lut, levels=new_levels)
            self.im2.setImage(out2.T, lut=lut, levels=new_levels)
        else:
            self.im1.setImage(d1.data.T, lut=lut, levels=lvl)
            self.im2.setImage(d2.data.T, lut=lut, levels=lvl)

        # draw the z-plane on the xz view (when not in MIP mode)
        if self.view_mip:
            self.ln2.setData(x=[], y=[])
        else:
            if self.view_slab == 1:
                xlim = self.axlim[self.view_ax['j']]
                ylim = [self.view_k_index-self.view_slab_pad, self.view_k_index+self.view_slab_pad+1]
                x = [xlim[0], xlim[1], xlim[1], xlim[0], xlim[0]]
                y = [ylim[0], ylim[0], ylim[1], ylim[1], ylim[0]]
                self.ln2.setData(x=x, y=y)

            elif self.view_slab == 0:
                self.ln2.setData(x=self.axlim[self.view_ax['j']], y=[self.view_k_index]*2)

        # plot points for blob centers
        ctr = self.get_centers()

        # it is not clear whether sc1/sc2 should be PlotDataItems or ScatterPlotItems
        # PlotDataItems (use correct brush colors, but sth messed up with hover/click signals)
        #self.sc1.setData(x=ctr['x'], y=ctr['y'], symbol=ctr['symb'], symbolBrush=ctr['brush'], symbolSize=ctr['size'], pen=None)
        #self.sc2.setData(x=ctr['x'], y=ctr['z'], symbol=ctr['symb'], symbolBrush=ctr['brush'], symbolSize=ctr['size'], pen=None)

        # ScatterPlotItems (bug with brush colors, but these emit hover/click signals)
        self.sc1.setData(x=ctr['x'], y=ctr['y'], symbol=ctr['symb'], brush=ctr['brush'], size=ctr['size'], hoverable=True, hoverSize=20)
        self.sc2.setData(x=ctr['x'], y=ctr['z'], symbol=ctr['symb'], brush=ctr['brush'], size=ctr['size'])

        # plot the labels that accompany the blob centers
        for i, b in enumerate(self.blobs):
            if b.stash.get('gui_vb1_label', None) is None:
                b.stash['gui_vb1_label'] = pg.TextItem(text=b.ID)
                #, color=(20,20,20), fill=(150,150,150,150))
                self.vb1.addItem(b.stash['gui_vb1_label'])

            #### trying to "get" the TextItem text..
            #print(b.stash['gui_vb1_label'].toPlainText())

            xx = b.posd[self.view_ax['j']]
            yy = b.posd[self.view_ax['i']]
            b.stash['gui_vb1_label'].setPos(xx, yy)

            if b.stash.get('visible', False):
                b.stash['gui_vb1_label'].setVisible(True)
            else:
                b.stash['gui_vb1_label'].setVisible(False)


        # top bbox overlay
        bbb = self.get_zoom_bbox_border()
        if self.view_blob_index == -1:
            pp = 'b'
        else:
            pp = 'c'
        self.sc1box.setData(x=bbb['JI'][0].tolist(), y=bbb['JI'][1].tolist(), pen=pp)
        self.sc2box.setData(x=bbb['JK'][0].tolist(), y=bbb['JK'][1].tolist(), pen=pp)

        #--------------------------------
        # BOTTOM, ZOOM CHUNK VIEW
        #--------------------------------
        # zoom box
        zch = self.get_zoom_chunk()

        if 'C' in zch.dims:
            imIJ = zch.max_ip(dim=self.view_ax['k']).reorder_dims(['C', self.view_ax['i'], self.view_ax['j']]).squeeze().data.T
            imKJ = zch.max_ip(dim=self.view_ax['i']).reorder_dims(['C', self.view_ax['k'], self.view_ax['j']]).squeeze().data.T
            if self.layers[self.view_layer]['chunk'].dim_len.get('C', 0) == 4:
                imIJ, lvlIJ = getRGBA(imIJ.T, lvl)
                imKJ, lvlKJ = getRGBA(imKJ.T, lvl)
                imIJ = imIJ.T
                imKJ = imKJ.T
                lvl = lvlIJ
        else:
            imIJ = zch.max_ip(dim=self.view_ax['k']).reorder_dims([self.view_ax['i'], self.view_ax['j']]).data.T
            imKJ = zch.max_ip(dim=self.view_ax['i']).reorder_dims([self.view_ax['k'], self.view_ax['j']]).data.T
        self.im1Z.setImage(imIJ, lut=lut, levels=lvl, border=pp)
        self.im2Z.setImage(imKJ, lut=lut, levels=lvl, border=pp)

        # update hud
        self.update_hud()

        self.update_v3D()

    def reset_view(self):
        """reset view state"""
        self.view_axflip = dict(i=0, j=0)
        for x in [self.vb1, self.vb2, self.vb1Z, self.vb2Z]:
            x.invertX(False)
            x.invertY(False)

        # hacky, reuse flip_IJ to redraw
        self.view_ax = dict(i='X', j='Y', k='Z')
        self.flip_IJ()

    def toggle_slab(self):
        if self.view_slab == 0:
            self.view_slab = 1
            #self.console.write('SLAB VIEW ON\n')
        elif self.view_slab == 1:
            self.view_slab = 0
            #self.console.write('SLAB VIEW OFF\n')
        if not self.view_mip:
            self.update_plots()

    def flip_LR(self):
        """flip view left right"""
        if self.view_axflip['j']==0:
            self.vb1.invertX(True)
            self.vb2.invertX(True)
            self.vb1Z.invertX(True)
            self.vb2Z.invertX(True)
            self.view_axflip['j'] = 1
        else:
            self.vb1.invertX(False)
            self.vb2.invertX(False)
            self.vb1Z.invertX(False)
            self.vb2Z.invertX(False)
            self.view_axflip['j'] = 0
        self.update_checklist()
        self.update_plots()

    def flip_UD(self):
        """flip view up/down"""
        if self.view_axflip['i']==0:
            self.vb1.invertY(True)
            self.vb1Z.invertY(True)
            self.view_axflip['i'] = 1
        else:
            self.vb1.invertY(False)
            self.vb1Z.invertY(False)
            self.view_axflip['i'] = 0
        self.update_checklist()
        self.update_plots()

    def flip_IJ(self):
        """transpose the two leading dimensions"""
        kk = list(self.view_ax.keys())
        vv = list(self.view_ax.values())
        self.view_ax = {
            kk[0]:vv[1],
            kk[1]:vv[0],
            kk[2]:vv[2]
        }

        # recompute and replot
        self.setup_mips()
        self.setup_qrects()
        self.update_checklist()

        self.update_plots()
        self.im1.setRect(self.qr_ji)
        self.im2.setRect(self.qr_jk)
        self.vb1.autoRange(padding=0.005)    # fix crazy initial scaling
        self.vb2.autoRange(padding=0.005)
        self.vb1Z.autoRange(padding=0.01)    # fix crazy initial scaling
        self.vb2Z.autoRange(padding=0.01)

    def toggle_scatter(self):
        """button and key both connect here"""
        nextt = {
            'one':'slice',
            'slice':'all',
            'all':'one'}
        self.view_centers = nextt[self.view_centers]
        self.update_plots()

    def key_pressed(self, event):
        """key press events are directed here"""
        ### uncomment this to find out a key's int code
        # print("Key %i pressed" % event.key())
        # if event.modifiers() == QtCore.Qt.ShiftModifier:
        #     print("  shift")

        # IMPORTANT catch if we are editing any QLineEdit widgets
        if self.buttonM3.hasFocus():
            return
        if self.buttonR8.hasFocus():
            return
        if self.console.input.hasFocus():
            return


        if event.key() == 67:
            # the 'c' key toggles layer
            self.view_layer = np.mod(self.view_layer+1, len(self.layers))
            self.update_plots()

        if event.key() == 65:
            # the 'a' key toggles MIP or single z-plane
            self.view_mip = not self.view_mip
            self.update_plots()

        if event.key() == 68:
            # the 'd' key scatter plot of all points
            self.toggle_scatter()
            self.update_plots()

        if event.key() == 16777234:
            # left key decreases view_blob_index
            if self.view_blob_index == -1:
                self.view_blob_index = self.view_blob_index_prev
                #self.view_blob_index_prev = -1
            else:
                self.view_blob_index = max(self.view_blob_index-1, 0)
            self.view_k_index = np.rint(self.get_current_blob().posd[self.view_ax['k']]).astype(int)
            self.update_plots()

        if event.key() == 16777236:
            # right key increases view_blob_index
            if self.view_blob_index == -1:
                self.view_blob_index = self.view_blob_index_prev
                #self.view_blob_index_prev = -1
            else:
                self.view_blob_index = min(self.view_blob_index+1, len(self.blobs)-1)
            self.view_k_index = np.rint(self.get_current_blob().posd[self.view_ax['k']]).astype(int)
            self.update_plots()

        if event.key() == 16777235:
            # the 'up' key increments z-plane step +1 (and de-selects current blob)
            if self.view_blob_index != -1:
                self.view_blob_index_prev = self.view_blob_index
            self.view_mip = False
            self.view_k_index = min(self.view_k_index+1, self.axlim[self.view_ax['k']][1]-1)
            #print('iz=%2i ' % self.view_k_index + '-'*self.view_k_index)
            #print(self.axlim)
            self.clickblob = self.get_current_blob().clone()
            dd = dict(zip(self.clickblob.dims, [0,1,2]))
            self.clickblob.pos[dd[self.view_ax['k']]] = self.view_k_index
            self.view_blob_index = -1
            self.update_plots()

        if event.key() == 16777237:
            # the 'down' key decreases z-plane step -1 (and de-selects current blob)
            if self.view_blob_index != -1:
                self.view_blob_index_prev = self.view_blob_index
            self.view_mip = False
            self.view_k_index = max(self.view_k_index-1, self.axlim[self.view_ax['k']][0])
            #print('iz=%2i ' % self.view_k_index + '-'*self.view_k_index)
            self.clickblob = self.get_current_blob().clone()
            dd = dict(zip(self.clickblob.dims, [0,1,2]))
            self.clickblob.pos[dd[self.view_ax['k']]] = self.view_k_index
            self.view_blob_index = -1
            self.update_plots()


    # def update(self):
    #     """update the model and/or view"""
    #     self.update_plots()
    #     pass

    def start(self):
        """start the event loop, some boilerplate I don't grok yet"""

        # this just keeps rendering all the time, avoids calling update_plots
        #timer = QtCore.QTimer()
        #timer.timeout.connect(self.update_plots)
        #timer.start(50)
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            self.app.exec_()

        #print('TODO: make this return something (all saved blob files? the latest blob file?)')
        #return self.blobs

        return pd.DataFrame(self.get_table()).sort_values(by=self.view_ax['j'])



def make_layers(tr=None, crop_request=None, chan_requests=None, **kwa):
    """create the input layers for the viewer

    input
    ------
    tr : TiffReader (instance) or json file
    crop_request: see example
    han_request: see example

    returns
    ------
    layers (list) : list of DataChunk s

    examples
    ------
    crop_request = dict(Z=(9,37), Y=(19, 130), X=(0,800))
    chan_requests = [dict(C=[3]), dict(C=[0,1,2]), dict(C=[0,1,2,3])]
    """
    if isinstance(tr, TiffReader):
        pass
    elif isinstance(tr, str):
        tr = TiffReader.from_json(tr)

    cr = crop_request if crop_request is not None else {}

    layers = []
    for x in chan_requests:
        lay = dict(
            name=x.get('name', 'xx'),
            chunk=tr.getchunk(req=cr).subchunk(req=x.get('req', None)).squeeze()
        )
        layers.append(lay)

    return layers


if __name__ == '__main__':
    params = dict(
        tr='/home/gbubnis/prj/FOCO_FCD_v0.2/rainbow/tr-rainbow.json',
        dest='scratch',
        crop_request=dict(Z=(9,37), Y=(19, 130), X=(0,800)),
        chan_requests=[
            {'name':'Pan_neur', 'req':dict(C=[3])},
            {'name':'RGBW', 'req':dict(C=[0,1,2,3])},
            ]
    )
    layers = make_layers(**params)
    g = NPEXCurator(layers=layers, blobs=None, dest=params['dest'], windows_mode=False)
    g.start()

