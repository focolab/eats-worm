# peakfinding and filtering schemes

import pdb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
from improc.segfunctions import gaussian3d, get_bounded_chunks, medFilter2d, findpeaks3d_26nn, template_filter
from .data.datachunk import DataChunk
from .blob import SegmentedBlob

def tile_images(images=None, numcol=14):
    """tile a list of 2D images"""
    numrow = np.ceil(len(images)/numcol).astype(int)

    pxh, pxw = images[0].shape
    #print('num chunk:', len(images))
    #print('numcol   :', numcol)
    #print('numrow   :', numrow)
    #print('pxh pxw  :', pxh, pxw)

    tiled_image = np.zeros((pxh*numrow, pxw*numcol))
    for irow in range(numrow):
        ia = pxh*irow
        ib = ia+pxh
        for jcol in range(numcol):
            ja = pxw*jcol
            jb = ja+pxw
            ix = irow*numcol+jcol
            if ix<len(images):
                tiled_image[ia:ib, ja:jb] = images[ix]
            else:
                break
    return tiled_image

def filter_qmg(im1, quantile=None, median=None, gaussian=None):
    """Filtering. quantile then median then gaussian (qmg)

    NOTE: im1 is assumed to have ZYX dimension order, but the gaussian filter
    dimensions are XYZ (or YXZ) or something else..

    For peak finding, follow this up with findpeaks3d (from segfunctions)

    input
    ------
    im1: (np.ndarray) ZYX dimension order
    quantile: (float in [0,1])
    median: (integer)
    gaussian: (??)
    """
    im1 = im1*1
    if quantile is not None:
        im1 = np.array(im1*np.array(im1>np.quantile(im1, quantile)))
    if median is not None:
        im1 = medFilter2d(im1, median)
    if gaussian is not None:
        im1 = gaussian3d(im1, gaussian)
    return im1

def filter_mgt(im1, median=None, gaussian=None, quantile=None):
    """GCE filter scheme: Median->Gaussian->Threshold

    NOTE: im1 is assumed to have ZYX dimension order, but the gaussian filter
    dimensions are XYZ (or YXZ) or something else..

    For peak finding, follow this up with findpeaks3d (from segfunctions)

    input
    ------
    im1: (np.ndarray) ZYX dimension order
    gaussian: (??)
    quantile: (float in [0,1])
    median: (integer) 
    """
    #### copypasta from Extractor.calc_blob_threads()
    # im1 = self.im.get_t()
    # im1 = medFilter2d(im1)
    # im1 = gaussian3d(im1,self.gaussian)
    # peaks = findpeaks3d(np.array(im1 * np.array(im1 > np.quantile(im1,self.quantile))))

    #### copypasta from FilterSweeper
    # blurred = gaussian3d(copy.deepcopy(stack), (self.width_x_val, self.width_y_val, self.sigma_x, self.sigma_y, self.width_z_val, self.sigma_z))
    # filtered = medFilter2d(blurred, self.median_sizes[self.median_index])
    # thresholded = filtered > np.quantile(layer.data, self.quantile)

    im1 = im1*1
    if median is not None:
        im1 = medFilter2d(im1, median)
    if gaussian is not None:
        im1 = gaussian3d(im1, gaussian)
    if quantile is not None:
        im1 = np.array(im1*np.array(im1>np.quantile(im1, quantile)))
    return im1

def peakfinder(filt=None, chunk=None, params=None, pad=None):
    """Do filtering and peakfinding, then some extra useful things

    Parameters
    ----------
    filt : str
        Which filter scheme to use.
        'mgt' -> mean, gaussian, threshold original GCE scheme
        'template' -> template filtering
    chunk : DataChunk
    params : dict
        These get passed to pf
    pad : list/array
        Defines bounding box around a peak by padding voxels to both sides

    Returns
    -------
    out : dict
        May be bloated, but holds extra info useful for diagnostics

    """
    if filt == 'mgt':
        filtered = filter_mgt(chunk.data, **params)
    elif filt == 'template':
        filtered = template_filter(chunk.data, **params)


    peaks = findpeaks3d_26nn(filtered)
    chunks = get_bounded_chunks(data=chunk.data, peaks=peaks, pad=pad)
    # cull out peaks for which the bounding box goes beyond data bounds
    peaks_inbounds, blobs, = [], []
    for i, x in enumerate(peaks):
        blob = SegmentedBlob(index=i, pos=x, dims=chunk.dims, pad=pad, prov='detected')
        try:
            peaks_inbounds.append(x)
            blobs.append(blob)
        except:
            pass

    # helpful for diagnostics
    filtered_chunk = DataChunk(data=filtered, dims=chunk.dims)
    avg3D_chunk = DataChunk(np.mean([x for x in chunks], axis=0), dims=chunk.dims)
    med3D_chunk = DataChunk(np.median([x for x in chunks], axis=0), dims=chunk.dims)

    df_peaks = pd.DataFrame([b.posd for b in blobs])
    df_peaks['prov'] = [b.prov for b in blobs]
    df_peaks['status'] = [0]*len(df_peaks)
    df_peaks['ID'] = ['']*len(df_peaks)
    #### OUTPUT
    out = dict(
        num_peaks=len(peaks),
        peaks=np.asarray(peaks_inbounds),
        df_peaks=df_peaks,
        filtered_chunk=filtered_chunk,
        blobs=blobs,
        input_chunk=chunk,
        avg3D_chunk=avg3D_chunk,
        med3D_chunk=med3D_chunk,
    )
    return out

class BlobTemplate(object):
    """3D blob template

    Class for a "learned" 3D blob template, should be approximately Gaussian
    and isotropic.
    - holds the chunk of data
    - handles anisotropic voxel size/spacing
    - computes intensity profiles and Gaussian fits
    - plots a bunch of visualizations
    """
    def __init__(self, chunk=None, scale=None, parent=None, blobs=None):
        """
        Parameters
        ----------
        chunk: (DataChunk) 3D DataChunk of the template itself
        scale: (dict) voxel size/spacing for each dimension (in micron)
        parent: (DataChunk) Parent DataChunk from which template was extracted
        blobs: (list) List of SegmentedBlobs (used to derive the template)
        """

        if 'C' in chunk.dims:
            raise NotImplementedError('Color templates not yet implemented')

        self.parent = parent
        self.blobs = blobs
        
        self.chunk = chunk
        if scale is None:
            scale = {d:1 for d in chunk.dims}
        self.scale = scale
        
        # compute some helpful things
        self.chunk_center = {}
        for dim, sz in chunk.dim_len.items():
            self.chunk_center[dim] = np.floor(sz/2).astype(int)

        self.profiles_1D = self.get_1D_profiles()

    def get_1D_center_profile(self, dim):
        """1D intensity profile that skewers the chunk center"""
        req = {k:v for k, v in self.chunk_center.items()}
        i = req.pop(dim)
        return self.chunk.subchunk(req=req).squeeze().data

    def get_1D_profiles(self):
        """Get 1D intensity profiles through center and Gaussian fits
        """
        chunk = self.chunk
        scale = self.scale

        profiles = {}
        fits = []
        for dim in chunk.dims:
            x_px = np.arange(chunk.dim_len[dim])-self.chunk_center[dim]
            x_um = x_px*scale[dim]
            val = self.get_1D_center_profile(dim=dim)
            profiles[dim] = [x_px, x_um, val]

            # gaussian fit (in pixel and micron units)
            # special case zero padding
            if len(x_px) == 1:
                x_px = np.asarray([x_px[0]]*3)
                val = np.asarray([val]*3)

            popt = self._gauss_fitter_1D(x_px, val)
            cols = ['amplitude', 'mean_px', 'std_px']
            fd = dict(zip(cols, popt))
            fd['mean_um'] = fd['mean_px']*scale[dim] 
            fd['std_um'] = fd['std_px']*scale[dim]
            fits.append(fd)

        df_gauss = pd.DataFrame(fits)
        df_gauss['dimension'] = chunk.dims 
        newcols = ['dimension']+cols+['mean_um', 'std_um']
        df_gauss = df_gauss[newcols]

        out = dict(
            profiles=profiles,
            df_gauss=df_gauss,
        )

        return out

    def about(self):
        """"""
        print('-------- BlobTemplate.about() --------')
        print('#### DataChunk')
        self.chunk.about()
        print('#### 1D intensity cuts:')
        print(self.profiles_1D['df_gauss'])

    def plot_template_summary(self):
        """Summary figure combining other plot methods
        
        includes:
        - tiled montage of all detected blobs (2D MIPs)
        - 1D intensity profiles
        - 2D intensity slices
        """
        fig = plt.figure(figsize=(12, 14))
        axtopA = plt.subplot(3,3,(1,3))
        axtopB = plt.subplot(3,3,(4,6))
        ax0 = plt.subplot(3,3,7)
        ax1 = plt.subplot(3,3,8)
        ax2 = plt.subplot(3,3,9)

        self.plot_blob_montage(ax=axtopA, mip='Z', numcol=20)
        self.plot_mip(ax=axtopB, mip='Z')
        self.plot_template_profiles_1D(ax=ax0, units='px')
        self.plot_template_profiles_1D(ax=ax1, units='um')
        self.plot_template_slices_2D(ax=ax2)

        plt.tight_layout()
        return [ax0, ax1, ax2], fig
    
    def plot_template_slices_2D(self, ax=None):
        """plot 2D slices, all passing through the center

        TODO: Factor out to wbu-viz? Take x, y, z as args? MIP option?
        """
        ijkdim = ['X', 'Y', 'Z']

        chk = self.chunk.reorder_dims(ijkdim).data
        LI, LJ, LK = chk.shape
        ctri, ctrj, ctrk = np.floor(np.asarray(chk.shape)/2).astype(int)

        chIJ = np.atleast_2d(chk[:,:,ctrk].squeeze())
        chKJ = np.atleast_2d(chk[ctri,:,:].squeeze())
        chIK = np.atleast_2d(chk[:,ctrj,:].squeeze())

        im = np.zeros(shape=(LI+LK, LJ+LK))
        im[0:LI, 0:LJ] = chIJ
        im[LI:LI+LK, 0:LJ] = chKJ.T
        im[0:LI, LJ:LJ+LK] = chIK
        
        ax.axhline(LI-0.5, color='black')
        ax.axvline(LJ-0.5, color='black')
        
        fmt = dict(
            fontsize='x-large',
            fontproperties=dict(weight='bold'),
            color='red',
            ha='center',
            va='center',
        )
        ax.text(0, ctri, ijkdim[0], **fmt)
        ax.text(0, LI+ctrk, ijkdim[2], **fmt)
        ax.text(ctrj, 0, ijkdim[1], **fmt)
        ax.text(LJ+ctrk, 0, ijkdim[2], **fmt)
        
        ax.imshow(im, cmap='bone')

    def plot_template_profiles_1D(self, ax=None, units='um'):
        """plot 1D profiles and gaussian fits (through blob center)"""
        info = self.profiles_1D
        
        if units in ['um', 'micron', 'microns']:
            units = 'um'
        elif units in ['px', 'pixels']:
            units = 'px'

        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        
        if units == 'um':
            xdom = np.linspace(-3, 3, 50)
        elif units == 'px':
            xdom = np.linspace(-10, 10, 50)

        # compute/plot the dataset's intensity histogram
        template_max = np.max(self.chunk.data.ravel())
        dclip = np.clip(self.parent.data.ravel(), None, template_max*1.05)
        hh, ee = np.histogram(dclip, bins=150)
        hh = hh/np.max(hh)*max(xdom)*0.6
        cc = (ee[1:]+ee[:-1])/2

        # plot greycount histogram
        ax.axvline(0, color='grey', linewidth=1)
        ax.axhline(0, color='grey', linewidth=1)
        ax.plot(hh, cc, color='salmon', label='histo', lw=2)
        
        for i, (k,v) in enumerate(info['profiles'].items()):
            color = colors[i]
            
            # plot the gaussian fit
            dfg = info['df_gauss']
            row = dfg[dfg['dimension'] == k]
            if units == 'um':
                popt = row[['amplitude', 'mean_um', 'std_um']].values.squeeze()
            elif units == 'px':
                popt = row[['amplitude', 'mean_px', 'std_px']].values.squeeze()
            ax.plot(xdom, self._gaussian(xdom, *popt), '-', color=color, lw=2)

            # plot the data
            if units == 'um':
                xval = v[1]
            elif units == 'px':
                xval = v[0]
            ax.plot(xval, v[2], 'o', ms=5, label=k, color=color, mec='grey')


        # make it fancy
        ax.set_ylim([-template_max*0.15, template_max*1.05 ])
        ax.legend(ncol=4, loc=8)
        ax.grid()
        if units == 'um':
            ax.set_xlabel('distance from center [um]')
        elif units == 'px':
            ax.set_xlabel('distance from center [pixels]')
        ax.set_ylabel('intensity')
        ax.set_title('Intensity profiles, Gaussian fits')

    def plot_blob_montage(self, ax=None, mip='Z', numcol=20):
        """2D tiled montage of extracted blob MIPs"""
        chunk = self.parent
        blobs = self.blobs

        if ax is None:
            fig = plt.figure(figsize=(12, 6))
            ax = plt.subplot(111)

        bboxes = [chunk.subchunk(req=b.chreq()) for b in blobs]
        ims = [c.max_ip(mip).data for c in bboxes]
        montage = tile_images(images=ims, numcol=numcol)
        ax.imshow(np.ma.log(montage), cmap='bone', interpolation='none')  # https://stackoverflow.com/a/21753270
        ax.set_title('all blobs (N=%i)' % len(bboxes))
        return ax

    def plot_mip(self, ax=None, mip='Z', xydims=['Y', 'X'], mkwa=None):
        """plots mip with peaks superimposed"""
        chunk = self.parent

        if ax is None:
            fig = plt.figure(figsize=(16, 12))
            ax = plt.gca()

        ## orient MIP in landscape mode
        xydims = ['X', 'Y']
        if chunk.dim_len['X'] > chunk.dim_len['Y']:
            xydims = ['Y', 'X']

        #xydims = ['Y', 'X']
        im = chunk.max_ip(mip).squeeze().reorder_dims(xydims).data
        _ = ax.imshow(np.ma.log(im), cmap='bone')

        txt = 'num_found : %i\n' % len(self.blobs)
        xx = [b.posd[xydims[1]] for b in self.blobs]
        yy = [b.posd[xydims[0]] for b in self.blobs]

        ax.set_xlabel(xydims[0])
        ax.set_ylabel(xydims[1])

        mk = dict(marker='x', s=20, color='r')
        if mkwa is not None:
            mk.update(mkwa)
        #_ = ax.scatter(xx, yy, marker='x', s=50, color='r')
        #_ = ax.scatter(xx, yy, marker='x', s=20, color='r')
        _ = ax.scatter(xx, yy, **mk)
        _ = ax.text(4, 4, txt, fontfamily='monospace', color='w', ha='left', va='top', fontsize='large')

    def _gaussian(self, x, amplitude, mean, stddev):
        return amplitude*np.exp(-0.5*((x-mean)/stddev)**2)

    def _gauss_fitter_1D(self, x, data):
        """returns: amplitude, mean, std"""
        popt, _ = optimize.curve_fit(self._gaussian, x, data)
        popt[2] = np.abs(popt[2])
        return popt



if __name__ == "__main__":
    pass