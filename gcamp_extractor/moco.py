import numpy as np
import os
from improc.segfunctions import medFilter2d, gaussian2d
import imreg_dft as ird

import napari

from skimage.registration import phase_cross_correlation




def compute_moco_offsets(mft=None, t_max=-1, suppress_output=True, 
    filter_params=None, method='ird', ird_filter_pcorr=0, upsample_factor=0):
    """2D motion correction

    Parameters:
    -----------
    mft : (MultiFileTiff)
    t_max : (int) Only evaluate the first `t_max` timepoints
    suppress_output : (Bool)
    filter_params : (dict) passed to get_moco_im()
    method : (str) 'ird' or 'skimage
    ird_filter_pcorr : (int) only applied for 'ird' case
    upsample_factor : (int) only applied for 'skimage' case

    Returns:
    -------
    offsets : (list) List of Z,Y,X offsets (Z always zero)
    """

    defaults = dict(mode_2D='mid', median=0, gaussian=0)
    if isinstance(filter_params, dict):
        defaults.update(filter_params)
    else:
        filter_params = defaults
    upsample_factor = 0
    if t_max == -1:
        t_max = 1000000

    previous_frame = get_moco_im(mft=mft, t=0, **filter_params)
    offsets = [np.asarray([0, 0, 0])]
    for i in range(1, t_max):
        try:
            this_frame = get_moco_im(mft=mft, t=i, **filter_params)
        except:
            break
        if method == 'ird':
            result = ird.translation(previous_frame, this_frame, filter_pcorr=ird_filter_pcorr)
            offsets.append(np.hstack([[0], result['tvec']]))
        elif method == 'skimage':
            result = phase_cross_correlation(previous_frame, this_frame, upsample_factor=upsample_factor)
            offsets.append(np.hstack([[0], *result[0]]))
        previous_frame = this_frame
        if not suppress_output:
            print('offset [%3i]: [%8.3f %8.3f %8.3f]' % (i, *offsets[-1]))
    return np.array(offsets)

def get_moco_im(mft, t, mode_2D='mid', median=0, gaussian=0):
    """helper to fetch and filter an image for moco comparison

    A volume is "compressed" to 2D either by taking the middle z-plane
    (`mode='mid'`) or taking a MIP (`mode='mip'`). Then, 2D median and gaussian
    filtering (both optional) are applied.

    parameters
    ----------
    mft: (MultiFileTiff)
    t: (int) timepoint
    mode_2D: (str) 'mid' (default) or 'mip' to use middle z-plane or MIP
    median: (int) 2D median filter width (odd)
    gaussian: (float) standard deviation, in pixels, for 2D gaussian blur

    Returns:
    --------
    im (2D ndarray): image
    """
    ix_mid = mft.frames[int(len(mft.frames)/2)]
    if mode_2D == 'mip':
        im = np.amax(mft.get_t(t), axis=0)
    else:
        im = mft.get_tbyf(t, ix_mid)

    if median > 0:
        im = medFilter2d(im, median)
    if gaussian > 0:
        im = gaussian2d(im, (gaussian, np.ceil(gaussian*2+1))) # sigma and kernel size

    return im


def view_moco_offsets(mft, offsets):
    """view raw MIP and MOCO MIP in napari"""
    pad = 1

    #df_offsets = pd.DataFrame(np.cumsum(offsets, axis=0), columns=['Z', 'Y', 'X'])

    numt = len(offsets)
    di, dj = mft.sizexy
    offsets_cum = np.vstack(([0, 0, 0], np.cumsum(offsets, axis=0)))

    tracks = []
    full_stack = np.zeros(shape=(numt, di*2+pad, dj))
    for i in range(numt):
        i_shift = np.rint(offsets_cum[i][1]).astype(int)
        j_shift = np.rint(offsets_cum[i][2]).astype(int)
        #print('moco %i: %i %i' % (i, i_shift, j_shift))

        this_im = np.zeros(shape=(di*2+pad, dj))+1000
        this_im[0:di, 0:dj] = np.amax(mft.get_t(i), axis=0)
        this_im[di+pad:, 0:dj] = np.roll(np.roll(this_im[0:di, 0:dj]*1, i_shift, axis=0), j_shift, axis=1)
        full_stack[i] = this_im

        tracks.append([0, i, di//2-i_shift, dj//2-j_shift])
        tracks.append([1, i, di//2, dj//2])

    v = napari.Viewer()
    v_im = v.add_image(full_stack, name='moco')
    v_tr = v.add_tracks(tracks,  name='track')
    napari.run()

# def plot_moco_offsets(self):
#     df = self.spool.to_dataframe(dims=self.dims)
#     # ip = df['prov']
#     print(df.head(30))
#     print(self.df_moco_offsets.head(30))
#     import matplotlib
#     import matplotlib.pyplot as plt
#     pp = self.df_moco_offsets.plot(figsize=(8, 4))
#     plt.suptitle(self.im.root)
#     plt.xlabel('volume, T')
#     plt.ylabel('pixel offset')
#     plt.show()