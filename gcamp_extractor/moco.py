import os
import numpy as np
import imreg_dft as ird
import cv2
import napari
from skimage.registration import phase_cross_correlation

def compute_moco_offset(frame1, frame2, median=0, gaussian=0, method='ird', ird_filter_pcorr=0, upsample_factor=0):
    """filter and then compute offset between a pair of (2D) images 

    Parameters
    ----------
    median: (int) 2D median filter width (odd)
    gaussian: (float) standard deviation, in pixels, for 2D gaussian blur
    method : (str) 'ird' or 'skimage
    ird_filter_pcorr : (int) only applied for 'ird' case
    upsample_factor : (int) only applied for 'skimage' case
    """
    # using cv2 functions directly (improc functions do not handle 2D)
    if median > 0:
        frame1 = cv2.medianBlur(frame1, median)
        frame2 = cv2.medianBlur(frame2, median)
    if gaussian > 0:
        kw = np.ceil(gaussian*6+1).astype(int)  # kernel width
        frame1 = cv2.GaussianBlur(frame1, (kw, kw), gaussian)
        frame2 = cv2.GaussianBlur(frame2, (kw, kw), gaussian)
    if method == 'ird':
        result = ird.translation(frame1, frame2, filter_pcorr=ird_filter_pcorr)
        offset = np.hstack([[0], result['tvec']])
    elif method == 'skimage':
        result = phase_cross_correlation(frame1, frame2, upsample_factor=upsample_factor)
        offset = np.hstack([[0], *result[0]])
    return offset

def compute_moco_offsets(mft=None, t_max=-1, suppress_output=True, mode_2D='mid',
    filter_params=None, method='ird', ird_filter_pcorr=0, upsample_factor=0):
    """2D motion correction for a whole recording

    Parameters:
    -----------
    mft : (MultiFileTiff)
    t_max : (int) Only evaluate the first `t_max` timepoints
    suppress_output : (Bool)
    mode_2D: (str) 'mid' or 'mip' for midplane or MIP, respectively
    filter_params : (dict) passed to get_moco_im()
    method : (str) 'ird' or 'skimage'
    ird_filter_pcorr : (int) only applied for 'ird' case
    upsample_factor : (int) only applied for 'skimage' case

    Returns:
    -------
    offsets : (list) List of Z,Y,X offsets (Z always zero)
    """
    defaults = dict(median=0, gaussian=0)
    if isinstance(filter_params, dict):
        defaults.update(filter_params)
    else:
        filter_params = defaults
    if t_max == -1:
        t_max = 1000000
    ix_mid = mft.frames[int(len(mft.frames)/2)]

    # first frame
    if mode_2D == 'mip':
        previous_frame = np.amax(mft.get_t(0), axis=0)
    else:
        previous_frame = mft.get_tbyf(0, ix_mid)

    offsets = [np.asarray([0, 0, 0])]
    for i in range(1, t_max):
        try:
            if mode_2D == 'mip':
                this_frame = np.amax(mft.get_t(i), axis=0)
            else:
                this_frame = mft.get_tbyf(i, ix_mid)
        except:
            break
        
        dxdy = compute_moco_offset(previous_frame, this_frame, **filter_params,
            method=method,  ird_filter_pcorr=ird_filter_pcorr, upsample_factor=upsample_factor)
        offsets.append(dxdy)
        previous_frame = this_frame
        if not suppress_output:
            print('offset [%3i]: [%8.3f %8.3f %8.3f]' % (i, *offsets[-1]))
    return np.array(offsets)

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
    v_im = v.add_image(full_stack, name='moco', gamma=0.5)
    v_tr = v.add_tracks(tracks,  name='track')
    napari.run()
