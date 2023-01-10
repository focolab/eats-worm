#!/usr/bin/env python3

from improc.segfunctions import medFilter2d
import json
import numpy as np
import pyqtgraph as pg
import os
from cv2 import cvtColor, normalize, resize, VideoWriter, VideoWriter_fourcc
from eats_worm import *

def get_neuron_mips(extractor, indices, window_size=60, zoom=1, quant_z_radius=1):
    im_shape = extractor.im.get_t(0).shape
    mip_shape = (window_size, window_size)
    mips = np.zeros((len(indices), extractor.spool.t, window_size * zoom, window_size * zoom), dtype=np.uint16)
    half_window = window_size // 2
    im = None
    subim  = None
    filtered = None
    mip = None
    highlights = np.zeros(mips.shape, dtype=np.uint8)
    quantified_voxels = None
    if hasattr(extractor, 'curator_layers') and 'quantified_roi_voxels' in extractor.curator_layers:
        quantified_voxels = extractor.curator_layers['quantified_roi_voxels']['data']

    for t in range(extractor.spool.t):
        im = extractor.im.get_t(t)
        for index in indices:
            position = extractor.spool.threads[index].get_position_t(t).tolist()
            min_z = np.rint(max(0, position[0] - quant_z_radius)).astype(int)
            max_z = np.rint(min(len(extractor.frames) - 1, position[0] + quant_z_radius)).astype(int)
            min_x = np.rint(max(0, position[1] - half_window)).astype(int)
            max_x = np.rint(min(im_shape[1], position[1] + half_window)).astype(int)
            min_y = np.rint(max(0, position[2] - half_window)).astype(int)
            max_y = np.rint(min(im_shape[2], position[2] + half_window)).astype(int)
            quant_start = (np.rint(position[1]).astype(int) - min_x - 3, np.rint(position[2]).astype(int) - min_y - 3)
            quant_end = (np.rint(position[1]).astype(int) - min_x + 3, np.rint(position[2]).astype(int) - min_y + 3)
            subim = im[min_z:max_z + 1, min_x:max_x, min_y:max_y]
            filtered = medFilter2d(subim, 3)
            mip = np.max(filtered, axis=0)
            if mip.shape != mip_shape:
                corrected = np.zeros(mip_shape, dtype=np.uint16)
                corrected[:mip.shape[0], :mip.shape[1]] = mip
                x_roll = half_window - np.rint(position[1] - min_x).astype(int)
                y_roll = half_window - np.rint(position[2] - min_y).astype(int)
                corrected = np.roll(corrected, (x_roll, y_roll), axis=(0, 1))
                mip = corrected
            if zoom != 1:
                mip = cv2.resize(mip, tuple(np.multiply(zoom, mip.shape)))
            if quantified_voxels is not None:
                voxels = quantified_voxels[index][t]
                highlight = np.zeros((window_size, window_size), dtype=np.uint8)
                for voxel in voxels:
                    z, x, y = voxel
                    voxel -= np.array([min_z, min_x, min_y])
                    z, x, y = voxel
                    highlight[x, y] = 60
                if zoom != 1:
                    highlight = cv2.resize(highlight, tuple(np.multiply(zoom, highlight.shape)))
                highlights[indices.index(index), t] = highlight
            mips[indices.index(index), t] = mip
    
    colored_mips = np.zeros(mips.shape + (3,), dtype=np.uint8)
    for index in range(mips.shape[0]):
        frames = mips[index]
        num_frames, x_shape, y_shape = frames.shape
        frames = frames.reshape((num_frames * x_shape, y_shape))
        normalized = cv2.normalize(frames, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        colored = cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)
        colored = colored.reshape((num_frames, x_shape, y_shape, 3))
        if quantified_voxels is not None:
            colored[:,:,:,2][highlights[index] != 0] = np.iinfo(colored.dtype).max
        for t in range(num_frames):
            colored[t] = cv2.putText(colored[t], str(indices[index]), (3, 13), cv2.FONT_HERSHEY_SIMPLEX, .35, (0, 255, 255))
        colored_mips[index] = colored
    return colored_mips

def write_roi_mip_montage_video(extractor, output_path, video_label=None, window_size=20, zoom=4, curation_filter='not trashed', quant_z_radius=1):
    skip = None
    if curation_filter != 'all':
        try:
            with open(os.path.join(extractor.output_dir, 'curate.json')) as f:
                curated_json = json.load(f)
                if curation_filter == 'kept':
                    skip = [roi for roi in curated_json.keys() if curated_json[roi] != 'keep']
                elif curation_filter == 'not trashed':
                    skip = [roi for roi in curated_json.keys() if curated_json[roi] == 'trash']
        except:
            print("No curate.json found. Falling back to curation_filter='all'.")
    
    output_x, output_y = 1920, 1080
    frames = np.zeros((extractor.spool.t, 1080, 1920, 3), dtype=np.uint8)
    app = pg.mkQApp()
    draw_index = 0
    neuron_mips = get_neuron_mips(extractor, [index for index in range(len(extractor.spool.threads)) if not skip or str(index) not in skip], window_size=window_size, zoom=zoom, quant_z_radius=quant_z_radius)
    mip_size = window_size * zoom
    mips_per_line = output_x // mip_size
    x_remainder = output_x % mip_size
    for neuron_mip in neuron_mips:
        x_start = x_remainder // 2 + draw_index % mips_per_line * mip_size
        y_start = draw_index // mips_per_line * mip_size
        if x_start >= output_x or y_start >= output_y:
            print("ran out of space for index ", draw_index)
        else:
            frames[:, y_start:y_start+mip_size, x_start:x_start+mip_size, :] = neuron_mip
        draw_index += 1
    if video_label is not None:
        for i in range(len(frames)):
            frames[i] = cv2.putText(frames[i], '{}, T={}'.format(video_label, i), (3, 1067), cv2.FONT_HERSHEY_SIMPLEX, .35, (0, 255, 255))
    output_path_dir = os.path.dirname(output_path)
    if not os.path.exists(output_path_dir):
        os.makedirs(output_path_dir)
    video_writer = VideoWriter(output_path, VideoWriter_fourcc(*'mp4v'), 30, (frames[0].shape[1], frames[0].shape[0]))
    for frame in frames:
        video_writer.write(frame)
    video_writer.release()
    print("Wrote video to {}.".format(output_path))
