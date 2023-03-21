
import os
import bz2
import json
#import pickle
import itertools
import _pickle as cPickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import napari
from collections import Counter

from .peaks import peakfinder
from .peaks import BlobTemplate
from .alignment import scale_blob_coords
from .alignment import align_data_to_reference
from .gui import NPEXCurator
from .blob import SegmentedBlob
from .data.tiffreader import TiffReader

# def compressed_pickle(title, data):
#     """
#     source: (https://betterprogramming.pub/load-fast-load-big-with-compressed-pickles-5f311584507e)
#     """
#     with bz2.BZ2File(title + '.pbz2', 'w') as f: 
#         cPickle.dump(data, f)

# def decompress_pickle(file):
#     data = bz2.BZ2File(file, 'rb')
#     data = cPickle.load(data)
#     return data

class NPExtractor():
    """NeuroPAL Extractor

    Consolidated NeuroPAL analysis workflow, similar to gcamp-extractor.

    TODO:
    - **data input cases: tiff file, TiffReader, TiffReader(json), info.json, DataManager
    - **switch to `datasource` and `datasource_type`?
    - image vol crop parameter
    - allow for comparing and adding new peakfinding schemes without making a hornets nest
    X- find peaks should return only df_peaks_detected and other (bulky) stuff seperately
    X- lazy reloading (to be able to skip decompressing filtered image layers)
    X- *have the NPExtractor retrieve curated peaks from the GUI

    Attributes:
    -----------
    f: str
    peakfinding_parameters: dict
    tr: TiffReader
    rgbw_channels: list/array
    data_tag: str
    output_folder: str
    df_peaks_detected: DataFrame of detected peaks
    df_peaks_curated: DataFrame of curated peaks
 
    """
    def __init__(self, f, output_folder=None, output_folder_base=None, 
                 peakfinding_parameters=None, pixel_size=None, **kwargs):
        """
        Parameters
        ----------
        f: str
            Image data file (or json pointing to it)
        peakfinding_parameters: dict
            Parameters for peakfinding, these include:
                pad : (array) integer 3-vector of voxel padding that defines a bounding volume around a peak
                median: (int) median filter kernel size
                gaussian: (tuple) it's complicated
                quantile: (float) quantile threshold 0<q<1
                threshold: (float) threshold applied after template filter -1<t<1
                peakfind_channel: (int) optional index of the channel to use for peakfinding, otherwise defaults to rgbw_channels[3]
        output_folder_base: str
            Base name for output folder, the full name is a concatanation of the base plus the `data_tag`
        output_folder: str
            Full output folder path (takes precedence over output_folder_base)
        rgbw_channels: list/array
            4-vector of RGBW channel order
        data_tag: str
            Unique ID for this dataset. Recommended YYYY-MM-DD_wormWW_actAA_extra_stuff_but_not_too_much
        pixel_size: dict
            e.g. {'X':0.12, 'Y':0.12, 'Z':0.75, } sizes in microns

        # TODO: pad is ZYX (matches data) and gaussian is XYZ or YXZ :(
        """
        # peakfinding_params
        self.peakfinding_parameters = dict(
            pad=[2, 6, 6],
            median=5,
            gaussian=(7, 7, 0.45, 0.45, 7, 0.45),
            quantile=0.99,
            threshold=0.75,
            peakfind_channel=None
        )
        if isinstance(peakfinding_parameters, dict):
            self.peakfinding_parameters.update(peakfinding_parameters)

        # connect to image data, different cases depending on file input
        # TODO: be more robust to filenames (filetypes infojson, tiffreaderjson, tiff/tif)
        self.f = f
        filename = os.path.basename(f)
        if filename == 'info.json':
            raise Exception('cannot load from info.json anymore')
        elif filename == 'tiffreader.json':
            # tiffreader.json
            self.tr = TiffReader.from_json(f)
            self.rgbw_channels = kwargs['rgbw_channels']
            self.data_tag = kwargs.get('data_tag', 'default_data_tag')
        elif filename.endswith('tif') or filename.endswith('tiff'):
            # direct tiff load
            self.tr = TiffReader(f)
            self.rgbw_channels = kwargs['rgbw_channels']
            self.data_tag = kwargs.get('data_tag', 'default_data_tag')
        else:
            raise Exception('cannot load file %s' % filename)

        # determine pixel size
        self.pixel_size = pixel_size
        if not pixel_size:
            try:
                self.pixel_size = self.tr.pixel_size
                #print('pxsize from tiff:', self.pixel_size)
            except:
                self.pixel_size = dict(X=-1, Y=-1, Z=-1)
        if -1 in self.pixel_size.values():
            print('WARNING: align_to_ref() will not be possible due to invalid pixel size %s' % (str(self.pixel_size)))

        # setup output_folder
        if output_folder:
            self.output_folder = output_folder
        elif output_folder_base:
            self.output_folder = os.path.join(output_folder_base, 'anl-%s' % (self.data_tag))
        else:
            raise Exception('output_folder or output_folder_base are required')

        # load curated/blobs.csv
        blcsv = os.path.join(self.output_folder, 'curation', 'blobs.csv')
        if os.path.isfile(blcsv):
            self.df_peaks_curated = pd.read_csv(blcsv, index_col=0)
        else:
            self.df_peaks_curated = None

        # load detected peaks
        blcsv = os.path.join(self.output_folder, 'df_peaks_detected.csv')
        if os.path.isfile(blcsv):
            self.df_peaks_detected = pd.read_csv(blcsv, index_col=0)
        else:
            self.df_peaks_detected = None

        print('-------- NPExtractor --------')
        print('loaded  : ', self.f)

    @property
    def json_export_file(self):
        return os.path.abspath(os.path.join(self.output_folder, 'npextractor.json'))

    def export(self):
        """serialize the object"""
        os.makedirs(self.output_folder, exist_ok=True)
        dd = dict(
            f=os.path.abspath(self.f),
            peakfinding_parameters=self.peakfinding_parameters,
            rgbw_channels=self.rgbw_channels,
            data_tag=self.data_tag,
            pixel_size=self.pixel_size
        )
        out = self.json_export_file
        with open(out, 'w') as f:
            json.dump(dd, f, indent=2)
            f.write('\n')
        print('exported: ', out)

    @classmethod
    def load(cls, jf, f=None):
        """re-instantiate object from npextractor.json
        TODO: make tuple of dd['peakfinding_parameters']['gaussian']

        Parameters
        ----------
        jf : str
            json file of a serialized NPExtractor ('npextractor.json')
        f : str
            ```f``` argument for the constructor. Used when raw data is in
            a different location than the one recorded in npextractor.json
        """
        with open(jf) as jfopen:
            dd = json.load(jfopen)
            if f is not None:
                dd['f'] = f
        return cls(output_folder=os.path.abspath(os.path.dirname(jf)), **dd)

    def find_peaks(self):
        """find peaks (neuron centers)

        Specify a channel (indexed according to the associated tiffreader) for peakfinding operations

        Two steps:
            1. quick and dirty peakfinding in order to learn a template
            2. template filtering for more robust peakfinding

        TODO: 
        deprecate SegmentedBlobs in favor of a peaks dataframe?
        allow for specification of channel combinations for peakfinding
        """
        if self.peakfinding_parameters.get('peakfind_channel', None) is None:
            self.peakfinding_parameters['peakfind_channel'] = self.rgbw_channels[3]
        chunk = self.tr.getchunk(req=dict(C=self.peakfinding_parameters['peakfind_channel'])).squeeze()

        p1 = {k:self.peakfinding_parameters[k] for k in ['median', 'gaussian', 'quantile']}
        p2 = {k:self.peakfinding_parameters[k] for k in ['threshold']}
        pad = self.peakfinding_parameters['pad']
        os.makedirs(self.output_folder, exist_ok=True)

        # find peaks (quick and dirty) to learn the template
        res1 = peakfinder(filt='mgt', chunk=chunk, params=p1, pad=pad)
        bt1 = BlobTemplate(chunk=res1['avg3D_chunk'], scale=self.pixel_size, parent=chunk, blobs=res1['blobs'])
        print('made template')

        # find peaks via template match
        p2['template'] = bt1.chunk.data
        res2 = peakfinder(filt='template', chunk=chunk, params=p2, pad=pad)
        bt2 = BlobTemplate(chunk=res2['avg3D_chunk'], scale=self.pixel_size, parent=chunk, blobs=res2['blobs'])
        print('found peaks')

        self.df_peaks_detected = res2['df_peaks']
        self.df_peaks_detected.to_csv(os.path.join(self.output_folder, 'df_peaks_detected.csv'), float_format='%6g')

        ## optional extra output and diagnostics below here

        # plot template infoz and export png
        _, fig = bt2.plot_template_summary()
        fig.suptitle(self.data_tag, fontsize=24)
        fig.tight_layout()
        plt.savefig(os.path.join(self.output_folder, 'plot-template_matching_blobs.png'))

        # export template to csv
        csv = os.path.join(self.output_folder, 'template.csv')
        df = bt2.profiles_1D['df_gauss']
        df['date-worm-act'] = [self.data_tag]*3
        df['pixel_size_um'] = [self.pixel_size[k] for k in df['dimension']]
        df.to_csv(csv, float_format='%6g')

        # extra layers
        crop_request=dict()
        l2 = dict(name='MGQ filtered', chunk=res1['filtered_chunk'].subchunk(req=crop_request).squeeze())
        l3 = dict(name='template filtered', chunk=res2['filtered_chunk'].subchunk(req=crop_request).squeeze())
        self._extra_gui_layers = [l2, l3]

        self.export()

    def extract_rgbw(self, view_napari=False, output_folder=None):
        """
        TODO: segment neurons given peaks (floodfill?)
        TODO: if df_peaks_curated, extract those, otherwise df_peaks
        """
        if self.df_peaks_curated is None:
            return None

        if output_folder is None:
            output_folder = os.path.join(self.output_folder, 'extract_rgbw')
        os.makedirs(output_folder, exist_ok=True)

        # from gcamp_extractor.Extractor import default_quant_function
        # default_quant_function(im, positions, frames)
        # background_subtraction_quant_function(im, positions, frames, quant_radius=3, quant_z_radius=1, quant_voxels=20, background_radius=30, other_pos_radius=None):

        chunk_rgbw = self.tr.getchunk(req=dict(C=self.rgbw_channels)).squeeze()
        quant_kernel = dict(Z=3, X=5, Y=5)
        quant_quantile = 0.9
        background_quantile = 0.6
        kw = {k:v//2 for k,v in quant_kernel.items()}

        chunks, dataframes = [], []
        for ix, row in self.df_peaks_curated.iterrows():
            req = {k:list(range(np.rint(row[k]).astype(int)-kw[k], np.rint(row[k]).astype(int)+kw[k])) for k in ['X', 'Y', 'Z']}
            chunks.append(chunk_rgbw.subchunk(req=req))
            df = pd.DataFrame(list(itertools.product(req['X'], req['Y'], req['Z'])), columns=['X', 'Y', 'Z'])
            df['ix'] = [ix]*len(df)
            df['ID'] = [row.get('ID', '')]*len(df)
            dataframes.append(df)

        # mask incl clashes
        df_mask = pd.concat(dataframes).reset_index(drop=True)
        xyz = [tuple(x) for x in df_mask[['X', 'Y', 'Z']].values]
        clashes = [k for k,v in Counter(xyz).items() if v>1]
        df_mask['clash'] = [x in clashes for x in xyz]

        # compute RGBW values (ignores potential overlap of quantification regions)
        background_rgbw = [np.quantile(chunk_rgbw.subchunk(req=dict(C=i)).squeeze().data, background_quantile) for i in range(4)]
        all_rgbw = [[np.quantile(x.subchunk(req=dict(C=i)).squeeze().data, quant_quantile) for i in range(4)] for x in chunks]
        df_rgbw = (pd.DataFrame(all_rgbw, columns=['R', 'G', 'B', 'W'])-np.asarray(background_rgbw))/background_rgbw

        # TODO warn if any rgbw values < 0 (problem with background subtraction)

        # combine RGBW with the XYZ coords
        df_rgbw_raw = pd.concat([self.df_peaks_curated.reset_index(drop=True), df_rgbw], axis=1, ignore_index=True)
        df_rgbw_raw.columns = list(self.df_peaks_curated.columns)+list(df_rgbw.columns)
        df_rgbw_raw['data_tag'] = [self.data_tag]*len(df_rgbw_raw)

        # color histo rescaling
        df_rgbw_scaled = df_rgbw_raw.copy()
        df_rgbw_scaled['R'] = rescale_channel(df_rgbw_scaled['R'].values)
        df_rgbw_scaled['G'] = rescale_channel(df_rgbw_scaled['G'].values)
        df_rgbw_scaled['B'] = rescale_channel(df_rgbw_scaled['B'].values)
        df_rgbw_scaled['W'] = rescale_channel(df_rgbw_scaled['W'].values)

        # output
        df_rgbw_raw.to_csv(os.path.join(output_folder, 'df_extracted_rgbw_raw.csv'), float_format='%6g')
        df_rgbw_scaled.to_csv(os.path.join(output_folder, 'df_extracted_rgbw_scaled.csv'), float_format='%6g')
        df_mask.to_csv(os.path.join(output_folder, 'df_mask.csv'), float_format='%6g')

        params = dict(
            quant_kernel=quant_kernel,
            quant_quantile=quant_quantile,
            background_quantile=background_quantile
        )

        json_parameter_file = os.path.join(output_folder, 'parameters.json')
        with open(json_parameter_file, 'w') as f:
            json.dump(params, f, indent=2)
            f.write('\n')

        out_dict = dict(
            params=params,
            df_rgbw_raw=df_rgbw_raw,
            df_rgbw_scaled=df_rgbw_scaled,
            df_mask=df_mask
        )

        # PLOT: requires df_rgbw_raw and df_rgbw_scaled
        fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(8, 3)) #, gridspec_kw=gs_kw)
        color_dict = dict(b='#1f77b4', r='#d62728', g='#2ca02c')

        # ax[0], CDF of RGBW intensities
        colors = [color_dict['r'], color_dict['g'], color_dict['b'], 'grey']
        for i, col in enumerate(['R','G', 'B', 'W']):
            ax[0].plot(sorted(df_rgbw_raw[col].values), np.arange(len(df_rgbw_raw)), label=col, color=colors[i])
            ax[1].plot(sorted(df_rgbw_scaled[col].values), np.arange(len(df_rgbw_scaled)), label=col, color=colors[i])
        ax[0].legend()
        ax[0].grid()
        ax[0].set_xlabel('$\Delta F/F_0$')
        ax[1].set_xlabel('scaled')
        ax[1].grid()
        plt.savefig(os.path.join(output_folder, 'plot-intensity-cdf.png'))
        plt.close()

        # inspect (optional)
        if view_napari:
            # make mask and clash layers to view everything in napari
            x_mask = np.zeros(chunk_rgbw.subchunk(req=dict(C=1)).squeeze().reorder_dims(['Z', 'Y', 'X']).data.shape)
            ixx = df_mask[['Z', 'Y', 'X']].values.T
            ixx = tuple([tuple(x) for x in ixx])
            x_mask[ixx] = 1

            x_clash = x_mask*0
            ixx = df_mask[df_mask['clash']][['Z', 'Y', 'X']].values.T
            ixx = tuple([tuple(x) for x in ixx])
            x_clash[ixx] = 1

            v = napari.Viewer()
            arr = chunk_rgbw.reorder_dims(['C', 'Z', 'Y', 'X']).data
            scale = [self.pixel_size[k] for k in ['Z', 'Y', 'X']]
            v.add_image(arr[0], scale=scale, name='red', colormap='red', blending='additive', interpolation='nearest')
            v.add_image(arr[1], scale=scale, name='green', colormap='green', blending='additive', interpolation='nearest')
            v.add_image(arr[2], scale=scale, name='blue', colormap='blue', blending='additive', interpolation='nearest')
            v.add_image(arr[3], scale=scale, name='white', blending='additive', interpolation='nearest')
            v.add_image(x_mask, scale=scale, name='mask', blending='additive', interpolation='nearest', opacity=0.5)
            v.add_image(x_clash, scale=scale, name='clash', colormap='red', blending='additive', interpolation='nearest')
            napari.run()

        return out_dict

    def align_to_ref(self, output_folder=None):
        """"""
        if -1 in self.pixel_size.values():
            print('cannot align_to_ref() due to invalid pixel size %s' % (str(self.pixel_size)))
            return None
        if output_folder is None:
            output_folder = os.path.join(self.output_folder, 'alignment')
        atlas_csv = os.path.join(os.path.dirname(__file__), 'reference_coords/df_reference_coords.csv')
        df_atlas = pd.read_csv(atlas_csv, index_col=0).rename(columns={'names':'ID'})
        df_blobs_ICS = self.df_peaks_curated
        df_blobs_LCS = scale_blob_coords(df_blobs_ICS, s=self.pixel_size)
        result = align_data_to_reference(df_a=df_blobs_LCS, df_b=df_atlas, dest=output_folder)
        return result

    def launch_gui(self, windows_mode=False):
        """make layers and launch gui curator"""
        meta = dict(tag=self.data_tag, voxel_size=self.pixel_size)
        output_folder = os.path.join(self.output_folder, 'curation')

        # make layers (do this earlier?)
        crop_request = dict()
        chunk_w = self.tr.getchunk(req=dict(C=self.rgbw_channels[3], **crop_request)).squeeze()
        chunk_rgbw = self.tr.getchunk(req=dict(C=self.rgbw_channels, **crop_request)).squeeze()
        l0 = dict(name='W (pan neuronal)', chunk=chunk_w)
        l1 = dict(name='RGBW', chunk=chunk_rgbw)
        layers = [l0, l1]
        try:
            layers += self._extra_gui_layers
        except:
            pass

        # checklist
        checklist_json = os.path.join(os.path.dirname(__file__), 'reference_coords/checklist_head.json')
        with open(checklist_json) as jfopen:
            checklist = json.load(jfopen)

        # blobs
        blobs = []
        if self.df_peaks_detected is not None:
            for ix, row in self.df_peaks_detected.iterrows():
                sb = SegmentedBlob(
                    index=ix,
                    pos=[row['X'], row['Y'], row['Z']],
                    dims=['X', 'Y', 'Z'],
                    pad=self.peakfinding_parameters['pad'],
                    prov='detected'
                    )
                blobs.append(sb)

        gui = NPEXCurator(
            layers=layers,
            blobs=blobs,
            dest=output_folder,
            meta=meta,
            windows_mode=windows_mode,
            checklist=checklist,
        )
        blobs_curated = gui.start()
        self.df_peaks_curated = blobs_curated


def rescale_channel(x, rms_scale=5):
    """approximate histogram equalization

    rescale so the RMS=rms_scale and then (natural) log scale values >1
    """
    x_rms = np.sqrt(np.mean(x**2))
    x_new = (x/x_rms)*rms_scale
    xg1 = np.where(x_new>1)
    x_new[xg1] = 1+np.log(x_new[xg1])
    return x_new