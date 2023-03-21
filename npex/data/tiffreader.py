import itertools
import pdb
import json
import os
import time as time

import xmltodict
from tifffile import TiffFile
from tifffile import TiffPage
import numpy as np
import pandas as pd

from .datachunk import DataChunk, chunk_ix

class TiffReader(object):
    """lazy reader, pulls arbitrary TZCYX chunks from a tiff hyperstack

    Tiff data reader that serves up arbitrary TZCXY data chunks

    attributes
    ------
    files: (list) tiff file paths(s)
    axes: (str) 'TZYX' or whatever
    shape: (tuple) hyperstack dimensions (200, 10, 600, 800) 
    offset: (int) inital frames that are clipped off (not part of hyperstack)
    TF: (list) instantiated tifffile objects, one per file
    meta_extra: (dict) extra useful information, including the frame index
    meta: (dict) associated metadata (e.g. voxel size, expt date) should be
        json-izable. Don't abuse this :)
    page_byte_offsets: (list of lists) byte offsets for tiff pages. One list per
        file. These offsets enable direct page reads sans file traversal

    methods
    ------
    about(): print some useful info
    from_json(): alt constructor
    to_json(): serialize to json file
    dump_metadata(): (under construction) should be 'to_json()' 
    infer_params(): infers tiff attributes (axes/shape) from tiff metadata*
    build_index(): build an index hash, used to pull frames from >=1 tiff file
    getframe():
    getchunk():
    """

    def __init__(self, f=None, numZ=-1, numC=-1, numT=-1, offset=0, 
                shape=None, axes=None, meta=None, timing=False, pixel_size=None,
                wallclock=None, page_byte_offsets=None):
        """

        input
        ------
        f: (str or list of str) file or list of files
        numZ: (int) number of Z planes
        numC: (int) number of color channels
        numT: (int) number of time points
        offset: (int) number of frames clip off the start of the recording
        shape: (array) data dimensions (for all files combined)
        axes: (str) TZCYX TCZYX etc
        meta: (dict) metadata, unregulated but should be json-friendly
        pixel_size: (dict) e.g. {'X':1, 'Y':1, 'Z':0.7}
        wallclock: (array) array of page (i.e. frame) timestamps
        page_byte_offsets: (list of lists) byte offsets for tiff pages
        DEPRECATED file_pages: (list) number of tiff pages in each tiff file

        TiffReader instantiation requires building an index of all pages (2D
        images). If insufficient information is provided to the constructor, it
        can be slow or impossible to build the correct index solely by parsing
        the tiff files themselves.

        On a first encounter with a tiff file, deep page fetching is
        horrendously slow because pages can vary in size and the page byte
        offsets are not declared up front (page = metadata + image). As a
        result, requesting page 500 requires a scan through the first 500
        pages prior to reading the data. Although smart caching (and
        potentialy OS file-read caching) can accelerate access to previously
        scanned pages, this scheme is not feasible for TB/PB files and any
        cached information will be lost between instances/sessions.

        As an example of this slowness, using an external 7200 rpm drive
        connected by USB3, it takes about 140s to reach page 60000 of a 35 GB
        timelapse recording. For larger files it will only get worse.

        An "honorable mention" bottleneck is to parse the OME xml block
        associated with each page. The OME parse is used to get wall clock time
        information (potentially also avaialbe from micromanager generated json
        metadata files).

        TiffReader aims to vanquish both of these slow steps by only doing them
        once and saving the required metadata for re-use.

        The recommended workflow is to ingest a big recording by:
            1. instantiating a TiffReader
            2. serializing it (TiffReader.to_json()), preserving slow-to-parse
                information (expecially the page byte offsets)
        Subsequent instantiations from the json file (TiffReader.from_json()),
        will have direct page access without the slow file parsing.
        """
        t00 = time.time()
        # instantiate tifffiles for each tiff file
        if isinstance(f, list):
            self.files = f
        elif isinstance(f, str):
            if f.endswith('tiff') or f.endswith('tif'):
                self.files = [f]
            elif f.endswith('json'):
                print('use alt constructor TiffReader.from_json()')
                raise Exception()
            else:
                raise Exception('TiffReader argument \'f\' must be a .tiff or .tif')
        else:
            raise Exception('f input is neither a list nor str')
        self.TF = [TiffFile(f) for f in self.files]

        t01a = time.time()
        if meta is None:
            self.meta = {}
        else:
            self.meta = meta

        if pixel_size is None:
            pixel_size = {}

        # the quick and easy ones
        self.dtype = self.TF[0].pages[0].dtype
        self.offset = offset

        # get and store page offsets
        t01b = time.time()
        if page_byte_offsets:
            self.page_byte_offsets = page_byte_offsets
        else:
            self.page_byte_offsets = [[p.offset for p in x.pages] for x in self.TF]
        self.file_pages = [len(x) for x in self.page_byte_offsets]

        # wallclock times for frames (may/may not be in tiff metadata)
        t02 = time.time()
        if wallclock is None:
            self.params_inferred = self.infer_params()
            try:
                self.wallclock = list(self.params_inferred['df_index']['time'].values)
            except:
                self.wallclock = None
        else:
            self.wallclock = wallclock

        # CONDITIONAL HELL to determine axes and shape
        # either infered from tiff metadata or via manual input (numC, numT)
        t03 = time.time()
        if axes and shape:
            # this is the best case (EXPLICIT)
            self.axes = axes
            self.shape = shape
        elif numZ == -1 and numC == -1 and offset == 0:
            # use the inferred values and hope for the best
            self.params_inferred = self.infer_params()
            self.axes = self.params_inferred['axes']
            self.shape = self.params_inferred['shape']
        else:
            # infer one missing dim size and get YX dims from pages[0]
            total_pages =  int(np.asarray(self.file_pages).sum())
            if numZ == -1 and numC > -1 and numT > -1:
                numT = max(1, numT)
                numC = max(1, numC)
                numZ = (total_pages-offset)//(numT*numC)
            else:
                numZ = max(1, numZ)
                numC = max(1, numC)
                numT = (total_pages-offset)//(numZ*numC)
            if numC == 1:
                if numZ == 1:
                    raise NotImplementedError('TXY shape not implemented')
                else:
                    self.axes = 'TZYX'
                    self.shape = (numT, numZ, *self.TF[0].pages[0].shape)
            else:
                if axes is None:
                    raise Exception('need axes! (TCZYX or TZCYX)')
                if axes == 'TZCYX':
                    self.axes = 'TZCYX'
                    self.shape = (numT, numZ, numC, *self.TF[0].pages[0].shape)
                elif axes == 'TCZYX':
                    self.axes = 'TCZYX'
                    self.shape = (numT, numC, numZ, *self.TF[0].pages[0].shape)
        self.shapedict = dict(zip(self.axes, self.shape))

        # pixel size
        t04 = time.time()
        if pixel_size.get('X', -1)>0 and pixel_size.get('Y', -1)>0 and pixel_size.get('Z', -1)>0:
            self.pixel_size = pixel_size
        else:
            self.params_inferred = self.infer_params()
            self.pixel_size = {}
            self.pixel_size['X'] = self.params_inferred.get('pixel_size_X', -1)
            self.pixel_size['Y'] = self.params_inferred.get('pixel_size_Y', -1)
            self.pixel_size['Z'] = self.params_inferred.get('pixel_size_Z', -1)
            self.pixel_size.update(pixel_size)

        # yay finally build the index
        t05 = time.time()
        self.build_index()

        t06 = time.time()
        if timing:
            print('==== TiffReader.__init__() timing ====')
            print('init TiffFiles    %6.3f' % (t01a-t00))
            print('whaaa             %6.3f' % (t01b-t01a))
            print('page byte offsets %6.3f' % (t02-t01b))
            print('infer params 1    %6.3f' % (t03-t02))
            print('infer params 2    %6.3f' % (t04-t03))
            print('infer params 3    %6.3f' % (t05-t04))
            print('build index       %6.3f' % (t06-t05))
            print('-----')
            print('TOTAL             %6.3f' % (t06-t00))


        #### Testing PIL to count frames (no faster than Tifffile)
        # from PIL import Image
        # t000 = time.time()
        # tiffstack = Image.open(self.files[0])
        # t001 = time.time()
        # tiffstack.load()
        # t002 = time.time()
        # num = tiffstack.n_frames
        # t003 = time.time()
        # print('==== PIL timing ====')
        # print('open  ', t001-t000)
        # print('load  ', t002-t001)
        # print('num   ', t003-t002)

        #self.mmmd = [tiff.micromanager_metadata for tiff in self.TF]

    def about(self):
        """helpful information at a glance"""
        print('#------------------------')
        print('# TiffReader metadata')
        print('#------------------------')
        print('num_files      :', len(self.files))
        for i, x in enumerate(self.file_pages):
            print('file%3.3i        :' % i, x)
        print('pages/file     :', self.file_pages)
        print('total pages    :', int(np.asarray(self.file_pages).sum()))
        print(' clip_start    :', self.offset)
        print(' clip_end      :', self._clip_end)
        print('dtype          :', self.dtype)
        print('axes           :', self.axes)
        print('shape          :', self.shape)
        print('pixel_size     :', self.pixel_size)
        print('df_index       :')
        print(pd.concat([self.df_index.head(), self.df_index.tail()]))
        print('meta           :')
        print('#----')
        for k,v in self.meta.items():
            if isinstance(v, pd.DataFrame):
                print('%-20s:\n' % k, v)
            else:
                print('%20s:' % k, v)
        print('#------------------------')
        print()

    def clone(self, timing=False):
        """make another one, helpful cuz tifffile is not thread safe"""
        kwa = dict(
            f = [x for x in self.files],
            axes = self.axes+'',
            shape = tuple([x for x in self.shape]),
            pixel_size = {k:v for k,v in self.pixel_size.items()},
            offset=self.offset,
            wallclock=self.wallclock,
            meta={k:v for k,v in self.meta.items()},
            timing=timing,
            page_byte_offsets=self.page_byte_offsets
            )
        return TiffReader(**kwa)

    @classmethod
    def from_json(cls, f=None, timing=False):
        """alt constructor"""
        with open(f) as jfopen:
            jd = json.load(jfopen)

        # establish paths of tiff files, relative to the json input
        jfloc = os.path.dirname(os.path.abspath(f))
        files = [os.path.abspath(os.path.join(jfloc, f)) for f in jd['files']]

        kwa = dict(
            f=files,
            numT=jd.get('numT', -1),
            numZ=jd.get('numZ', -1),
            numC=jd.get('numC', -1),
            axes=jd.get('axes', None),
            shape=jd.get('shape', None),
            offset=jd.get('offset', 0),
            meta=jd.get('meta', {}),
            pixel_size=jd.get('pixel_size', {}),
            wallclock=jd.get('wallclock', None),
            timing=timing,
            page_byte_offsets=jd.get('page_byte_offsets', None)
            )

        return cls(**kwa)

    def to_json(self, out='tiffreader.json'):
        """better name"""
        self.dump_metadata(out=out)

    def dump_metadata(self, out='tiffmetadata.json'):
        """output metadata to json file"""

        if os.path.dirname(out) != '':
            outdir = os.path.dirname(out)
            os.makedirs(outdir, exist_ok=True)
        else:
            outdir = './'
        
        pwd = os.path.abspath('./')
        fabs = [os.path.join(pwd, f) for f in self.files]
        frel = [os.path.relpath(f, outdir) for f in fabs]

        #pdb.set_trace()
        # dd = {k:v for k,v in self.meta.items()}
        # dd['files'] = frel
        # dd['dtype'] = str(self.meta['dtype'])
        
        if self.wallclock is not None:
            wallclock = list(self.wallclock)
        else:
            wallclock = None

        dd = dict(
            #files=fabs,
            files=frel,
            dtype=str(self.dtype),
            axes=self.axes,
            shape=self.shape,
            offset=self.offset,
            pixel_size=self.pixel_size,
            wallclock=wallclock,
            page_byte_offsets=self.page_byte_offsets
        )

        #for k,v in dd.items():
        #    print('%20s : ' % (k), type(v))
        
        with open(out, 'w') as f:
            json.dump(dd, f, indent=2)
            f.write('\n')
        return

    def infer_params(self):
        """infer hyperstack parameters
        
        This is risky because tiff files are not reliable reporters of all
        dimension/shape attributes. Some typical problem cases
        - A TYX time series might be encoded as CYX
        - The offset (inital frames comprising an incomplete volume) cannot
            be inferred.
        - OME metadata missing/corrupt
        - micromanager metadata missing/corrupt
        """
        try:
            x = self.params_inferred
            return x
        except:
            pass

        out = dict(
            _about="TiffReader inferred parameters",
            axes=None,
            shape=None,
            shapeYX=None,
            pixel_size_X=-1,
            pixel_size_Y=-1,
            pixel_size_Z=-1,
            df_index=None
        )
        try:
            ### try to parse OME XML
            dd = parse_pg0_OME(self.TF[0])
            out.update(dd)
            out['shapeYX'] = out['shape'][-2:]
        except:            
            ### fallback use series[0]
            axes, shape = get_shape_sr0(self.TF[0])
            out['axes'] = axes
            out['shape'] = shape
            out['shapeYX'] = out['shape'][-2:]
        return out

    def build_index(self):
        """global frame index for >=1 tiff file(s)

        The dataframe df holds all of the tiff page indexing info. Columns
        T, Z, and C are self-explanatory. F is the (input) file index and 'ndx'
        is the page index WITHIN the corresponding file. This way, given, TZC,
        one can determine F and ndx.

        tzc2fndx is a dictionary. Keys are (T,Z,C) tuples (converted to strings)
        and values are (F, ndx) tuples
        """
        # essential parameters to build the index
        axes = self.axes
        shape = self.shape
        offset = self.offset
        total_pages = int(np.asarray(self.file_pages).sum())

        self._clip_end = (total_pages-offset)%(self.shapedict.get('Z', 1)*self.shapedict.get('C', 1))

        # file/page index
        ndx, fndx = [], []
        for i, ff in enumerate(self.files):
            ndx += list(range(self.file_pages[i]))
            fndx += [i]*self.file_pages[i]

        # split axes into tzc and xy
        xy_axes = axes[-2:]
        xy_shape = shape[-2:]
        tzc_axes = axes[:-2]
        tzc_shape = shape[:-2]
        tzc_ndx = list(itertools.product(*[range(n) for n in tzc_shape]))

        # some consistency checks
        if xy_axes not in ['XY', 'YX']:
            raise Exception('expecting the last two axes to be XY or YX ')

        # if len(ndx)-offset != len(tzc_ndx):
        #     print('WARNING: something is wrong')
        #     print('len ndx / tzc_ndx', len(ndx), len(tzc_ndx))
        #     raise Exception()

        # a dataframe to hold ALL the indexing information
        df = pd.DataFrame(data=tzc_ndx, columns=list(tzc_axes))
        for col in ['T', 'Z', 'C']:
            if col not in df.columns:
                df[col] = np.zeros(len(df), dtype=int)
        df['F'] = fndx[offset:(total_pages-self._clip_end)]  # clip beginning and end
        df['ndx'] = ndx[offset:(total_pages-self._clip_end)] # clip beginning and end
        df = df[['T', 'Z', 'C', 'F', 'ndx']]
        if self.wallclock is not None:
            df['wallclock'] = self.wallclock[offset:(total_pages-self._clip_end)]
        self.df_index = df

        # build the hash: given TZC, return [file,frame] indices
        tzc2fndx = {str(tuple(x[:3])): (x[3],x[4]) for x in self.df_index[['T', 'Z', 'C', 'F', 'ndx']].values}

        # some extra metadata
        self.meta_extra = dict(
            axes_xy=xy_axes,
            shape_xy=xy_shape,
            axes_tzc=tzc_axes,
            shape_tzc=tzc_shape,
            tzc2fndx=tzc2fndx
            )
        return

    def getframe(self, Z=0, T=0, C=0):
        """returns a (YX) frame given (hyper)stack indices Z,T,C

        TODO: direct request of a frame subregion (YX)?
        """
        f, index = self.meta_extra['tzc2fndx'][str(tuple([T, Z, C]))]
        # data1 = self.TF[f].pages[index].asarray()
        # turbo access :)
        _ = self.TF[f].filehandle.seek(self.page_byte_offsets[f][index])
        data = TiffPage(self.TF[f], index=0).asarray()
        axes = self.meta_extra['axes_xy']
        dtype = self.dtype
        meta = dict(Z=Z, T=T, C=C, dtype=dtype)
        return dict(data=data, axes=axes, meta=meta)

    def getchunk(self, req=None):
        """Assemble a DataChunk, using only the required frame requests 

        req : dict
            The chunk request. Accepts values, ranges, slices of the 
            dimensions. If a dimension is not included, then all values are
            retrieved 
        
            examples:
                # range of times, two z planes, one color channel
                req = dict(T=(0,10), Z=[0,1], C=1)
                # every 10th time
                req = dict(T=(None, None, 10))

        TODO: better naming for chunk requests (compact vs full?)
        """
        # dtype = self.meta['dtype']
        dtype = self.dtype

        #== confirm that requested axes exist in the file
        for a in req.keys():
            if a not in self.axes:
                raise Exception('requested axis (%s) not in the tiff axes (%s)'
                % (a, self.axes))

        #---------------------------------------------------------------
        # Enumerate TZC combinations (YX frames) and generate ALL of the
        # getframe requests. Each request returns a dictionary
        shape_tzc = self.meta_extra['shape_tzc']
        axes_tzc = self.meta_extra['axes_tzc']
        req_tzc = {k:req.get(k, None) for k in axes_tzc}
        ix_tzc = chunk_ix(shape=shape_tzc, dims=axes_tzc, req=req_tzc)
        #== list of tuples
        TZC = list(itertools.product(*ix_tzc))

        #== the XY request (basically a crop) is done on each TZC frame
        shape_xy = self.meta_extra['shape_xy']
        axes_xy = self.meta_extra['axes_xy']
        req_xy = {k:req.get(k, None) for k in 'XY'}
        ix_xy = chunk_ix(shape=shape_xy, dims=axes_xy, req=req_xy)

        #== determine the shape of the output chunk
        new_shape_tzc = [len(x) for x in ix_tzc]
        new_shape_xy = [len(x) for x in ix_xy]

        #== build a list of frames, then reshape it
        shp = [len(TZC)] + new_shape_xy
        data = np.zeros(shp, dtype=dtype)
        for i, row in enumerate(TZC):
            reqi = dict(zip(axes_tzc, row))
            frm = self.getframe(**reqi)
            # if no req_xy, do not use DataChunk
            if req_xy['X'] == None and req_xy['Y'] == None:
                data[i] = frm['data']
            else:
                frm['dims'] = frm['axes']
                frm.pop('axes')
                chk = DataChunk(**frm).subchunk(req=req_xy)
                data[i] = chk.data
            #print(row, dict(zip(axes_tzc, row)))

        #== reshape
        shape = new_shape_tzc+new_shape_xy
        data = data.reshape(shape)

        #== figure out the offset
        offset = {}
        offset_tzc = dict(zip(axes_tzc ,[i[0] for i in ix_tzc]))
        offset_xy  = dict(zip(axes_xy ,[i[0] for i in ix_xy]))
        offset.update(offset_tzc)
        offset.update(offset_xy)

        #== build the DataChunk
        axes = axes_tzc + axes_xy
        meta = dict(
            _about='DataChunk from a tiff file',
            req=req,
            offset=offset
            )
        chunk = DataChunk(data=data, dims=axes, meta=meta)
        return chunk

def sort_this_mess(f):
    """sorts a list of tiff files (comprising one recording)

    Parameters:
    -----------
    f: (list) unsorted list of filenames

    Returns:
    --------
    fsrt: (list) sorted list of filenames

    Micro-manager creates output files with a naming convention that is
    incompatible with a lexical sort. For example,
    ```
    image.ome.tif
    image_1.ome.tif
    image_10.ome.tif
    image_11.ome.tif
    image_2.ome.tif
    ```

    This function first detects if the MM scheme was used to name the files.
    If so, it sorts them correctly and otherwise it falls back to a regular
    lexical sort.

    Assumptions:
        - all successive files are present
        - the files are all in one directory
    """
    assert isinstance(f, list)

    # zero or one file
    if len(f) in [0, 1]:
        return f

    # generate filenames using the micromanager convention
    prefix = os.path.commonprefix(f)
    suffix = os.path.commonprefix([x[::-1] for x in f])[::-1]
    fsrt = [prefix+suffix]+[prefix+'_%i'%i+suffix for i in range(1, len(f))]

    # test if generated files match actual files, if not use lexical sort
    isMM = all([x in f for x in fsrt])
    if isMM:
        #print('applying MM sorting')
        return fsrt
    else:
        #print('applying lexical sorting')
        return sorted(f)




def speed_test_infer_shape(tf):
    """A speed test comparison of methods to infer tiff hyperstack shape (and
    dimension order)
    
    Three methods:
    1. parsing micromanager: 
    2. parsing tiff page0 (OME): 
    3. parsing tiff series0: 

    1 is the slowest, 2 is the fastest, and 3 is both slow and error prone.

    Parameters
    ----------
    tf : (tifffile.TiffFile)
    """
    #### method 1 (micromanager tags)
    t00 = time.time()
    get_shape_mm(tf)

    #### method 2 (pages[0] ome)
    t01 = time.time()
    axes, shape = get_shape_ome(tf)

    #### method 3 (series[0])
    t02 = time.time()
    axes, shape = get_shape_sr0(tf)

    t03 = time.time()
    print('---- speed test to infer tiff hyperstack shape ----')
    print('time mm : %7.2f ms' % ((t01-t00)*1000) )
    print('time pg0: %7.2f ms' % ((t02-t01)*1000) )
    print('time sr0: %7.2f ms' % ((t03-t02)*1000) )


def parse_pg0_OME(tf):
    """digs out metadata from tiff page zero. 

    Getting essential metadata for OME tiffs requires an annoying and slow xml
    parse. So we do it ONCE here

    dig out:
        - shape, axes
        - index dataframe (TZC and wall clock time)
        - pixel size
    
    NOTE: this is incompatible with tifffile/imageJ written files which
        use a totally different scheme for the ImageDescription tag...
    NOTE: Length 1 dimensions (i.e. C for single channel recording) are not
    included in the output.

    Parameters
    ----------
    tf : tifffile.TiffFile
    
    Returns
    -------
    dd : dict
        dict of results
    """
    # for tag in tf.pages[0].tags:
    #     print(tag.name)
    # print('---')
    # print(tf.pages[0].tags['ImageDescription'].value)
    # print(type(tf.pages[0].tags['ImageDescription'].value))
    # print('----')

    #### INDEX
    try:
        description = xmltodict.parse(tf.pages[0].tags['ImageDescription'].value)
    except:
        pass
        # TODO: write imageJ parser
        #print('xml parse FAIL:')
        #print(tf.pages[0].tags['ImageDescription'].value)
    plane = description['OME']['Image']['Pixels']['Plane']
    data = []
    for p in plane:
        data.append(dict(
            time=float(p['@DeltaT']),
            #exposure=float(p['@ExposureTime']),
            #posZ=float(p['@PositionZ']),
            T=int(p['@TheT']),
            C=int(p['@TheC']),
            Z=int(p['@TheZ']),
            )
        )    
    df_out = pd.DataFrame(data)

    #### PIXEL SIZE
    pixel_size_X = float(description['OME']['Image']['Pixels'].get('@PhysicalSizeX', -1))
    pixel_size_Y = float(description['OME']['Image']['Pixels'].get('@PhysicalSizeY', -1))
    pixel_size_Z = float(description['OME']['Image']['Pixels'].get('@PhysicalSizeZ', -1))

    #### AXES/SHAPE (infers size, shape, and dimension order)
    dims = []
    for k in ['T', 'Z', 'C']:
        x = df_out[k].values
        ix = np.where(x != 0)[0]
        size = np.max(x) - np.min(x) + 1
        if size > 1:
            dims.append(dict(dim=k, size=size, ix=ix[0]))
    dims = sorted(dims, key=lambda xx: xx['ix'], reverse=True)
    shapeYX = tf.pages[0].shape
    axes = ''.join([d['dim'] for d in dims] + ['Y', 'X'])
    shape = tuple([int(d['size']) for d in dims] + list(shapeYX))
    #### PACK UP
    dd = dict(
        axes=axes,
        shape=shape,
        pixel_size_X=pixel_size_X,
        pixel_size_Y=pixel_size_Y,
        pixel_size_Z=pixel_size_Z,
        df_index=df_out,
    )
    return dd


def get_shape_ome(tf):
    """Uses OME metadata in pages[0] to infer tiff hyperstack shape
    
    parameters
    ----------
    tf : tifffile.TiffFile

    returns
    -------
    axes : (str) Hyperstack axes e.g. 'TZCYX' or 'TZYX'
    shape : (list) Hypserstack shape (corresponding to axes)
    """
    dd = parse_pg0_OME(tf)
    return dd['axes'], dd['shape']

def get_shape_sr0(tf):
    """Uses series[0] to infer tiff hypserstack shape

    parameters
    ----------
    tf : tifffile.TiffFile

    returns
    -------
    axes : (str) Hyperstack axes e.g. 'TZCYX' or 'TZYX'
    shape : (list) Hypserstack shape (corresponding to axes)

    In theory, series[0] is a reliable place to find hyperstack shape unlike
    imageJ and Micromanager metadata which are not guaranteed to be in every
    tiff file.

    HOWEVER, for the most recent GCaMP tiffs (beanshell script), calling
    series[0] will
    - spam a ton of logging warnings about OME index errors
    - and corrupt the tifffile reader
    
    To remedy these problems, this routine
    - temporarily switches off logging errors
    - creates a copy/sacrificial TiffFile instance that is closed when done
    """
    import logging

    # switch logging to suppress spam
    logging.getLogger("tifffile").setLevel(logging.ERROR)

    # instantiate a sacrificial TiffFile (in case it is corrupted)
    tf_loc = TiffFile(tf.filehandle.path)

    series0 = tf_loc.series[0]
    axes = series0.axes
    shape = series0.shape

    # close the local TiffFile
    tf_loc.close()

    # restore logging
    logging.getLogger("tifffile").setLevel(logging.WARNING)

    return axes, shape

def get_shape_mm(tf):
    """Uses micromanger metadata tags to infer tiff hyperstack shape
    NOTE: this is incomplete.. (does not infer dimension order)
    """
    max_slice_index = -1
    max_channel_index = -1
    for tif in [tf]:
        for page in tif.pages:
            max_channel_index = max(max_channel_index, page.tags['MicroManagerMetadata'].value['ChannelIndex'])
            max_slice_index = max(max_slice_index, page.tags['MicroManagerMetadata'].value['SliceIndex'])
    numz = max_slice_index + 1
    numc = max_channel_index + 1



def infer_dimension_order(df):
    """dataframe with columns T Z C"""
    print(df.head())
    for k in ['T', 'Z', 'C']:
        x = df[k].values
        ix = np.where(x != 0)[0]
        size = np.max(x) - np.min(x) + 1
        if size > 1:
            dims.append(dict(dim=k, size=size, ix=ix[0]))
    dims = sorted(dims, key=lambda xx: xx['ix'], reverse=True)
    #shapeYX = tf.pages[0].shape
    axes = ''.join([d['dim'] for d in dims]) #+ ['Y', 'X'])
    #shape = tuple([d['size'] for d in dims] + list(shapeYX))
    return axes
