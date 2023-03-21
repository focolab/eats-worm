import copy
import pdb
import itertools
import json
import os
import base64

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def chunk_ix(shape=None, dims=None, req=None):
    """indices for a chunk request (expanding a chunk request)

    'shape' and 'dims' describe the shape and dims names of an array A of
    arbitrary dimensions. 'req' is a (compact, dictionary) request for a chunk
    of A (np.ndarray).

    arguments
    ------
    shape (list): array shape
    dims (list or str): array dims ('XYZTC' or ['X','Y','Z','T','C'])
    req (dict): chunk request (e.g. {'X':1, 'T':[0,2], 'Y':(0,5)})

    req format options:
        X=1             interpreted as [1]
        X=(0,3)         interpreted as range(0,3)
        X=[0,3]         interpreted as itself (a list)
        X=slice(0,3)    python slice (e.g. slice(None,None,10) for every 10th)
        X=None          (default) all values

    returns
    ------
    ix (list):  a list of lists of index values for each dimension. Using the 
                output, numpy.ix_(ix) is a valid slice of A. Similarly, 
                itertools.product(*ix) makes a list of tuples for chunk
                elements in A.

    TODO: how to handle missing dims in the request? Return all/zero/error?
            A: if size is 1, use it, otherwise
    """

    #== check if requested dims DNE in the data
    for k in req.keys():
        if k not in dims:
            raise Exception('requested axis (%s) not in dims (%s)' % (k, str(dims)))

    #== a partial request, missing some dims, is made explicit
    reqloc = {k:v for k,v in req.items()}
    for a in dims:
        if a not in req.keys():
            reqloc[a] = None
            #print('WARNING: request set to None for %s' % a)

    ix = []
    for size, dim in zip(shape, dims):
        r = reqloc[dim]
        if isinstance(r, (int, np.integer)):
            ix.append([r])
        elif isinstance(r, tuple):
            ix.append(list(range(*r)))
        elif isinstance(r, list):
            ix.append(r)
        elif isinstance(r, slice):
            ix.append(range(size)[r])
        elif r is None:
            ix.append(list(range(size)))
        else:
            raise Exception('chunk index: (%s) not recognized' %
                            (str(reqloc[dim])))

    return ix


class DataChunk(object):
    """ND data chunk and metadata

    attributes
    ------
    data: (np.ndarray)
    axes: (str) 'TZCYX' or similar
    meta: (dict) json-izable metadata, otherwise unrestricted


    TODO: coords attribute (echoing xarray)
    TODO: global channel info
    TODO: chunk type attribute (montage, etc..)
    TODO: axes values and units
    TODO: track an offset, possibly a stride too
    TODO: global vs local properties? (i.e. global intensity limits vs local)

    json en/de-coding
    https://stackoverflow.com/a/6485943/6474403
    """

    def __init__(self, data, dims=None, meta=None, axes=None):

        self.data = data

        if axes is not None:
            raise Exception('axes no longer accepted, used dims instead')
        self.dims = dims
        if meta is None:
            self.meta = {}
        else:
            self.meta = meta

        #== derived
        self.dim_len = dict(zip(self.dims, self.data.shape))

        self._channel_ranges = None


    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def channel_ranges(self, dim='C'):

        if self._channel_ranges is None:
            ranges = []
            for ic in range(self.dim_len[dim]):
                data = self.subchunk(req={dim:ic}).data.ravel()
                ranges.append([np.min(data), np.max(data)])
            self._channel_ranges = ranges
        return self._channel_ranges


    def about(self, verbose=False):
        """report about this DataChunk"""
        print('dtype    :', self.dtype)
        print('dims     :', self.dims)
        print('shape    :', self.shape)
        print('meta     :')
        for k, v in self.meta.items():
            if len(str(v))>120:
                beg = str(v)[:60]
                end = str(v)[-60:]
                print('  %10s : %s' % (str(k), beg))
                print('  %10s   %s' % ('', '...'))
                print('  %10s   %s' % ('', end))

            else:
                print('  %10s : %s' % (str(k), str(v)))
        if verbose:
            print('data     :')
            print(self.data)

    def squeeze(self, inplace=False):
        """drop dimensions with shape==1

        TODO: put squeezed out dimensions into meta
        """
        if inplace:
            raise Exception('inplace not yet implemented')

        data = np.squeeze(self.data)
        keep = np.where(np.asarray(self.shape) > 1)[0]
        dims = [self.dims[k] for k in keep]
        md = self.meta

        return DataChunk(data=data, dims=dims, meta=md)


    def reorder_dims(self, dims_new, inplace=False, verbose=False):
        """reorder ALL dims (have to specify them all)

        TODO: make shortcuts (leading, trailing) that call this
        TODO: np.transpose might be cleaner
        """
        if inplace:
            raise Exception('inplace=True not yet implemented')

        for ax in dims_new:
            if ax not in self.dims:
                raise Exception('bogus request, (%s) not found in existing dims (%s)'
                                % (str(ax), str(self.dims)))

        ndx = list(range(len(self.dims)))       #== source dims to grab
        newloc = dict(zip(dims_new, ndx))       #== new locations, by name
        dst = [newloc[a] for a in self.dims]    #== new locations, by ndx

        if verbose:
            print('ndx:', ndx, self.dims)
            print('dst:', dst, dims_new)

        data = np.moveaxis(self.data, ndx, dst)
        md = self.meta.copy()

        return DataChunk(data=data, dims=dims_new, meta=md)

    def subchunk(self, req=None, inplace=False, squeeze=False, verbose=False):
        """return a subchunk of a data (also a DataChunk)

        chunk indexing format options:
            X=1         interpreted as index 1
            X=(0,3)     interpreted as a range
            X=[0,3]     interpreted as a list
            X=None      all values
        """
        fill = 0

        if inplace:
            raise Exception('inplace=True not yet implemented')

        #== building the chunk request
        req_ix = chunk_ix(shape=self.shape, dims=self.dims, req=req)
        ix = np.ix_(*req_ix)


        ### HANDLE case of the chunk request running out of bounds
        inbounds = True
        req_ix_legal = []
        for vals, size, dim in zip(req_ix, self.shape, self.dims):
            vals = np.asarray(vals)
            legal = np.logical_and(vals>=0, vals<size)
            if False in legal:
                inbounds = False

        if inbounds:
            # The chunk request is in bounds so we just carve it out
            data = self.data[ix]
        else:
            # The chunk request goes out of bounds. We build a chunk of the
            # in-bounds data padded with 'fill' to make the expected shape
            data = np.zeros(tuple([len(x) for x in req_ix])) + fill

            # for each dim get the parent ix that insert into child ix
            ix_parent, ix_child = [], []
            for vals, size, dim in zip(req_ix, self.shape, self.dims):
                vals = np.asarray(vals)
                legal = np.logical_and(vals>=0, vals<size)

                #### to here
                aa = np.where(legal)[0][0]
                bb = np.where(legal)[0][-1]

                #### from here
                a = vals[aa]
                b = vals[bb]

                ix_parent.append(list(range(a, b+1)))
                ix_child.append(list(range(aa, bb+1)))

            # indexing is sorted, now we can extract the (partial) chunk!!
            data[np.ix_(*ix_child)] = self.data[np.ix_(*ix_parent)]


        dims = self.dims
        md = self.meta.copy()

        # current offset
        offset_vals = [i.ravel()[0] for i in ix]
        offset_dict = dict(zip(dims, offset_vals))

        prev_offset = md.get('offset', None)
        if prev_offset is not None:
            # Q: WTF to do here..
            # A: Add offsets? (they don't make sense for color, but that's another issue)
            new_offset = {k:v for k,v in offset_dict.items()}
            for k,v in offset_dict.items():
                new_offset[k] += prev_offset.get(k, 0)
            md['offset'] = new_offset

            if verbose:
                print('--- subchunk offset calculation ---')
                print('prev', prev_offset)
                print('this', offset_dict)
                print('new ', new_offset)
                print('---')
        else:
            md['offset'] = offset_dict
            # pdb.set_trace()

        if verbose:
            print('-----------')
            print('dims   :', self.dims)
            print('shape  :', self.shape)
            print('request:', req)
            print('ix     :', ix)
            print('data   :', data)
            print('dtype  :', self.data.dtype)
            #print(np.squeeze(chunk))
            print('-----------')

        return DataChunk(data=data, dims=dims, meta=md)

    def max_ip(self, dim=None):
        """ carry out max intensity projection on one dimension (by name)
        """

        ax2num = dict(list(zip(self.dims, range(len(self.dims)))))
        dimnum = ax2num[dim]
        arrr = np.amax(self.data, axis=dimnum)

        # build the output DataChunk, squeeze out extra dimensions
        dims = self.dims[:dimnum] + self.dims[dimnum+1:]
        meta = {k:v for k, v in self.meta.items()}
        return DataChunk(data=arrr, dims=dims, meta=meta).squeeze()

    def stack_dims(self, major=None, minor=None):
        """TODO: the general case for montage

        Q: is tile a better name?
        1. declare major and minor axes
        2. reorder dims with major and minor first
        3. for each major step, hstack minors
        4. make new axis name M[m]?

        ex. Given an XYZ chunk, Z[X] would tile XY images (Z planes) along X
        (10, 6, 8) -> (80, 6)
        (Z, Y, X) -> (Z[X], Y)
        """
        pass

    def to_napari(self, scale=None):
        """spawn napari viewer of this chunk

        input
        -----
        scale: (dict)
            scale factors for the data e.g. {'X':0.16, 'Y':0.16, 'Z':1}

        TODO: would like to enable view->axes->visible by default..
        """
        import napari
        with napari.gui_qt():
            viewer = napari.Viewer(ndisplay=2, axis_labels=self.dims)
            if isinstance(scale, dict):
                scale = [scale[x] for x in self.dims]
            layer = viewer.add_image(self.data, scale=scale)

    def montage(self, I=None, J=None, II='', JJ='', T='', C=''):
        """
        stack (images/frames/panels) horizontally and/or vertically

        The canonical case is to stack together multiple XY frames
        (I,J indexed), to show multiple Z-planes or time points

        This method allows horizontal and vertical stacking, where II and JJ
        denote the outer stack dimensions. I and J denote inner dimensions of
        a single frame (possibly with a 3rd color dimesion, indicated with C).
        I/II and J/JJ are row and column directions, respectively

        I,J,II,JJ,C,T have the following roles here:
            I: frame/inner vertical axis (row)
            J: frame/inner horizontal axis (column)
            II: stack/outer vertical axis (row)
            JJ: stack/outer horizontal axis (column)
            C: (optional) color axis for multichannel datasets
            T: (optional) time/animate axis (a movie has one (I,II,J,JJ) per T)

        NOTES:  This should not swamp memory, provided the parent DataChunk
                comfortably fits in memory. To dump or stream big montages,
                consider using multiple DataChunks (probably split on the time
                dimension), rather than one colossal chunk.

                This is mainly for visualization/output, not downsteam
                analysis. The stacked dimensions are not set up for easy
                manipulation. Coords would have to be generalized for the
                stacked dimensions, reminiscent of a pandas multi-index
                dataframe. Currently, the metadata 'stacked_dims' is just sort
                of dumped into meta without a real future plan.

                The standalone function montage_labels(chunk) takes a montage
                (DataChunk) and returns indices and label strings for labeling
                a montage image (tested assuming matplotlib.imshow())

        RETURNS
        ------
        a DataChunk with data in TIJC dimension order. This allows direct
        plotting via Matplotlib imshow, for example, which requires (M,N) or
        (M,N,3) data for monochrome/rgb cases.

        TODO: this is a specific case of a more general data reshaping process
        TODO: carry stuff forward (global color channel limits, coords, meta)
        TODO: COORDS GETS BORKED (Y,Z -> 'Y:Z' ? )
        TODO: make an iterator that streams chunks..
        """

        if 'montage' in self.meta.keys():
            raise Exception('cannot montage twice, you weirdo.. start over')

        # verify the request is legit
        for ax in [I, J, II, JJ, T, C]:
            if ax is not '' and ax not in self.dims:
                raise Exception('bogus montage request, (%s) not found in existing dims (%s)'
                                % (str(ax), str(self.dims)))

        # reorder dimensions and make a new chunk
        req_reorder = T+II+JJ+I+J+C
        chunk = self.reorder_dims(req_reorder) #, verbose=True)

        # convenience, data sizes
        Ni = chunk.dim_len.get(I, 1)
        Nj = chunk.dim_len.get(J, 1)
        Nii = chunk.dim_len.get(II, 1)
        Njj = chunk.dim_len.get(JJ, 1)
        Nt = chunk.dim_len.get(T, 1)

        # assemble the montage data, could be accelerated
        data = []
        for it in range(Nt):
            frames = []
            req = {T:it} if Nt>1 else {}
            paneldata = []
            for row in range(Nii):
                if Nii > 1:
                    req[II] = row
                rowdata = []
                for col in range(Njj):
                    req[JJ] = col
                    rowdata.append(chunk.subchunk(req=req).squeeze().data)
                paneldata.append(np.hstack(rowdata))
            data.append(np.vstack(paneldata))
        data = np.asarray(data)


        # update metadata with montage info
        meta = {k:v for k,v in self.meta.items()}

        md = dict(panel_width=Nj,
                  panel_height=Ni,
                  dim_JJ=JJ,
                  dim_II=II,
                  num_JJ=Njj,
                  num_II=Nii,
                  dim_J=J,
                  dim_I=I,
                  num_J=Nj,
                  num_I=Ni,
                  is2D=False)

        if Nii>1 and Njj>1:
            md['is2D'] = True
            md['stackmode'] = '2D'
        elif Nii>1:
            md['stackmode'] = 'v'
        elif Njj>1:
            md['stackmode'] = 'h'
        else:
            raise Exception('Nii<1 and Njj<1.. wtf??')

        meta['montage'] = md


        # build the output DataChunk, squeeze out extra dimensions
        dims = [T, I, J, C]
        return DataChunk(data=data, dims=dims, meta=meta).squeeze()



    def to_jdict(self, data_encoding=None):
        """json compatible dictionary

        data_encoding (str): 'raw' or 'b64str'
        """

        jdict = dict(_about='a DataChunk in json/dictionary form',
                     dims=self.dims,
                     shape=self.shape,
                     dtype=str(self.dtype),
                     meta=self.meta)

        if data_encoding in [None, 'raw']:
            #== serialize the data into a 1D list
            dsrl = self.data.ravel().tolist()
            jdict['data_1D'] = dsrl
        else:
            #== encode data as a base64 string
            data_b64 = base64.b64encode(self.data.ravel())
            data_b64str = data_b64.decode('utf-8')
            jdict['data_b64str'] = data_b64str

        return jdict

    @classmethod
    def from_jdict(cls, d):
        """alternate constructor, using json compatible dictionary"""

        dims = d.get('dims', None)
        shape = d.get('shape', None)
        meta = d.get('meta', None)
        dtype = d.get('dtype', None)

        #== decode and reshape the serialized data
        if 'data_b64str' in d.keys():
            data_b64str = d['data_b64str']
            dd = base64.b64decode(data_b64str)
            data = np.frombuffer(dd, dtype=dtype).reshape(shape)
        elif 'data_1D' in d.keys():
            dsrl = d['data_1D']
            data = np.asarray(dsrl, dtype=dtype).reshape(shape)
        else:
            err = 'no data in the jdict, need either data_1D or data_b64str'
            raise Exception(err)

        kwa = dict(dims=dims,
                   meta=meta,
                   data=data)

        return cls(**kwa)

    def to_json(self, out='chunk.json', make_path=True,
                data_encoding=None):
        """dump DataChunk to a json file"""

        if make_path:
            os.makedirs(os.path.dirname(out), exist_ok=True)

        jdict = self.to_jdict(data_encoding=data_encoding)
        with open(out, 'w') as f:
            json.dump(jdict, f, indent=2)
            f.write('\n')

    @classmethod
    def from_json(cls, j):
        """alternate constructor, from json file"""

        with open(j) as jfopen:
            jdict = json.load(jfopen)

        return cls.from_jdict(jdict)




def montage_labels(chunk=None):
    """generate panel labels for a montage

    NOTE: this REQUIRES coords..

    given a chunk with montage metadata, returns
        xypos: a list of upper left pixel coordinates for the panels
        vals: values of the montage coordinates (not really used)
        labels: string labels (e.g. 'Z=2 T=4') for each panel
    """

    coords = chunk.meta['coords']

    md = chunk.meta['montage']

    #== label positions
    xpos = md['panel_width']* np.arange(md.get('num_JJ', 1))
    ypos = md['panel_height']*np.arange(md.get('num_II', 1))
    xypos = itertools.product(xpos, ypos)

    #==
    if md['is2D']:
        dh = md.get('dim_JJ', '')
        dv = md.get('dim_II', '')
        xval = coords[dh]
        yval = coords[dv]
        xyval = itertools.product(xval, yval)
        vals = xyval
        labels = ['%s=%s \n%s=%s' % (dh, x, dv, y)  for x,y in xyval]

    else:
        if md['stackmode'] == 'h':
            dh = md.get('dim_JJ', '')
            xval = coords[dh]
            vals = xval
            labels = ['%s=%s' % (dh, x)  for x in xval]
        elif md['stackmode'] == 'v':
            dv = md.get('dim_II', '')
            yval = coords[dv]
            vals = yval
            labels = ['%s=%s' % (dv, y)  for y in yval]

    return xypos, vals, labels

