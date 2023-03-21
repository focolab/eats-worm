import json
import numpy as np

class SegmentedBlob(object):
    """holds the info for a blob that has been segmented from an image(vol)

    Right now, this is only intended to describe a blob in a static, volumetric
    image (CZYX) and is not intended for a time varying blob (termed a thread
    in gcamp_extractor). A more general blob/thread spec, where this is just a
    special case, would be ideal.

    The provenence attribute ```prov``` can 'detected', 'curated', or 'imputed'

    The attributes ```status``` and ```index``` are not really intrinsic to a
    blob, but are tacked on for convenience.

    ```status``` in [-1, 0, 1] is a quick and easy way to classify a blob that
    is (-1) marked for deletion, (0) needs checking, or (1) has passed human
    curation. If a list of ```SegmentedBlob```s is curated in multiple
    sessions, this status is important to preserve.

    ```index``` is just a numerical index for sorting and, like ```status``` is
    bound to the blob.

    ```stash``` is not serialized or saved with a SegmentedBlob, but could be
    used to hold temporary associated information, e.g. a pre-computed
    bounding box DataChunk or voxel mask, during an interactive session. Such
    a mask or masks could/should also be their own attributes.
    """
    def __init__(self, index=-1, pos=None, status=0, pad=[10, 10, 4], dims=['X','Y','Z'], ID='', prov='x'):
        self.pos = pos          # position
        self.dims = dims        # let's be explicit and not lose track of these
        self.ID = ID            # string for the cell name
        self.pad = pad
        self.prov = prov        # provenence
        self.status = status    # not well defined yet
        self.index = index      # just an int for tracking
        self.stash = {}         # could stash the bbox datachunk here for speed

    @property
    def posd(self):
        """position, but in dict form so dimensions don't get twisted"""
        return dict(zip(self.dims, self.pos))

    def lower_corner(self):
        """lower corner of the bounding box"""
        return {d:r-p for r,p,d in zip(self.pos, self.pad, self.dims)}

    def clone(self):
        """This is mainly used to create a dummy/provisional blob in a GUI.
        A hard copy is done so as not to be referring to another blob's
        attributes
        """
        po = [x for x in self.pos]
        pa = [x for x in self.pad]
        di = [x for x in self.dims]
        return SegmentedBlob(pos=po, pad=pa, dims=di, status=0, prov=self.prov)

    def to_jdict(self):
        """serialize to a json-izable dictionary"""
        dd = dict(
            pos=[int(x) for x in self.pos],
            pad=[int(x) for x in self.pad],
            dims=self.dims,
            ID=self.ID,
            status=int(self.status),
            index=int(self.index),
            prov=self.prov
        )
        return dd

    def chreq(self, offset=None, rounding=None, pad=None):
        """chunk request for the blob's bounding box

        Parameters
        ----------
        offset : dict
            A dictionary of index offsets for the parent DataChunk,
            needed to correctly extract a bounding chunk from a dataset that
            has been cropped down (i.e. a DataChunk with an offset).
            For example, {'X': 10, 'Y':0, 'Z':4}

        rounding : str
            The method to round a non-integer blob center to have
            integer coordinates (i.e. associate it with a voxel).

        pad : dict
            (optional to override the attribute)
            Same format as offset.
            The chunk is defined by the central voxel plus padding. i.e. the
            interval [X-pad['X'], X+pad['X']] and the equivalent for 'Y' and
            'Z'.


        Returns
        -------
        req: dict
            A DataChunk.subchunk request

        Assuming that the raw data is in a DataChunk, this is a helper that
        generates the DataChunk.subchunk() request, for a blob's bounding box.

        TODO: make sure it does not run outside of parent box dimensions!
        """
        if rounding == None:
            pos_int = self.pos
        elif rounding == 'floor':
            pos_int = np.floor(self.pos).astype(np.int)
        elif rounding == 'nearest':
            pos_int = np.rint(self.pos).astype(np.int)
            #raise NotImplementedError('rounding (%s) not YET implemented', 'closest')
        else:
            raise NotImplementedError('rounding (%s) not implemented', rounding)

        if offset is None:
            offset = {}

        if pad is None:
            pad_arr = self.pad
        else:
            pad_arr = [pad[k] for k in self.dims]

        req = {}
        for r,p,d in zip(pos_int, pad_arr, self.dims):
            req[d] = (r-p-offset.get(d, 0), r+p-offset.get(d, 0)+1)
        return req


def load_blobs(j=None):
    """load blobs from json"""
    with open(j) as jfopen:
        data = json.load(jfopen)
    #return [SegmentedBlob(**x) for x in data]

    if isinstance(data, list):
        # list case
        blobs = [SegmentedBlob(**x) for x in data]
    else:
        # preferred, dict case
        d = data.get('blobs', [])
        blobs = [SegmentedBlob(**x) for x in d]

    return blobs
