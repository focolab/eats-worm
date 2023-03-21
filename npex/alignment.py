# utils for rotational alignment

import json
import os
import pdb

import numpy as np
import pandas as pd
from scipy.linalg import orthogonal_procrustes

def scale_blob_coords(df, s=None):
    """from pixel coords ICS to LCS, (scale the XYZ data by voxel size)

    df: DataFrame with X Y Z columns
    s: dict voxel size [microns]
    """
    if s is None:
        s = dict(X=1, Y=1, Z=1)
    cols = list(s.keys())
    vals = list(s.values())
    df_out = df[cols]*np.asarray(vals)

    # tack on the other columns 
    other_cols = [x for x in df.columns if x not in cols]
    df_out[other_cols] = df[other_cols]
    return df_out



def align_data_to_reference(df_a=None, df_b=None, dest=None, tag='data_tag', dump_pdbs=True):
    """align 3D point set data (a) to a reference (b)

    Parameters
    ----------
    df_a : DataFrame
        dataframe of with X Y Z and ID cols
    df_b : DataFrame
        dataframe of reference (atlas) coordinates. Should have X Y Z and ID cols
    dest : str
        if passed, output folder

    Returns
    -------
    out : dict
        lotta stuff..

    this is doing too many things
    - staging/culling/sorting coordinate sets
    - crusty alignment
    - pdb export

    TODO could this be a method(s) of CrustyAligner?
    """
    xyzcols = ['X', 'Y', 'Z']
    sort_axis = 'X'

    # align using the set intersection of neuron names
    s1 = set(df_a['ID'])
    s2 = set(df_b['ID'])
    blob_IDs = sorted(list(s1.intersection(s2)))

    if len(blob_IDs) < 3:
        print('fewer than three blobs, cannot perform alignment')
        return None

    # make dataframes with just the fit subset
    ix_a = [i for i,x in enumerate(df_a['ID']) if x in blob_IDs]
    ix_b = [i for i,x in enumerate(df_b['ID']) if x in blob_IDs]

    assert len(ix_b) == len(ix_a), 'incompatible fit subsets (duplicates?)'

    # register the fit subsets by ID then sort by sort_axis
    df_a_fit = df_a.iloc[ix_a].sort_values('ID').reset_index(drop=True)
    df_b_fit = df_b.iloc[ix_b].sort_values('ID').reset_index(drop=True)
    if sort_axis:
        ix_sorted = np.argsort(df_b_fit[sort_axis].values)
        df_a_fit = df_a_fit.iloc[ix_sorted].reset_index(drop=True)
        df_b_fit = df_b_fit.iloc[ix_sorted].reset_index(drop=True)

    # CRUSTY ALIGNMENT
    cra = CrustyAligner()
    cra.train(df_b_fit[xyzcols].values, df_a_fit[xyzcols].values)
    xyz_c = cra.align(df_a[xyzcols].values)
    xyz_c_fit = cra.align(df_a_fit[xyzcols].values) 

    # make output dataframes
    df_c_fit = pd.DataFrame(xyz_c_fit, columns=xyzcols)
    df_c_fit['ID'] = df_b_fit['ID']
    df_c_fit['distance'] = np.sqrt(np.sum((df_b_fit[xyzcols].values - df_c_fit[xyzcols].values)**2, axis=1))

    df_c =  pd.DataFrame(xyz_c, columns=xyzcols)
    df_c_fit['ID'] = df_b_fit['ID']

    # alignment scores
    scores = dict(
        d_avg=np.mean(df_c_fit['distance']),
        d_rms=np.sqrt(np.mean(df_c_fit['distance']**2)),
        d_min=np.min(df_c_fit['distance']),
        d_max=np.max(df_c_fit['distance']),
        n_fit=len(df_c_fit),
        scale=cra.a2b['norm2']/cra.a2b['norm1']
    )


    # EXPORT pdbs (for pymol viz)
    if dest is not None:
        os.makedirs(dest, exist_ok=True)
        if dump_pdbs:
            cij = [(i,i+1) for i in range(len(df_c_fit))] [:-1]
            pdb_a = pdb_lines(df_b_fit[xyzcols].values.T, cij=cij)
            pdb_b = pdb_lines(xyz_c_fit.T, cij=cij)
            pdb_c = pdb_lines(df_b[xyzcols].values.T)
            pdb_d = pdb_lines(xyz_c.T)
            dump_lines(pdb_a, os.path.join(dest, 'pdb_a_fit_ref.pdb'))
            dump_lines(pdb_b, os.path.join(dest, 'pdb_b_fit_aligned.pdb'))
            dump_lines(pdb_c, os.path.join(dest, 'pdb_c_all_ref.pdb'))
            dump_lines(pdb_d, os.path.join(dest, 'pdb_d_all_aligned.pdb'))

            label_filename = os.path.join(dest, 'labels.pml')
            with open(label_filename, 'w') as fopen:
                for ix, row in df_c_fit.iterrows():
                    fopen.write('label i. %i and pdb_a*, \"%s\" \n' % (ix, row['ID']))
                    fopen.write('label i. %i and pdb_b*, \"%s\" \n' % (ix, row['ID']))

        # EXPORT other stuff
        dd = dict(
            data_tag=tag,
            scores=scores,
            a2b=cra.a2b_exportable,
            files={
                'df_c_fit':os.path.join(dest, 'df_c_fit.csv'),
                'df_c':os.path.join(dest, 'df_c.csv'),
            },
        )

        df_c_fit.to_csv(os.path.join(dest, 'df_c_fit.csv'), float_format='%6g')
        df_c.to_csv(os.path.join(dest, 'df_c.csv'), float_format='%6g')
        scorefile = os.path.join(dest, 'scores.json')
        with open(scorefile, 'w') as f:
            json.dump(scores, f, indent=2)
            f.write('\n')

    out = dict(
        #dest=os.path.abspath(dest),
        df_c=df_c,
        df_c_fit=df_c_fit,
        scores=scores,
        a2b=cra.a2b_exportable
    )

    return out
    # return dd


class AnotherAligner(object):
    def __init__(self):
        pass
    def execute(self):
        pass
    def export(self):
        pass
    @classmethod
    def load(cls, self):
        pass

class CrustyAligner(object):
    """Procrustes coordinate transformer

    Learns and applies a Procrustes transformation to align two sets of labeled
    points with correspondence (1:1 mapping).

    The Procrustes transformation can include net translation, isotropic
    scaling (including symmetry inversion) and rigid rotation. Anisotropic
    scaling corrections (e.g. for imaged confocal volumes) must be applied
    prior to this transformation.
    """
    def __init__(self, a2b=None):
        self.a2b = a2b
    def train(self, data1, data2):
        """learn the transformation to align data2 to data1"""
        # TODO report goodness of fit?
        self.a2b = procrusty(data1, data2)
    def align(self, data, cols=None):
        """align data from coord-sys 2 into coord-sys 1"""
        R = self.a2b.get('R')
        com1 = self.a2b.get('com1')
        com2 = self.a2b.get('com2')
        norm1 = self.a2b.get('norm1')
        norm2 = self.a2b.get('norm2')
        return np.dot(R,(data-com2).T).T*(norm1/norm2)+com1
    @property
    def a2b_exportable(self):
        """json safe version"""
        d = {k:v for k,v in self.a2b.items()}
        d['com1'] = d['com1'].tolist()
        d['com2'] = d['com2'].tolist()
        d['R'] = d['R'].tolist()
        return d

def procrusty(data1, data2):
    """modified scipy.spatial.procrustes

    Fits data2 to data1 using the procrustes transformation. Unlike 
    scipy.spatial.procrustes, this returns all parameters needed to
    re-apply the same transformation.
    """
    mtx1 = np.array(data1, dtype=np.double, copy=True)
    mtx2 = np.array(data2, dtype=np.double, copy=True)

    if mtx1.ndim != 2 or mtx2.ndim != 2:
        raise ValueError("Input matrices must be two-dimensional")
    if mtx1.shape != mtx2.shape:
        raise ValueError("Input matrices must be of same shape")
    if mtx1.size == 0:
        raise ValueError("Input matrices must be >0 rows and >0 cols")

    # COM locations
    com1 = np.mean(mtx1, 0)
    com2 = np.mean(mtx2, 0)

    # translate all the data to the origin
    mtx1 -= np.mean(mtx1, 0)
    mtx2 -= np.mean(mtx2, 0)

    norm1 = np.linalg.norm(mtx1)
    norm2 = np.linalg.norm(mtx2)

    if norm1 == 0 or norm2 == 0:
        raise ValueError("Input matrices must contain >1 unique points")

    # change scaling of data (in rows) such that trace(mtx*mtx') = 1
    mtx1 /= norm1
    mtx2 /= norm2

    # transform mtx2 to minimize disparity
    R, s = orthogonal_procrustes(mtx1, mtx2)
    mtx2 = np.dot(mtx2, R.T) * s

    # measure the dissimilarity between the two datasets
    disparity = np.sum(np.square(mtx1 - mtx2))

    dd = dict(
        com1=com1,
        com2=com2,
        norm1=norm1,
        norm2=norm2,
        disparity=disparity,
        R=R,
        s=s
    )
    return dd









#### these are for export and then visualization with pymol
def dump_lines(lines, fname):
    """helper to dump lines to file"""
    with open(fname, 'w') as fopen:
        for x in lines:
            fopen.write(x+'\n')

def pdb_lines(xyz=None, cij=None, time_=0, model=0, shift=[0,0,0], box=[20,20,20]):
    """dump xyz coords to pdb (protein data bank) format, (list of strings)

        pdb = Protein Data Bank
        allows subsequent viewing with e.g. pymol

        input
        ------
        xyz (np.ndarray) : shape (2 or 3, num_neruons)
        cij (??) : connectivity
        time_ (int or float) : assign a time to a frame
        model (int) : whaaa?
        shift (list): translation shift (length 3)
        box (list): box size (length 3)

        returns
        ------
        out (list) : list of strings (rows to be printed)
    """
    if xyz.shape[0] == 2:
        xyz = np.vstack([xyz, xyz[0]*0])

    # header
    h1 = 'TITLE     spring-mass toy system t=  %g' % (time_)
    h2 = 'CRYST1  %7.3f  %7.3f  %7.3f  90.00  90.00  90.00 P 1           1' % (box[0], box[1], box[2])
    h3 = 'MODEL        %i' % (model)
    out = [h1, h2, h3]

    # atoms
    for ii, row in enumerate(xyz.T):
        x,y,z = row+np.asarray(shift)
        aa = 'ATOM  %5i  CN1 DMPCB%4i    %8.3f%8.3f%8.3f  1.00  0.00           C' % (ii, ii, x, y, z)
        out.append(aa)

    # bonds
    if cij is not None:
        for ii, jj in cij:
            sss = 'CONECT%5i%5i' % (ii,jj)
            out.append(sss)

    out.append('ENDMDL')
    return out


