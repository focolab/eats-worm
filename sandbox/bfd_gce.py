#
#   (WIP) Illustrating three ways to do GCE analsysis
#   - calling classes/methods (ipynb style)
#   - classic mode (Extractor.calc_blob_threads(), Extractor.extract())
#   - BFD pipeline (not using Extractor)
#
#   TODO: hook up filtersweeper
#

import os

from bfdcore.pipeline import Node, Pipeline
from bfdcore.bfdgui import BFDGUI

from gcamp_extractor import MultiFileTiff
from gcamp_extractor import BlobThreadTracker_alpha
from gcamp_extractor import quantify
from gcamp_extractor import Extractor
from gcamp_extractor import FilterSweeper
from gcamp_extractor import Curator
from gcamp_extractor import Spool
from gcamp_extractor import compute_moco_offsets
from gcamp_extractor import view_moco_offsets

# # class node_filtersweeper(Task):
# #     about = 'filtersweeper'
# #     def main(self, data=None, params=None):
# #         chunk = data['chunk']
# #         sw = FilterSweeper(stacks=[image for image in chunk.data])
# #         sw.sweep_parameters()
# #         sweeper_params = dict(median=sw.median, quantile=sw.quantile, gaussian=sw.gaussian)
# #         return dict(sweeper_params=sweeper_params)


class node_MFT(Node):
    def main(self, data=None, params=None):
        return dict(mft=MultiFileTiff(**params, output_dir=self.loc))

class node_MOCO(Node):
    def main(self, data=None, params=None):
        self.stash = dict(mft=data['mft'])
        return dict(moco_offsets=compute_moco_offsets(mft=data['mft'], **params))
    def inspect(self):
        view_moco_offsets(mft=self.stash['mft'], offsets=self.data['moco_offsets'])

class node_findpeaks(Node):
    def main(self, data=None, params=None):
        btt = BlobThreadTracker_alpha(mft=data['mft'], params=params)
        return dict(spool=btt.calc_blob_threads())

class node_quantify(Node):
    def main(self, data=None, params=None):
        return dict(timeseries=quantify(mft=data['mft'], spool=data['spool']))

class node_curate(Node):
    def main(self, data=None, params=None):
        cc = Curator(mft=data['mft'], spool=data['spool'], timeseries=data['timeseries'])
        return dict()

def make_gce_pipeline(datarun_folder='my_datarun'):
    """make Nodes (incl wiring) and the Pipeline"""
    defaults = dict(frozen=False)
    pn_A = node_MFT(ID='load_MFT', task_params=dict(), save_data=False, **defaults)
    pn_A2 = node_MOCO(ID='MOCO', task_params=dict(), parents=pn_A, save_data=False, **defaults)
    pn_B = node_findpeaks(ID='findpeaks', parents=pn_A, save_data=True, **defaults)
    pn_C = node_quantify(ID='quantify', parents={'mft':pn_A, 'spool':pn_B}, save_data=True, **defaults)
    pn_D = node_curate(ID='curate', parents={'mft':pn_A, 'spool':pn_B, 'timeseries':pn_C}, save_data=False, **defaults)
    return Pipeline(nodes=[pn_A, pn_A2, pn_B, pn_C, pn_D], datarun_folder=datarun_folder)

if __name__ == "__main__":

    FCD = '/home/gbubnis/prj/bfd/FOCO_FCD_v0.1/'

    params_mft = {
        'root':FCD+'RLD_selected_recordings/shortened_recs',
        'numz':10,
        'numc':1,
        #'frames':[1,2,3,4,5,6,7,8,9,10,11],
        'offset': 0,
        'anisotropy':(10,1,1),
    }
    params_pf = {
        't':8,
        'gaussian':(41,5,3,1),
        'quantile':0.99,
        'reg_peak_dist':8,
        'blob_merge_dist_thresh': 8,
        'register_frames':True,
        'predict':True,
        'regen_mft':False,
        '3d':True,        
    }

    print('==================================================================')
    print('=================== classic style ===============================')
    output_dir='gce_output_A'
    e = Extractor(output_dir=output_dir, **params_mft, **params_pf)
    e.calc_blob_threads()
    e.quantify()

    print(e.results_as_dataframe())

    print('==================================================================')
    print('=================== full manual ==================================')
    #### load data
    output_dir = 'gce_output_B'
    mft = MultiFileTiff(output_dir=output_dir, **params_mft)

    #### get threads
    btt = BlobThreadTracker_alpha(mft=mft, params=params_pf)
    spool = btt.calc_blob_threads()
    spool.export(f=os.path.join(output_dir, 'threads.obj'))

    #### quantify
    ts = quantify(mft=mft, spool=spool)

    # #### curate
    # cc = Curator(mft=mft, spool=btt.spool, timeseries=ts)

    print(spool.to_dataframe(dims=['Z', 'Y', 'X']))

    print('==================================================================')
    print('===================== bfd style ==================================')
    pipe = make_gce_pipeline(datarun_folder='gce_output_C')
    pipe.about()

    pipe.id2node['load_MFT'].task_params = params_mft
    pipe.id2node['findpeaks'].task_params = params_pf
    pipe.id2node['MOCO'].task_params = dict(t_max=100)

    xx = BFDGUI(p=pipe).start()
    print(pipe.id2node['findpeaks'].data['spool'].to_dataframe(dims=['Z', 'Y', 'X']))

