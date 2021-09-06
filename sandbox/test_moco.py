#
#   Example of standalone MOCO calculation and visualization in Napari
#

from gcamp_extractor import MultiFileTiff
from gcamp_extractor import view_moco_offsets

if __name__ == "__main__":

    FCD = '/home/gbubnis/prj/bfd/FOCO_FCD_v0.1/'

    ### 2021-07-30/worm06_act1 is a head thrashing drifter
    params_mft = {
        #'root':FCD+'RLD_selected_recordings/shortened_recs',
        #'numz':10,
        # 'root':'/media/gbubnis/Extreme SSD/yakalo/jackson/2021-05-29/worm5_act1/GCAMP',
        # 'numz':12,
        # 'root':'/media/gbubnis/Extreme SSD/yakalo/jackson/2021-07-21/worm15_act1/GCAMP',
        # 'numz':12,
        'root':'/media/gbubnis/Extreme SSD/yakalo/jackson/2021-07-30/worm06_act1/GCAMP',
        'numz':12,
        'numc':1,
        #'frames':[1,2,3,4,5,6,7,8,9,10,11],
        'offset': 0,
        'anisotropy':(10,1,1),
    }

    output_dir = 'gce_output_mocotest'
    mft = MultiFileTiff(output_dir=output_dir, **params_mft)
    view_moco_offsets(mft=mft, t_max=400)

