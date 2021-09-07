#
#   Example of standalone MOCO calculation and visualization in Napari
#

from gcamp_extractor import MultiFileTiff
from gcamp_extractor import compute_moco_offsets
from gcamp_extractor import view_moco_offsets

if __name__ == "__main__":
    ykroot = '/media/gbubnis/Extreme SSD/yakalo/'
    params_mft = {
        ## three worms that drift
        # 'root':ykroot+'jackson/2021-07-30/worm04_act1/GCAMP',
        'root':ykroot+'jackson/2021-07-30/worm06_act1/GCAMP',
        # 'root':ykroot+'jackson/2021-07-31/worm01_act1/GCAMP',
        'numz':12,
        'numc':1,
        'offset': 0,
        'anisotropy':(10,1,1),
    }

    output_dir = 'gce_output_mocotest'
    mft = MultiFileTiff(output_dir=output_dir, **params_mft)
    offsets = compute_moco_offsets(mft=mft, t_max=500, suppress_output=False, mode='mid')
    view_moco_offsets(mft=mft, offsets=offsets)
