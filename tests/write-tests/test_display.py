import sys
sys.path.append('/Users/stevenban/Documents/eats_worm/eats_worm')
from Extractor import *
from Threads import *
from Curator import *


e = load_extractor(default_arguments['root'])
c = Curator(e)





