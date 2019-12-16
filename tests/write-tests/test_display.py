import sys
sys.path.append('/Users/stevenban/Documents/gcamp_extractor/gcamp_extractor')
from Extractor import *
from Threads import *
from Curator import *


e = load_extractor(default_arguments['root'])
c = Curator(e)





