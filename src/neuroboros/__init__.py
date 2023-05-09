"""
==============================
Neuroboros (:mod:`neuroboros`)
==============================

"""

from .spaces import get_mask as mask
from .spaces import get_morphometry as morphometry
from .spaces import get_parcellation as parcellation
from .spaces import get_geometry as geometry
from .spaces import get_mapping as mapping
from .spaces import get_distances as distances
from .spaces import smooth as smooth
from .searchlights import get_searchlights as sls
from .datasets import Bologna, Forrest, Dalmatians, SpaceTop, CamCAN, ID1000
from .utils import save, save_results, load, percentile
from .utils import save_results as record
# from . import idm as idm
from .plot2d import brain_plot as plot
# from . import stats as stats
from . import linalg as linalg
from . import surface as surface
