"""
==============================
Neuroboros (:mod:`neuroboros`)
==============================

"""

# from . import stats as stats
from . import linalg as linalg
from . import surface as surface
from .surface import Surface, Sphere
from ._version import __version__, __version_tuple__
from .datasets import (
    ID1000,
    Bologna,
    Budapest,
    CamCAN,
    Dalmatians,
    Forrest,
    MonkeyKingdom,
    Raiders,
    SpaceTop,
)

# from . import idm as idm
from .plot2d import brain_plot as plot
from .searchlights import get_searchlights as sls
from .spaces import get_distances as distances
from .spaces import get_geometry as geometry
from .spaces import get_mapping as mapping
from .spaces import get_mask as mask
from .spaces import get_morphometry as morphometry
from .spaces import get_parcellation as parcellation
from .spaces import smooth as smooth
from .utils import load, percentile, save
from .utils import save_results
from .utils import save_results as record
