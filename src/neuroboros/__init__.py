"""
==============================
Neuroboros (:mod:`neuroboros`)
==============================

"""

from . import linalg as linalg
from . import stats as stats
from . import surface as surface
from ._version import __version__, __version_tuple__

# from . import idm as idm
from .benchmark import classification
from .datasets import (
    ID1000,
    Bologna,
    Budapest,
    CamCAN,
    Dalmatians,
    Forrest,
    Life,
    MonkeyKingdom,
    Raiders,
    SpaceTop,
)
from .datasets import get_dataset as dataset
from .glm import glm
from .plot2d import brain_plot as plot
from .plot_mebrains import plot_mebrains as plot_mebrains
from .searchlights import get_searchlights as sls
from .spaces import get_distances as distances
from .spaces import get_geometry as geometry
from .spaces import get_mapping as mapping
from .spaces import get_mask as mask
from .spaces import get_morphometry as morphometry
from .spaces import get_parcellation as parcellation
from .spaces import smooth as smooth
from .surface import Sphere, Surface
from .utils import load, percentile, save
from .utils import save_results
from .utils import save_results as record
