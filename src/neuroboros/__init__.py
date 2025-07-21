"""
==============================
Neuroboros (:mod:`neuroboros`)
==============================

"""

from . import ensemble as ensemble
from . import linalg as linalg
from . import stats as stats
from . import surface as surface
from ._version import __version__, __version_tuple__

# from . import idm as idm
from .archive import archive, archive_check
from .benchmark import classification
from .datasets import (
    HBNSSI,
    IBC,
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
    WhiplashC1,
    WhiplashC2,
    WhiplashC3,
)
from .datasets import get_dataset as dataset
from .glm import glm
from .isc import compute_isc as isc
from .plot2d import Image
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
from .stats import pearsonr
from .surface import Sphere, Surface
from .surface.align_surface import Aligner
from .utils import load, percentile, save
from .utils import save_results
from .utils import save_results as record
