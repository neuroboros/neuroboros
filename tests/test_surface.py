import os

import pytest

from neuroboros.surface import Surface


class TestSurface:
    def test_freesurfer(self):
        fn = "data/lh.pial"
        if not os.path.exists(fn):
            pytest.skip(f"Skipping test of FreeSurfer I/O because {fn} does not exist.")
        surf = Surface.from_file(fn)
        fn2 = "data/lh.pial.2"
        surf.to_fs(fn2)
        surf2 = Surface.from_file(fn2)
        assert surf == surf2

    def test_gifti(self):
        fn = "data/hemi-L_pial.surf.gii"
        if not os.path.exists(fn):
            pytest.skip(f"Skipping test of GIFTI I/O because {fn} does not exist.")
        surf = Surface.from_file(fn)
        fn2 = "data/hemi-L_pial.surf.gii"
        surf.to_gifti(fn2)
        surf2 = Surface.from_file(fn2)
        assert surf == surf2
