import hashlib
import io
import json
import os
import warnings

import numpy as np
import pytest
import requests
from scipy import sparse

import neuroboros as nb
from neuroboros import spaces, templates
from neuroboros.io import load

MANIFEST = {
    "updated": "2026-11-01",
    "files": [
        "sparta/v0.9/lh_onavg-ico64.npy",
        "sparta/v0.9/meta.json",
        "sparta/v0.9/rh_onavg-ico64.npy",
        "sparta/v1.0/lh_onavg-ico64.npy",
    ],
}
SHA256 = {
    "lh": "b49c66d76c2a75013f59cb08fc7a68b4effb315a60141eb8df46fb229b711407",
    "rh": "a2cf50c0a4ba9d2c6c91446bae83a0990e31578b82d5d5096cae547b9a2ac535",
}


class FakeDataset:
    """Stands in for ``hyperbrain_dataset``; records requested paths.

    Like the real manager, it writes a requested file under ``root`` when
    it is absent and loads it from there. Its manifest carries the correct
    sha256 of each file unless ``manifest`` is modified before download.
    """

    kind = "https"

    def __init__(self, root, shape=(3, 4)):
        self.root = str(root)
        self.shape = shape
        self.calls = []
        self.downloads = []
        self.manifest = dict(MANIFEST)
        self.manifest["sha256"] = {
            rel: hashlib.sha256(self.content(rel)).hexdigest()
            for rel in MANIFEST["files"]
        }

    def _manifest_fn(self):
        return os.path.join(self.root, templates.MANIFEST_FN)

    def local_fn(self, rel):
        return os.path.join(self.root, *rel.split("/"))

    def content(self, rel):
        """Bytes of a file as the fake remote serves it."""
        parts = rel.split("/")
        if parts[-1] == "meta.json":
            return json.dumps({"name": "Sparta", "version": parts[1][1:]}).encode()
        offset = 0.0 if parts[-1].startswith("lh") else 100.0
        arr = np.arange(np.prod(self.shape), dtype=np.float32).reshape(self.shape)
        buf = io.BytesIO()
        np.save(buf, arr + offset)
        return buf.getvalue()

    def download(self, fn, local_fn, on_missing="warn"):
        self.downloads.append(fn)
        os.makedirs(os.path.dirname(local_fn), exist_ok=True)
        if fn == templates.MANIFEST_FN:
            with open(local_fn, "w") as f:
                json.dump(self.manifest, f)
        else:
            with open(local_fn, "wb") as f:
                f.write(self.content("/".join(fn)))

    def get(self, fn, load_func=None, on_missing="warn"):
        self.calls.append(fn)
        if fn == templates.MANIFEST_FN:
            local_fn = self._manifest_fn()
        else:
            local_fn = self.local_fn("/".join(fn))
        if not os.path.exists(local_fn):
            self.download(fn, local_fn, on_missing=on_missing)
        return load(local_fn)


@pytest.fixture
def fake(tmp_path, monkeypatch):
    ds = FakeDataset(tmp_path)
    monkeypatch.setattr(templates, "hyperbrain_dataset", ds)
    monkeypatch.setattr(templates, "_manifest", None)
    return ds


def array_calls(ds):
    return [c for c in ds.calls if c != templates.MANIFEST_FN]


def test_public_alias():
    assert nb.tpl is templates.get_template
    assert nb.tpl.update is templates.update_manifest
    assert nb.tpl.verify is templates.verify_manifest


class TestPaths:
    @pytest.mark.parametrize("which", ["l", "lh"])
    def test_left(self, fake, which):
        templates.get_template(which, mask=False)
        assert array_calls(fake) == [("sparta", "v0.9", "lh_onavg-ico64.npy")]

    @pytest.mark.parametrize("which", ["r", "rh"])
    def test_right(self, fake, which):
        templates.get_template(which, mask=False)
        assert array_calls(fake) == [("sparta", "v0.9", "rh_onavg-ico64.npy")]

    def test_meta(self, fake):
        meta = templates.get_template("meta")
        assert meta == {"name": "Sparta", "version": "0.9"}
        assert array_calls(fake) == [("sparta", "v0.9", "meta.json")]

    def test_version(self, fake):
        templates.get_template("l", mask=False, version="sparta-1.0")
        assert array_calls(fake) == [("sparta", "v1.0", "lh_onavg-ico64.npy")]

    def test_manifest_fetched_once(self, fake):
        templates.get_template("l", mask=False)
        templates.get_template("r", mask=False)
        assert fake.calls.count(templates.MANIFEST_FN) == 1


class TestValues:
    def test_unmasked_native_dtype(self, fake):
        tpl = templates.get_template("l", mask=False, dtype=None)
        expected = np.arange(12, dtype=np.float32).reshape(3, 4)
        assert tpl.dtype == np.float32
        np.testing.assert_array_equal(tpl, expected)

    def test_default_dtype(self, fake):
        tpl = templates.get_template("l", mask=False)
        assert tpl.dtype == np.float64
        np.testing.assert_array_equal(tpl, np.arange(12).reshape(3, 4))

    def test_lr(self, fake):
        tpl = templates.get_template("lr", mask=False)
        lh = np.arange(12).reshape(3, 4)
        np.testing.assert_array_equal(tpl, np.concatenate([lh, lh + 100], axis=1))
        assert array_calls(fake) == [
            ("sparta", "v0.9", "lh_onavg-ico64.npy"),
            ("sparta", "v0.9", "rh_onavg-ico64.npy"),
        ]

    def test_masked(self, tmp_path, monkeypatch):
        ds = FakeDataset(tmp_path, shape=(1284, 40962))
        monkeypatch.setattr(templates, "hyperbrain_dataset", ds)
        monkeypatch.setattr(templates, "_manifest", None)
        tpl = templates.get_template("l")
        n_targets = nb.mask("lr", "onavg-ico8").sum()
        n_vertices = nb.mask("l", "onavg-ico64").sum()
        assert tpl.shape == (n_targets, n_vertices)
        assert tpl.dtype == np.float64


class FakeMapping:
    """Stands in for ``get_mapping``: sums adjacent column pairs (4 -> 2)."""

    def __init__(self):
        self.calls = []

    def __call__(self, lr, source, target, mask=None):
        self.calls.append((lr, source, target, mask))
        return sparse.csr_matrix(
            np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=np.float64)
        )


@pytest.fixture
def fake_mapping(monkeypatch):
    fm = FakeMapping()
    monkeypatch.setattr(templates, "get_mapping", fm)
    return fm


class TestSpace:
    def test_default_space_does_not_resample(self, fake, fake_mapping):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            templates.get_template("l", mask=False)
        assert fake_mapping.calls == []

    def test_resample(self, fake, fake_mapping):
        with pytest.warns(UserWarning, match="forces resampling"):
            tpl = templates.get_template("l", mask=False, space="onavg-ico32")
        lh = np.arange(12).reshape(3, 4)
        expected = np.stack([lh[:, :2].sum(axis=1), lh[:, 2:].sum(axis=1)], axis=1)
        np.testing.assert_array_equal(tpl, expected)
        assert tpl.dtype == np.float64
        assert fake_mapping.calls == [("l", "onavg-ico64", "onavg-ico32", None)]

    def test_resample_native_dtype(self, fake, fake_mapping):
        with pytest.warns(UserWarning):
            tpl = templates.get_template(
                "r", mask=False, space="onavg-ico32", dtype=None
            )
        assert tpl.dtype == np.float32
        assert tpl.shape == (3, 2)

    def test_resample_lr(self, fake, fake_mapping):
        with pytest.warns(UserWarning):
            tpl = templates.get_template("lr", mask=False, space="onavg-ico32")
        assert tpl.shape == (3, 4)
        assert [c[0] for c in fake_mapping.calls] == ["l", "r"]

    def test_resample_meta_ignores_space(self, fake, fake_mapping):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            meta = templates.get_template("meta", space="onavg-ico32")
        assert meta["name"] == "Sparta"
        assert fake_mapping.calls == []

    def test_resample_masked(self, tmp_path, monkeypatch):
        ds = FakeDataset(tmp_path, shape=(1284, 40962))
        monkeypatch.setattr(templates, "hyperbrain_dataset", ds)
        monkeypatch.setattr(templates, "_manifest", None)
        calls = []

        def recording_mapping(*args, **kwargs):
            calls.append((args, kwargs))
            return spaces.get_mapping(*args, **kwargs)

        monkeypatch.setattr(templates, "get_mapping", recording_mapping)
        with pytest.warns(UserWarning):
            tpl = templates.get_template("l", space="onavg-ico32")
        n_targets = nb.mask("lr", "onavg-ico8").sum()
        n_vertices = nb.mask("l", "onavg-ico32").sum()
        assert tpl.shape == (n_targets, n_vertices)
        assert tpl.dtype == np.float64
        assert calls == [(("l", "onavg-ico64", "onavg-ico32"), {"mask": True})]


class TestErrors:
    @pytest.mark.parametrize("which", ["x", "both", "L"])
    def test_bad_which(self, fake, which):
        with pytest.raises(ValueError, match="Invalid value for `which`"):
            templates.get_template(which)

    def test_bad_version_string(self, fake):
        with pytest.raises(ValueError, match="Invalid template version"):
            templates.get_template("l", version="sparta")

    def test_missing_version_does_not_download(self, fake):
        with pytest.raises(ValueError, match="sparta-0.9"):
            templates.get_template("l", version="sparta-9.9")
        assert array_calls(fake) == []

    def test_missing_hemisphere_does_not_download(self, fake):
        # v1.0 in the fake manifest only has the left hemisphere.
        with pytest.raises(ValueError, match="not in the hyperbrain manifest"):
            templates.get_template("r", version="sparta-1.0")
        assert array_calls(fake) == []


class TestUpdate:
    def test_update_refetches(self, fake):
        stale = {"files": ["sparta/v0.9/lh_onavg-ico64.npy"]}
        with open(fake._manifest_fn(), "w") as f:
            json.dump(stale, f)
        assert templates.load_manifest() == set(stale["files"])
        with pytest.raises(ValueError):
            templates.get_template("r", mask=False)

        manifest = templates.get_template.update()
        assert manifest == fake.manifest
        assert templates.load_manifest() == set(MANIFEST["files"])
        templates.get_template("r", mask=False)
        with open(fake._manifest_fn()) as f:
            assert json.load(f) == fake.manifest
        assert not os.path.exists(fake._manifest_fn() + ".new")

    def test_failed_update_keeps_old_manifest(self, fake, monkeypatch):
        stale = {"files": ["sparta/v0.9/lh_onavg-ico64.npy"]}
        with open(fake._manifest_fn(), "w") as f:
            json.dump(stale, f)
        assert templates.load_manifest() == set(stale["files"])
        templates.get_template("l", mask=False)

        def failing_download(fn, local_fn, on_missing="warn"):
            raise RuntimeError("network down")

        monkeypatch.setattr(fake, "download", failing_download)
        with pytest.raises(RuntimeError, match="network down"):
            templates.get_template.update()
        with open(fake._manifest_fn()) as f:
            assert json.load(f) == stale
        assert templates.load_manifest() == set(stale["files"])
        templates.get_template("l", mask=False)

    @pytest.mark.parametrize(
        "content, error",
        [
            ("not json", ValueError),
            ('{"version": 1}', ValueError),
            ("[]", ValueError),
            ('{"files": [], "sha256": []}', ValueError),
            ('{"files": [], "sha256": {"a": null}}', ValueError),
            ('{"files": [], "sha256": {"a": "abc"}}', ValueError),
            ('{"files": [], "sha256": {"a": "' + "g" * 64 + '"}}', ValueError),
        ],
    )
    def test_invalid_update_keeps_old_manifest(self, fake, monkeypatch, content, error):
        stale = {"files": ["sparta/v0.9/lh_onavg-ico64.npy"]}
        with open(fake._manifest_fn(), "w") as f:
            json.dump(stale, f)
        assert templates.load_manifest() == set(stale["files"])

        def bad_download(fn, local_fn, on_missing="warn"):
            with open(local_fn, "w") as f:
                f.write(content)

        monkeypatch.setattr(fake, "download", bad_download)
        with pytest.raises(error):
            templates.get_template.update()
        assert not os.path.exists(fake._manifest_fn() + ".new")
        with open(fake._manifest_fn()) as f:
            assert json.load(f) == stale
        assert templates.load_manifest() == set(stale["files"])


LH = "sparta/v0.9/lh_onavg-ico64.npy"
META = "sparta/v0.9/meta.json"


def corrupt(fn):
    """Overwrite a cached file with different but loadable content."""
    data = load(fn)
    if fn.endswith(".npy"):
        np.save(fn, data + 1)
    else:
        with open(fn, "w") as f:
            json.dump(dict(data, corrupted=True), f)


class TestChecksums:
    def test_download_is_verified(self, fake, monkeypatch):
        hashed = []
        sha256 = templates._sha256
        monkeypatch.setattr(
            templates, "_sha256", lambda fn: hashed.append(fn) or sha256(fn)
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            templates.get_template("l", mask=False)
        assert hashed == [fake.local_fn(LH)]

    def test_corrupted_download_warns(self, fake):
        fake.manifest["sha256"][LH] = "0" * 64
        with pytest.warns(UserWarning, match="Checksum mismatch") as record:
            tpl = templates.get_template("l", mask=False)
        assert tpl.shape == (3, 4)
        message = str(record[0].message)
        assert fake.local_fn(LH) in message
        assert "0" * 64 in message
        assert fake.manifest["sha256"][META] not in message
        assert "Delete the file" in message

    def test_unreadable_download_still_warns(self, fake, monkeypatch):
        monkeypatch.setattr(fake, "content", lambda rel: b"not a numpy file")
        with pytest.warns(UserWarning, match="Checksum mismatch"):
            with pytest.raises((ValueError, OSError)):
                templates.get_template("l", mask=False)
        assert fake.downloads == [templates.MANIFEST_FN, tuple(LH.split("/"))]

    def test_uppercase_digest_matches(self, fake):
        fake.manifest["sha256"][LH] = fake.manifest["sha256"][LH].upper()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            templates.get_template("l", mask=False)
            assert templates.get_template.verify() == {LH: True}

    @pytest.mark.parametrize(
        "kind, expected, unexpected",
        [
            ("https", "Delete the file", "datalad"),
            ("datalad", f"git annex fsck {LH}", "Delete the file"),
            ("local", "nb.tpl.update()", "Delete the file"),
        ],
    )
    def test_recovery_advice(self, fake, kind, expected, unexpected):
        fake.kind = kind
        fake.manifest["sha256"][LH] = "0" * 64
        with pytest.warns(UserWarning) as record:
            templates.get_template("l", mask=False)
        message = str(record[0].message)
        assert expected in message
        assert unexpected not in message

    def test_cached_file_is_not_hashed(self, fake, monkeypatch):
        templates.get_template("l", mask=False)
        corrupt(fake.local_fn(LH))

        def no_hashing(fn):
            raise AssertionError(f"unexpected hashing of {fn}")

        monkeypatch.setattr(templates, "_sha256", no_hashing)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            templates.get_template("l", mask=False)

    def test_verify_checks_downloaded_files(self, fake):
        assert templates.get_template.verify() == {}
        templates.get_template("l", mask=False)
        templates.get_template("meta")
        assert templates.get_template.verify() == {LH: True, META: True}

        corrupt(fake.local_fn(LH))
        with pytest.warns(UserWarning, match="Checksum mismatch") as record:
            results = templates.get_template.verify()
        assert results == {LH: False, META: True}
        assert len(record) == 1
        assert fake.local_fn(LH) in str(record[0].message)

    def test_manifest_without_checksums(self, fake, monkeypatch):
        del fake.manifest["sha256"]

        def no_hashing(fn):
            raise AssertionError(f"unexpected hashing of {fn}")

        monkeypatch.setattr(templates, "_sha256", no_hashing)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            templates.get_template("l", mask=False)
            assert templates.get_template.verify() == {}

    def test_missing_entry_is_skipped(self, fake):
        del fake.manifest["sha256"][LH]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            templates.get_template("l", mask=False)
            templates.get_template("meta")
            assert templates.get_template.verify() == {META: True}


@pytest.mark.skipif(
    not os.environ.get("NEUROBOROS_NETWORK_TESTS"),
    reason="set NEUROBOROS_NETWORK_TESTS=1 to query GIN",
)
def test_remote_manifest_and_checksums():
    base = "https://gin.g-node.org/neuroboros/hyperbrain/raw/master/"
    manifest = requests.get(base + "manifest.json", timeout=30).json()
    for fn in [
        "sparta/v0.9/lh_onavg-ico64.npy",
        "sparta/v0.9/rh_onavg-ico64.npy",
        "sparta/v0.9/meta.json",
    ]:
        assert fn in manifest["files"]
    meta = requests.get(base + "sparta/v0.9/meta.json", timeout=30).json()
    for lr in ["lh", "rh"]:
        assert meta["files"][lr]["sha256"] == SHA256[lr]
        assert manifest["sha256"][f"sparta/v0.9/{lr}_onavg-ico64.npy"] == SHA256[lr]
