"""Compact sky-patch masks must give exactly the values the original float32 arrays give."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

torch = pytest.importorskip("torch")
up = pytest.importorskip("solweig_gpu.utci_process")

from solweig_sens.compact_masks import (FINGERPRINTS, LazyDiffsh, MaskFile, PackedMask, compact_mask,  # noqa: E402
                                        compact_masks, patched_sources)


def _masks(seed=0, shape=(17, 23, 153)):
    rng = np.random.default_rng(seed)
    return [(rng.random(shape) > p).astype(np.float32) for p in (0.3, 0.2, 0.25)]


def test_packed_mask_indexing_matches_original():
    s, _, _ = _masks()
    pm = PackedMask.from_numpy(s, "cpu")
    ref = torch.from_numpy(s)
    assert pm.shape == s.shape and pm.nbytes() == s.shape[0] * s.shape[1] * 20
    for i in [0, 1, 7, 8, 9, 63, 64, 100, 151, 152, torch.tensor(5)]:
        assert torch.equal(pm[:, :, i].float(), ref[:, :, int(i)])
    for sl in [slice(0, 8), slice(3, 11), slice(144, 153), slice(0, 153), slice(150, 160)]:
        assert torch.equal(pm[:, :, sl].float(), ref[:, :, sl])


@pytest.mark.parametrize("mode", ["uint8", "bits"])
def test_lazy_diffsh_bit_identical(mode):
    s, v, _ = _masks(1)
    transVeg = 3.0 / 100.0
    S, V = torch.from_numpy(s), torch.from_numpy(v)
    original = torch.zeros(S.shape)
    for i in range(S.shape[2]):  # exactly the original construction
        original[:, :, i] = S[:, :, i] - (1 - V[:, :, i]) * (1 - transVeg)
    lazy = LazyDiffsh(compact_mask(s, "cpu", mode), compact_mask(v, "cpu", mode), transVeg)
    for i in range(S.shape[2]):
        assert torch.equal(lazy[:, :, i], original[:, :, i])


@pytest.mark.parametrize("mode", ["uint8", "bits"])
def test_mask_expressions_used_by_solweig_are_exact(mode):
    s, v, b = _masks(2)
    S, V, B = (torch.from_numpy(x) for x in (s, v, b))
    cS, cV, cB = (compact_mask(x, "cpu", mode) for x in (s, v, b))
    for i in (0, 37, 152):
        assert torch.equal(((S[:, :, i] == 1) & (V[:, :, i] == 1)), ((cS[:, :, i] == 1) & (cV[:, :, i] == 1)))
        assert torch.equal(((V[:, :, i] == 0) | (B[:, :, i] == 0)), ((cV[:, :, i] == 0) | (cB[:, :, i] == 0)))
        assert torch.equal(((1 - S[:, :, i]) * B[:, :, i]) == 1, ((1 - cS[:, :, i]) * cB[:, :, i]) == 1)
        assert torch.equal(S[:, :, i] * 0.37, cS[:, :, i] * 0.37)


def test_packed_mask_returns_float32_so_arithmetic_matches_original():
    s, _, _ = _masks(6)
    pm = PackedMask.from_numpy(s, "cpu")
    ref = torch.from_numpy(s)
    assert pm[:, :, 3].dtype == torch.float32 and pm[:, :, 0:8].dtype == torch.float32
    assert torch.equal(pm[:, :, 3] - 1, ref[:, :, 3] - 1)  # uint8 would wrap 0 - 1 to 255
    assert torch.equal(pm[:, :, 0:8] - 1, ref[:, :, 0:8] - 1)


def test_non_binary_mask_falls_back_to_float32():
    s, _, _ = _masks()
    s[0, 0, 0] = 0.5
    with pytest.warns(RuntimeWarning, match="other than 0 and 1"):
        out = compact_mask(s, "cpu", "bits")
    assert isinstance(out, torch.Tensor) and out.dtype == torch.float32 and out[0, 0, 0] == 0.5


def _svf_inputs(n=40):
    g = torch.Generator().manual_seed(3)
    dem = 100.0 + torch.rand((n, n), generator=g)
    bld = torch.zeros((n, n))
    bld[6:14, 8:18] = 9.0
    bld[25:33, 20:26] = 18.0
    a = dem + bld
    cdsm = torch.zeros((n, n))
    cdsm[4:9, 26:32] = 7.0
    cdsm[30:37, 4:10] = 11.0
    vegdem_height = cdsm + dem
    vegdem2_height = cdsm * 0.25 + dem
    bush = torch.logical_not(vegdem2_height * vegdem_height) * vegdem_height
    vegdem = cdsm + a
    vegdem[vegdem == a] = 0.0
    vegdem2 = cdsm * 0.25 + a
    vegdem2[vegdem2 == a] = 0.0
    amax = torch.maximum(a.max(), vegdem_height.max())
    return dict(amaxvalue=amax, a=a, vegdem=vegdem, vegdem2=vegdem2, bush=bush, scale=0.5)


@pytest.mark.parametrize("mode", ["uint8", "bits"])
def test_svf_calculator_compact_matches_original(mode):
    import solweig_gpu.shadow as sh

    inputs = _svf_inputs()
    original = sh.svf_calculator(2, **inputs)
    with compact_masks(mode):
        compact = sh.svf_calculator(2, **inputs)
    assert sh.svf_calculator is not None and len(original) == len(compact) == 19
    for x, y in zip(original, compact):
        if isinstance(y, PackedMask):
            y = y[:, :, :]
        assert torch.equal(x.float(), y.float())


def test_mask_builder_saves_exactly_what_the_original_saves():
    from solweig_sens.compact_masks import MaskBuilder

    s, _, _ = _masks(4)
    for mode in ("uint8", "bits"):
        mb = MaskBuilder(s.shape[0], s.shape[1], s.shape[2], "cpu", mode)
        for i in range(s.shape[2]):
            mb[:, :, i] = torch.from_numpy(s[:, :, i])
        assert np.array_equal(mb.saveable().numpy().astype(np.float32), s)


def test_mask_builder_falls_back_to_float32_for_non_binary_patches():
    from solweig_sens.compact_masks import MaskBuilder

    s, _, _ = _masks(5)
    s[:, :, 10] *= 2.0  # a patch with values 0 and 2
    mb = MaskBuilder(s.shape[0], s.shape[1], s.shape[2], "cpu", "bits")
    with pytest.warns(RuntimeWarning, match="other than 0 and 1"):
        for i in range(s.shape[2]):
            mb[:, :, i] = torch.from_numpy(s[:, :, i])
    out = mb.final()
    assert isinstance(out, torch.Tensor) and out.dtype == torch.float32
    assert np.array_equal(out.numpy(), s)


def test_patches_apply_and_restore():
    assert set(patched_sources("bits")) == set(FINGERPRINTS)
    originals = {n: getattr(up, n) for n in FINGERPRINTS}
    with compact_masks("bits"):
        for n in FINGERPRINTS:
            assert getattr(up, n) is not originals[n]
            assert getattr(up, n).__globals__ is vars(up)
    for n, f in originals.items():
        assert getattr(up, n) is f
    assert "_compact_mask" not in vars(up) and "_LazyDiffsh" not in vars(up) and "_MaskFile" not in vars(up)
    import solweig_gpu.shadow as sh
    assert up.svf_calculator is sh.svf_calculator and "_MaskBuilder" not in vars(sh)


KEYS = ("shadowmat", "vegshadowmat", "vbshmat")


def _cache_file(path, masks):
    np.savez_compressed(path, **dict(zip(KEYS, masks)))  # as SOLWEIG-GPU writes it: float32 0/1
    return path


def _loaded(path, mode, packed_cache=False):
    with MaskFile(path, mode, packed_cache) as f:
        return [compact_mask(f[k], "cpu", mode) for k in KEYS]


@pytest.mark.parametrize("mode", ["uint8", "bits"])
def test_mask_file_loads_what_np_load_loads(tmp_path, mode):
    masks = _masks(3)
    path = _cache_file(tmp_path / "shadowmats_0_0.npz", masks)
    for got, ref in zip(_loaded(path, mode), masks):
        expected = compact_mask(ref, "cpu", mode)
        if mode == "bits":
            assert torch.equal(got.packed, expected.packed) and got.n == expected.n
        else:
            assert torch.equal(got, expected)
    assert not (tmp_path / "shadowmats_0_0.bits.npz").exists()  # no copy unless asked for


def test_packed_cache_is_written_then_read_and_gives_the_same_bits(tmp_path, monkeypatch):
    masks = _masks(4)
    path = _cache_file(tmp_path / "shadowmats_0_0.npz", masks)
    first = _loaded(path, "bits", packed_cache=True)
    assert (tmp_path / "shadowmats_0_0.bits.npz").exists()
    monkeypatch.setattr(MaskFile, "_load", lambda self, key: pytest.fail("decompressed the float32 file"))
    second = _loaded(path, "bits", packed_cache=True)
    for a, b, ref in zip(first, second, masks):
        assert torch.equal(a.packed, b.packed) and a.n == b.n == 153
        assert torch.equal(b[:, :, :], torch.from_numpy(ref))


def test_packed_cache_is_rebuilt_when_the_source_changes(tmp_path):
    path = _cache_file(tmp_path / "shadowmats_0_0.npz", _masks(5))
    _loaded(path, "bits", packed_cache=True)
    new = _masks(6)
    _cache_file(path, new)  # e.g. the sky view factor was recomputed
    for got, ref in zip(_loaded(path, "bits", packed_cache=True), new):
        assert torch.equal(got[:, :, :], torch.from_numpy(ref))


def test_non_binary_masks_are_not_packed_or_cached(tmp_path):
    masks = _masks(7)
    masks[1][2, 3, 4] = 0.5
    path = _cache_file(tmp_path / "shadowmats_0_0.npz", masks)
    with pytest.warns(RuntimeWarning, match="other than 0 and 1"):
        got = _loaded(path, "bits", packed_cache=True)
    assert isinstance(got[1], torch.Tensor) and got[1][2, 3, 4] == 0.5
    assert isinstance(got[0], PackedMask)
    assert not (tmp_path / "shadowmats_0_0.bits.npz").exists()


@pytest.mark.parametrize("mode", ["uint8", "bits"])
def test_mask_file_decompresses_in_parallel_only_when_memory_allows(tmp_path, monkeypatch, mode):
    import solweig_sens.compact_masks as cm

    masks = _masks(8)
    path = _cache_file(tmp_path / "shadowmats_0_0.npz", masks)
    one_mask = masks[0].nbytes
    results = {}
    for available, workers in [(10 * one_mask, 3), (4 * one_mask, 1), (None, 1)]:
        monkeypatch.setattr(cm, "_available_bytes", lambda a=available: a)
        with MaskFile(path, mode) as f:
            assert f.workers == workers
            results[available] = [compact_mask(f[k], "cpu", mode) for k in KEYS]
    first, *others = results.values()
    for other in others:  # same masks whichever way they were decompressed
        for a, b in zip(first, other):
            assert torch.equal(a.packed, b.packed) if mode == "bits" else torch.equal(a, b)


def test_load_patch_opens_the_cache_through_mask_file():
    assert '_MaskFile(npz_path, "bits", True)' in patched_sources("bits", packed_cache=True)["load_cached_svf_outputs"]
    assert '_MaskFile(npz_path, "bits", False)' in patched_sources("bits")["load_cached_svf_outputs"]
