"""Faster, numerically equivalent replacements for three SOLWEIG-GPU v2.0.0 functions.

- ``fast_sky``: the per-patch anisotropic sky loop as matrix-vector products
  (equal to float32 rounding; summation order differs).
- ``fast_gvf``: ground view factor steps without per-step CPU-GPU syncs (bit-identical).
- ``fast_shadow``: shadow casting steps without per-step syncs (bit-identical).

Safeguards:
- Each replacement is tied to a SHA-256 fingerprint of the exact original source it was
  written and verified against. A different SOLWEIG-GPU version makes it refuse to run.
- TF32 matrix maths is switched off while the fast sky runs, so a library that enables it
  globally cannot silently lower its precision.

Usage::

    from solweig_sens.fast import fast_solweig
    with fast_solweig():
        run_utci_tiles(...)

Verification: ``scripts/verify_fast.py`` (every call compared with the original, sun-position
sweep, end-to-end runs over several configurations).
"""

from __future__ import annotations

import contextlib
import hashlib
import inspect
from typing import Iterator

FINGERPRINTS = {
    "define_patch_characteristics": "0c3d24183c8e8b147c56812a07f3ec09fae35dcd9798ce1e808655033f935d80",
    "sunonsurface_2018a": "0f8a92e8c6a16fea174de91eba49a162090402e634bea026437c08361a1acbe2",
    "shadowingfunction_wallheight_23": "6f6b47e51f43ac1e0ba37fbd5e55143a113f45e2e7f868a067de6cc98b0b5854",
}


def check_fingerprint(name: str) -> None:
    import solweig_gpu.solweig as sw

    digest = hashlib.sha256(inspect.getsource(getattr(sw, name)).encode()).hexdigest()
    if digest != FINGERPRINTS[name]:
        raise RuntimeError(f"solweig_gpu.solweig.{name} differs from the verified version; "
                           "re-verify the fast replacement before using it")


def build_fast_functions() -> dict:
    """The three replacement functions, after checking the originals' fingerprints."""
    from . import fast_gvf, fast_shadow, fast_sky

    for name in FINGERPRINTS:
        check_fingerprint(name)
    return {
        "define_patch_characteristics": fast_sky.define_patch_characteristics,
        "sunonsurface_2018a": fast_gvf.build(),
        "shadowingfunction_wallheight_23": fast_shadow.build(),
    }


@contextlib.contextmanager
def no_tf32() -> Iterator[None]:
    import torch

    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old


@contextlib.contextmanager
def fast_solweig(sky: bool = True, gvf: bool = True, shadow: bool = True) -> Iterator[None]:
    """Swap the replacements into solweig_gpu.solweig for the duration of a run."""
    import solweig_gpu.solweig as sw

    funcs = build_fast_functions()
    chosen = {k: v for k, v in funcs.items()
              if {"define_patch_characteristics": sky, "sunonsurface_2018a": gvf,
                  "shadowingfunction_wallheight_23": shadow}[k]}
    old = {k: getattr(sw, k) for k in chosen}
    with no_tf32():
        try:
            for k, v in chosen.items():
                setattr(sw, k, v)  # callers in the module look these up at call time
            yield
        finally:
            for k, v in old.items():
                setattr(sw, k, v)
