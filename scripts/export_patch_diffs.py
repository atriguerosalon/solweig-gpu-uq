"""Write a unified diff of every SOLWEIG-GPU function this repo changes: original source against ours.

Generated from the code itself (the same patched sources the context managers compile), so the
diffs cannot drift from what actually runs. Output: docs/patches/<change>_<function>.diff.

Usage:
    python scripts/export_patch_diffs.py
"""

from __future__ import annotations

import sys as _sys

if __name__ == "__main__" and any(a in ("-h", "--help") for a in _sys.argv[1:]):
    print(__doc__)
    raise SystemExit(0)

import difflib
import inspect
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    import solweig_gpu.shadow as shadow_mod
    import solweig_gpu.solweig as sw
    import solweig_gpu.utci_process as up

    from solweig_sens import compact_masks, fast_gvf, fast_shadow, fast_sky, fast_sky_shadow, heat_storage

    def orig(fn):
        return textwrap.dedent(inspect.getsource(fn))

    items = [
        ("B1", "Solweig_2022a_calc", orig(sw.Solweig_2022a_calc), heat_storage.patched_source(),
         "ground heat kept after sunset (experiment only)"),
        ("C1", "define_patch_characteristics", orig(sw.define_patch_characteristics),
         orig(fast_sky.define_patch_characteristics), "full replacement: sky patches as chunked matrix products"),
        ("C2", "sunonsurface_2018a", orig(sw.sunonsurface_2018a), fast_gvf.patched_source(),
         "ground view factor steps without per-step syncs"),
        ("C3", "shadowingfunction_wallheight_23", orig(sw.shadowingfunction_wallheight_23),
         fast_shadow.patched_source(), "shadow casting steps without per-step syncs"),
    ]
    c4 = compact_masks.patched_sources("bits")
    items += [
        ("C4", "load_cached_svf_outputs", orig(up.load_cached_svf_outputs), c4["load_cached_svf_outputs"],
         "load cached sky-patch masks compactly"),
        ("C4", "compute_utci", orig(up.compute_utci), c4["compute_utci"], "diffsh computed on demand"),
        ("C5", "svf_calculator", orig(shadow_mod.svf_calculator), compact_masks.patched_svf_source("bits"),
         "build sky-patch masks compactly in the sky view factor step"),
        ("C7", "shadow", orig(shadow_mod.shadow), fast_sky_shadow.patched_source(),
         "sky-patch shadow steps without per-step syncs"),
    ]
    out_dir = ROOT / "docs" / "patches"
    out_dir.mkdir(parents=True, exist_ok=True)
    for change, name, a, b, why in items:
        diff = difflib.unified_diff(a.splitlines(keepends=True), b.splitlines(keepends=True),
                                    fromfile=f"solweig_gpu v2.0.0: {name} (original)",
                                    tofile=f"solweig-sensitivity {change}: {name} ({why})")
        text = "".join(diff)
        path = out_dir / f"{change}_{name}.diff"
        path.write_text(text)
        added = sum(1 for l in text.splitlines() if l.startswith("+") and not l.startswith("+++"))
        removed = sum(1 for l in text.splitlines() if l.startswith("-") and not l.startswith("---"))
        print(f"{path.relative_to(ROOT)}: +{added} -{removed}")


if __name__ == "__main__":
    main()
