"""Experimental patch: let the ground keep some of its daytime heat after sunset.

In SOLWEIG 2022a (as run by SOLWEIG-GPU), ground upwelling longwave at night is
``SBC * emis * (Ta + 273.15)**4``: the ground drops to air temperature the moment the
sun sets, so heat stored in pavement during the day is lost. SOLWEIG 2026a adds a
force-restore ground scheme for this; SOLWEIG-GPU does not have it yet.

This patch is a deliberately simple stand-in. At every daytime step it records, per
pixel, the excess of the ground's upwelling longwave over that of a ground at air
temperature. At night it adds that excess back, decaying as exp(-hours since the last
daytime step / tau). Directional upwelling fluxes at night are set equal to it, as
SOLWEIG already does. Walls are not changed.

It works by compiling an edited copy of ``Solweig_2022a_calc`` and swapping it into
``solweig_gpu.utci_process`` inside a context manager; the installed package is not
modified. One tile at a time only (the state is not keyed by tile).
"""

from __future__ import annotations

import contextlib
import inspect
import math
import textwrap
from typing import Iterator

NIGHT_LUP = "Lup = SBC * emis_grid * ((Knight + Ta + Tg + 273.15) ** 4)"
DAY_LUPN = "LupN, timeaddnotused, Tgmap1N = TsWaveDelay_2015a(gvfLupN, firstdaytime, timeadd, timestepdec, Tgmap1N)"


class _State:
    def __init__(self, tau_hours: float) -> None:
        self.tau = tau_hours
        self.excess = None
        self.hours_since_day = 0

    def day(self, Lup, SBC, emis_grid, Ta) -> None:
        self.excess = (Lup - SBC * emis_grid * ((Ta + 273.15) ** 4)).clamp(min=0).detach().clone()
        self.hours_since_day = 0

    def night(self):
        self.hours_since_day += 1
        if self.excess is None:  # night before the first sunrise of the run: nothing stored
            return 0.0
        return self.excess * math.exp(-self.hours_since_day / self.tau)


def patched_source() -> str:
    """Source of Solweig_2022a_calc with the two hooks inserted."""
    import solweig_gpu.solweig as sw

    src = textwrap.dedent(inspect.getsource(sw.Solweig_2022a_calc))
    for needle in (NIGHT_LUP, DAY_LUPN):
        if src.count(needle) != 1:
            raise RuntimeError(f"SOLWEIG source changed; cannot find exactly one: {needle}")
    src = src.replace(NIGHT_LUP, NIGHT_LUP + " + _HEAT_STORAGE.night()")
    src = src.replace(DAY_LUPN, DAY_LUPN + "\n        _HEAT_STORAGE.day(Lup, SBC, emis_grid, Ta)")
    return src


@contextlib.contextmanager
def ground_heat_storage(tau_hours: float) -> Iterator[None]:
    """Run SOLWEIG-GPU with ground heat kept after sunset, decaying with time constant tau.

    The patched function is compiled into the live module namespace (and the module's own
    name restored straight after), so it calls whatever versions of the shadow, view-factor
    and sky functions are installed at call time. It therefore combines with the fast
    replacements in ``fast.py`` in either order.
    """
    import solweig_gpu.solweig as sw
    import solweig_gpu.utci_process as up

    namespace = vars(sw)
    original_in_module = sw.Solweig_2022a_calc
    namespace["_HEAT_STORAGE"] = _State(tau_hours)
    try:
        exec(compile(patched_source(), sw.__file__, "exec"), namespace)
        patched = namespace["Solweig_2022a_calc"]
    finally:
        sw.Solweig_2022a_calc = original_in_module  # only utci_process gets the patched version
    old = up.Solweig_2022a_calc
    up.Solweig_2022a_calc = patched
    try:
        yield
    finally:
        up.Solweig_2022a_calc = old
        namespace.pop("_HEAT_STORAGE", None)
