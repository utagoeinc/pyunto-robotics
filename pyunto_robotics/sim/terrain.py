"""Procedural ground for the outdoor scenes.

MuJoCo declares a heightfield's size in XML but its elevation data has to come from somewhere.
Loading a PNG would mean shipping a binary asset and would make the terrain a fixed thing;
generating it here keeps the whole world in code, lets a scene ask for "a lawn" or "the lunar
south pole" by name, and makes the terrain reproducible without a file.

Everything is written into `model.hfield_data` before the first step. The data is a row-major
grid of values in 0..1, which MuJoCo scales by the `size` attribute's third component -- so a
hfield declared `size="12 10 0.55 0.1"` spans 24 x 20 m with 0.55 m between the lowest and
highest point, and a base 0.1 m thick underneath.

There is no RNG seeded from the clock anywhere here. Terrain is generated from deterministic
functions of position, so two runs of the same scene are the same world -- which matters
because a patrol that works once should work again.
"""

from __future__ import annotations

import mujoco
import numpy as np


def _grid(model: mujoco.MjModel, name: str) -> tuple[int, int, int] | None:
    """(address, rows, cols) of a named heightfield, or None if the model has no such field."""
    field = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_HFIELD, name)
    if field < 0:
        return None
    return int(model.hfield_adr[field]), int(model.hfield_nrow[field]), int(
        model.hfield_ncol[field]
    )


def _write(model: mujoco.MjModel, name: str, height: np.ndarray) -> bool:
    """Normalise a height grid to 0..1 and write it into the model."""
    grid = _grid(model, name)
    if grid is None:
        return False
    address, rows, cols = grid
    if height.shape != (rows, cols):
        raise ValueError(f"height grid is {height.shape}, hfield '{name}' is {(rows, cols)}")

    lowest = float(height.min())
    span = float(height.max()) - lowest
    # A perfectly flat field would divide by zero; leaving it at zero is the right answer.
    normalised = (height - lowest) / span if span > 1e-9 else np.zeros_like(height)
    model.hfield_data[address : address + rows * cols] = normalised.ravel().astype(np.float32)
    return True


def lawn(model: mujoco.MjModel, name: str = "lawn_hf") -> bool:
    """Gently uneven ground with a slope on one side.

    Three things layered, each with a purpose:

      * a broad tilt, so the west of the site is about 0.4 m higher than the east. That is
        roughly 2 degrees over 24 m -- enough for the trunk controller's ground-relative
        height hold to have something to do, not so much that walking becomes a slope test.
      * long undulations, the scale of real ground settling.
      * a fine ripple, so no two footfalls land on identical terrain.

    All deterministic functions of position: no randomness, so the same scene is the same
    world every run.
    """
    grid = _grid(model, name)
    if grid is None:
        return False
    _, rows, cols = grid

    # u runs 0..1 across the field's x, v across its y.
    v, u = np.meshgrid(
        np.linspace(0.0, 1.0, cols), np.linspace(0.0, 1.0, rows), indexing="xy"
    )

    tilt = 0.42 * (1.0 - u)
    undulation = 0.10 * np.sin(2.2 * np.pi * u) * np.cos(1.7 * np.pi * v)
    ripple = 0.035 * np.sin(7.0 * np.pi * u + 1.3) * np.sin(5.0 * np.pi * v + 0.7)

    return _write(model, name, tilt + undulation + ripple)


def lunar_south_pole(model: mujoco.MjModel, name: str = "regolith_hf") -> bool:
    """Cratered regolith, lit obliquely -- the terrain of the lunar south pole.

    The shape of this terrain is the whole reason the lunar scene is interesting to drive on:

      * **Craters.** Bowl-shaped depressions with raised rims, which is what an impact leaves.
        A rover has to go round the large ones and can cross the small ones, so crater size is
        what makes route choice a real decision rather than a straight line.
      * **A regional slope** toward the pole, because the south polar terrain is a long grade
        into permanently shadowed basins.
      * **Boulder-scale roughness**, the thing that actually shakes a rover apart.

    Crater positions are a fixed list rather than random draws: a rover route that works has to
    work the same way tomorrow, and a terrain that reshuffles every run cannot be debugged.
    """
    grid = _grid(model, name)
    if grid is None:
        return False
    _, rows, cols = grid

    v, u = np.meshgrid(
        np.linspace(0.0, 1.0, cols), np.linspace(0.0, 1.0, rows), indexing="xy"
    )
    height = np.zeros((rows, cols), dtype=np.float64)

    # Regional grade, dropping toward one corner: the run-in to a polar basin.
    height += 0.55 * (1.0 - v) * 0.6 + 0.25 * u

    # (centre u, centre v, radius, depth). Two large craters worth avoiding, several small
    # ones that can be driven through, and one deep basin that stays in shadow.
    craters = [
        (0.30, 0.68, 0.150, 0.62),
        (0.72, 0.35, 0.130, 0.55),
        (0.52, 0.80, 0.075, 0.26),
        (0.18, 0.28, 0.065, 0.22),
        (0.84, 0.72, 0.055, 0.18),
        (0.44, 0.18, 0.050, 0.16),
        (0.62, 0.58, 0.042, 0.13),
    ]
    for centre_u, centre_v, radius, depth in craters:
        distance = np.sqrt((u - centre_u) ** 2 + (v - centre_v) ** 2)
        inside = distance < radius
        # Bowl floor: a cosine profile, deepest at the centre and flattening at the rim.
        bowl = -depth * np.cos(np.pi * distance / (2.0 * radius)) ** 2
        # Raised rim just outside the bowl, which is the ejecta blanket and is what a rover
        # actually has to climb to get out.
        rim_band = (distance >= radius) & (distance < radius * 1.35)
        rim = depth * 0.22 * np.cos(
            np.pi * (distance - radius) / (0.70 * radius)
        ) ** 2
        height += np.where(inside, bowl, 0.0) + np.where(rim_band, rim, 0.0)

    # Boulder-scale roughness. Deterministic, and fine enough that the wheels feel it.
    height += 0.030 * np.sin(11.0 * np.pi * u + 0.4) * np.sin(9.0 * np.pi * v + 1.1)
    height += 0.018 * np.sin(19.0 * np.pi * v + 2.2) * np.cos(17.0 * np.pi * u)

    return _write(model, name, height)


def mars_plain(model: mujoco.MjModel, name: str = "mars_hf") -> bool:
    """Martian terrain: an old outflow channel with dunes, not a cratered plain.

    Deliberately a different landscape from the Moon's, because a Mars demonstration that is
    the lunar one with red paint teaches nobody anything. The Moon is shaped by impact --
    bowls with raised rims, everywhere, at every scale. Mars has had wind and water, so what
    dominates is flowing shapes: a broad channel cut into the plain, transverse dunes marching
    across it, and scattered impact craters that are the older, softened remnants they are on
    a planet with weather.

    For a rover this makes route choice a different problem. Lunar craters are obstacles to go
    around; a channel is a corridor to follow, and dunes are a washboard that is crossable but
    slow. Both are navigable -- the point is that the terrain rewards choosing a line.

    Fixed rather than random, for the same reason the lunar one is: a route that works has to
    work again tomorrow.
    """
    grid = _grid(model, name)
    if grid is None:
        return False
    _, rows, cols = grid

    v, u = np.meshgrid(
        np.linspace(0.0, 1.0, cols), np.linspace(0.0, 1.0, rows), indexing="xy"
    )
    height = np.zeros((rows, cols), dtype=np.float64)

    # A gentle regional tilt. Nothing on Mars is level for long.
    height += 0.30 * u + 0.12 * v

    # The outflow channel: a broad, shallow trough running roughly west to east, with banks
    # either side. `sech`-like profile rather than a V, because water-cut channels are
    # flat-floored with rounded shoulders.
    channel_centre = 0.46 + 0.06 * np.sin(2.4 * np.pi * u)
    across = (v - channel_centre) / 0.17
    height -= 0.34 / np.cosh(across) ** 2
    # Banks: the material the channel cut through, standing proud either side.
    bank = (np.abs(across) > 1.0) & (np.abs(across) < 2.2)
    height += np.where(bank, 0.11 * np.cos(np.pi * (np.abs(across) - 1.6) / 1.2) ** 2, 0.0)

    # Transverse dunes across the channel floor. Asymmetric -- a long windward slope and a
    # short slip face -- which is what a dune actually is and what makes crossing one
    # directional.
    dune_phase = 13.0 * np.pi * u
    dunes = 0.055 * (np.sin(dune_phase) + 0.35 * np.sin(2 * dune_phase))
    height += np.where(np.abs(across) < 1.4, dunes, 0.0)

    # Old impact craters, softened. Shallower than the Moon's for their width, with rims
    # mostly eroded away: that is what a few billion years of wind does to one.
    craters = [
        (0.22, 0.18, 0.105, 0.26),
        (0.78, 0.82, 0.090, 0.21),
        (0.62, 0.12, 0.060, 0.13),
        (0.36, 0.88, 0.050, 0.10),
    ]
    for centre_u, centre_v, radius, depth in craters:
        distance = np.sqrt((u - centre_u) ** 2 + (v - centre_v) ** 2)
        inside = distance < radius
        bowl = -depth * np.cos(np.pi * distance / (2.0 * radius)) ** 2
        rim_band = (distance >= radius) & (distance < radius * 1.5)
        rim = depth * 0.10 * np.cos(np.pi * (distance - radius) / (1.0 * radius)) ** 2
        height += np.where(inside, bowl, 0.0) + np.where(rim_band, rim, 0.0)

    # Rock-strewn roughness. Mars rovers lose more time to this than to anything dramatic.
    height += 0.022 * np.sin(15.0 * np.pi * u + 0.7) * np.sin(12.0 * np.pi * v + 0.3)
    height += 0.013 * np.cos(23.0 * np.pi * v) * np.sin(21.0 * np.pi * u + 1.9)

    return _write(model, name, height)


def apply(model: mujoco.MjModel) -> list[str]:
    """Fill every heightfield this module knows how to generate. Returns the names it wrote.

    Called once after a model is compiled and before it is stepped. Scenes without a
    heightfield are unaffected, so it is safe to call for any model -- which is why Robot can
    call it unconditionally rather than each scene having to remember.
    """
    written = []
    for name, generator in (
        ("lawn_hf", lawn),
        ("regolith_hf", lunar_south_pole),
        ("mars_hf", mars_plain),
    ):
        if generator(model, name):
            written.append(name)
    return written
