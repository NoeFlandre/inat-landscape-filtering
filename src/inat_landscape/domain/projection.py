"""Equal Earth projection (Šavrič, Patterson and Jenny, 2018): it keeps the areas."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

_A1, _A2, _A3, _A4 = 1.340264, -0.081106, 0.000893, 0.003796
_M = np.sqrt(3) / 2


def equal_earth(lon: ArrayLike, lat: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the x and y on a sphere of radius 1 of each point (degrees)."""
    lam, phi = np.radians(lon), np.radians(lat)
    theta = np.arcsin(_M * np.sin(phi))
    t2 = theta**2
    t6 = t2**3
    x = lam * np.cos(theta) / (_M * (_A1 + 3 * _A2 * t2 + t6 * (7 * _A3 + 9 * _A4 * t2)))
    y = theta * (_A1 + _A2 * t2 + t6 * (_A3 + _A4 * t2))
    return x, y
