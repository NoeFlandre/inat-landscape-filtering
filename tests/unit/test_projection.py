"""Equal Earth projection: the published values, and true areas."""

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from inat_landscape.domain.projection import equal_earth


def test_equal_earth_gives_the_published_extent():
    x, y = equal_earth([180.0, 0.0, -180.0, 0.0], [0.0, 90.0, 0.0, -90.0])
    assert np.allclose(x, [2.706629, 0.0, -2.706629, 0.0], atol=1e-6)
    assert np.allclose(y, [0.0, 1.317363, 0.0, -1.317363], atol=1e-6)


def test_equal_earth_of_a_point_inside():
    x, y = equal_earth(np.array([90.0]), np.array([45.0]))
    assert np.allclose([x[0], y[0]], [1.159855, 0.860231], atol=1e-6)


def _area(lon0, lon1, lat0, lat1):
    """Area of the projected cell, from its outline with many points on each side."""
    up, flat = np.linspace(0, 1, 400), np.zeros(400)
    lon = lon0 + (lon1 - lon0) * np.r_[up, flat + 1, 1 - up, flat]
    lat = lat0 + (lat1 - lat0) * np.r_[flat, up, flat + 1, 1 - up]
    x, y = equal_earth(lon, lat)
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


@given(st.floats(-170, 160), st.floats(1, 20), st.floats(-80, 70), st.floats(1, 10))
def test_equal_earth_keeps_the_areas(lon, width, lat, height):
    sphere = np.radians(width) * (np.sin(np.radians(lat + height)) - np.sin(np.radians(lat)))
    assert np.isclose(_area(lon, lon + width, lat, lat + height), sphere, rtol=1e-3)
