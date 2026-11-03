"""Complete layers of the uniform order (ADR 0011): the cap that every H3 cell has reached."""

import numpy as np
from numpy.typing import NDArray


def complete_cap(ranks: NDArray[np.int64], done: NDArray[np.bool_], tolerance: float) -> int:
    """Return the largest cap k such that the photos of rank below k are done.

    `ranks` is the rank of each sample photo in its cell, and `done` tells if it has an
    embedding. Up to a share `tolerance` of the photos below k can be missing (failed
    downloads), at each layer counted from the first one.
    """
    if not len(ranks):
        return 0
    total = np.cumsum(np.bincount(ranks))
    missing = np.cumsum(np.bincount(ranks[~done], minlength=len(total)))
    bad = np.flatnonzero(missing > tolerance * total)
    return int(bad[0]) if len(bad) else len(total)
