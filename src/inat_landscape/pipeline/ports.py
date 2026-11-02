"""Interfaces that the stages need from the outside world."""

from collections.abc import Callable, Mapping, Sequence
from typing import Protocol

import numpy as np
from PIL import Image

from inat_landscape.adapters.images import Fetched


class Encoder(Protocol):
    """Maps images and texts into one embedding space, with SigLIP logit scale and bias."""

    model_id: str

    @property
    def scale(self) -> float: ...
    @property
    def bias(self) -> float: ...
    def encode_images(self, images: Sequence[Image.Image]) -> np.ndarray: ...
    def encode_texts(self, texts: Sequence[str]) -> np.ndarray: ...


Fetcher = Callable[[Mapping[int, str]], list[Fetched]]
Log = Callable[[str], None]
