"""SiglipEncoder with a tiny random SigLIP model (downloads about 1 MB once)."""

import numpy as np
import pytest
from PIL import Image

from inat_landscape.adapters.encoder import SiglipEncoder

TINY = "optimum-intel-internal-testing/tiny-random-SiglipModel"


@pytest.fixture(scope="module")
def encoder():
    return SiglipEncoder(TINY, batch_size=2)


def test_encodes_images_in_batches(encoder):
    images = [Image.new("RGB", (40, 30), (i * 40, 0, 0)) for i in range(5)]
    emb = encoder.encode_images(images)
    assert emb.shape[0] == 5
    assert emb.dtype == np.float32
    assert np.allclose(emb[:2], encoder.encode_images(images[:2]), atol=1e-5)


def test_encodes_texts(encoder):
    emb = encoder.encode_texts(["a photo of a landscape", "a leaf"])
    assert emb.shape[0] == 2
    assert not np.allclose(emb[0], emb[1])


def test_empty_image_list(encoder):
    assert encoder.encode_images([]).shape == (0, 0)


def test_logit_scale_and_bias_are_floats(encoder):
    assert isinstance(encoder.scale, float)
    assert isinstance(encoder.bias, float)
