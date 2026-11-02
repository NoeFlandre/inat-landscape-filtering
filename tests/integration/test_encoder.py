"""SiglipEncoder with a tiny random SigLIP model (downloads about 1 MB once)."""

import numpy as np
import pytest
import torch
import torch._inductor.config as inductor_config
from PIL import Image
from transformers import AutoModel, AutoProcessor

from inat_landscape.adapters.encoder import SiglipEncoder, autocast_dtype

TINY = "optimum-intel-internal-testing/tiny-random-SiglipModel"


@pytest.fixture(scope="module")
def encoder():
    return SiglipEncoder(TINY, batch_size=2)


def _images():
    rng = np.random.default_rng(0)
    return [
        Image.fromarray(rng.integers(0, 255, (40 + 7 * i, 30 + 5 * i, 3), dtype=np.uint8))
        for i in range(5)
    ]


def test_encodes_images_in_batches(encoder):
    images = _images()
    emb = encoder.encode_images(images)
    assert emb.shape[0] == 5
    assert emb.dtype == np.float32
    assert np.allclose(emb[:2], encoder.encode_images(images[:2]), atol=1e-3)


def test_torch_preprocessing_matches_the_model_processor():
    images = _images()
    processor = AutoProcessor.from_pretrained(TINY)
    model = AutoModel.from_pretrained(TINY).eval()
    with torch.inference_mode():
        out = model.get_image_features(**processor(images=images, return_tensors="pt"))
    expected = (out if isinstance(out, torch.Tensor) else out.pooler_output).numpy()
    got = SiglipEncoder(TINY, batch_size=8, precision="fp32").encode_images(images)
    cos = (got * expected).sum(1) / np.linalg.norm(got, axis=1) / np.linalg.norm(expected, axis=1)
    assert cos.min() > 0.9999


def test_compiled_encoder_freezes_weights_and_gives_the_same_embeddings(encoder):
    images = _images()
    compiled = SiglipEncoder(TINY, batch_size=2, compile=True).encode_images(images)
    assert inductor_config.freezing
    assert np.allclose(compiled, encoder.encode_images(images), atol=2e-2)


def test_encodes_texts(encoder):
    emb = encoder.encode_texts(["a photo of a landscape", "a leaf"])
    assert emb.shape[0] == 2
    assert not np.allclose(emb[0], emb[1])


def test_empty_image_list(encoder):
    assert encoder.encode_images([]).shape == (0, 0)


def test_logit_scale_and_bias_are_floats(encoder):
    assert isinstance(encoder.scale, float)
    assert isinstance(encoder.bias, float)


@pytest.mark.parametrize(
    ("device", "flags", "cuda_bf16", "expected"),
    [
        ("cpu", "fpu avx2 amx_bf16", False, torch.bfloat16),
        ("cpu", "fpu avx512_bf16", False, torch.bfloat16),
        ("cpu", "fpu avx2", False, None),
        ("cuda", "", True, torch.bfloat16),
        ("cuda", "", False, torch.float16),
    ],
)
def test_autocast_dtype(device, flags, cuda_bf16, expected):
    assert autocast_dtype(device, flags, cuda_bf16) == expected
