"""SigLIP image and text encoder on CPU or GPU (PyTorch + transformers).

Speed: images are resized with PIL in a thread pool (the same resize as the model
processor, in parallel), normalized with torch on the device, the vision tower runs under
bf16/fp16 autocast when the hardware supports it, and it can be compiled with torch.compile.
"""

import os
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from inat_landscape.domain.scoring import MODEL_ID

_RESIZE_THREADS = os.cpu_count() or 4


def autocast_dtype(device: str, cpu_flags: str, cuda_bf16: bool) -> torch.dtype | None:
    """Return the fast float type of the hardware, or None to stay in float32."""
    if device == "cuda":
        return torch.bfloat16 if cuda_bf16 else torch.float16
    fast_cpu = {"amx_bf16", "avx512_bf16"} & set(cpu_flags.split())
    return torch.bfloat16 if fast_cpu else None


class SiglipEncoder:
    """Encode images and texts into the shared SigLIP embedding space."""

    def __init__(
        self,
        model_id: str = MODEL_ID,
        batch_size: int = 32,
        *,
        compile: bool = False,
        precision: str = "auto",
    ) -> None:
        self.model_id = model_id
        self.batch_size = batch_size
        self.compile = compile
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = None if precision == "fp32" else _hardware_dtype(self.device)

    @cached_property
    def _model(self) -> Any:
        return AutoModel.from_pretrained(self.model_id).eval().to(self.device)

    @cached_property
    def _processor(self) -> Any:
        return AutoProcessor.from_pretrained(self.model_id)

    @cached_property
    def _vision(self) -> Any:
        tower = self._model.vision_model
        return torch.compile(tower, dynamic=False) if self.compile else tower

    @cached_property
    def _spec(self) -> tuple[tuple[int, int], int, float, torch.Tensor, torch.Tensor]:
        ip = self._processor.image_processor
        mean = torch.tensor(ip.image_mean, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(ip.image_std, device=self.device).view(1, 3, 1, 1)
        size = (ip.size.width, ip.size.height)  # PIL order
        return size, int(ip.resample), ip.rescale_factor, mean, std

    @cached_property
    def _pool(self) -> ThreadPoolExecutor:
        return ThreadPoolExecutor(_RESIZE_THREADS)

    @property
    def scale(self) -> float:
        return float(self._model.logit_scale.detach().exp())

    @property
    def bias(self) -> float:
        return float(self._model.logit_bias.detach())

    def encode_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        """Return one float32 embedding row per image."""
        rows = [
            self._embed_images(images[i : i + self.batch_size])
            for i in range(0, len(images), self.batch_size)
        ]
        return np.concatenate(rows) if rows else np.zeros((0, 0), np.float32)

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Return one float32 embedding row per text, padded as SigLIP was trained."""
        max_length = self._model.config.text_config.max_position_embeddings
        inputs = self._processor(
            text=list(texts),
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            return _pooled(self._model.get_text_features(**inputs))

    def _embed_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        pixels = self._pixels(images)
        n = len(pixels)
        if self.compile and n < self.batch_size:  # one fixed shape: no recompilation
            pixels = torch.cat([pixels, pixels[-1:].expand(self.batch_size - n, -1, -1, -1)])
        with (
            torch.inference_mode(),
            torch.autocast(
                self.device, dtype=self.dtype or torch.float32, enabled=self.dtype is not None
            ),
        ):
            return _pooled(self._vision(pixel_values=pixels))[:n]

    def _pixels(self, images: Sequence[Image.Image]) -> torch.Tensor:
        size, resample, rescale, mean, std = self._spec
        arrays = list(self._pool.map(lambda im: _resized(im, size, resample), images))
        batch = torch.from_numpy(np.stack(arrays)).to(self.device).permute(0, 3, 1, 2).float()
        return (batch * rescale - mean) / std


def _resized(image: Image.Image, size: tuple[int, int], resample: int) -> np.ndarray:
    """Resize as the model processor does (PIL), and return an RGB uint8 array."""
    return np.asarray(image.convert("RGB").resize(size, resample=resample), dtype=np.uint8)


def _hardware_dtype(device: str) -> torch.dtype | None:
    flags = Path("/proc/cpuinfo").read_text() if Path("/proc/cpuinfo").exists() else ""
    cuda_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    return autocast_dtype(device, flags, cuda_bf16)


def _pooled(output: Any) -> np.ndarray:
    tensor = output if isinstance(output, torch.Tensor) else output.pooler_output
    return tensor.float().cpu().numpy()
