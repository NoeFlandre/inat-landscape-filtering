"""SigLIP image and text encoder on CPU or GPU (PyTorch + transformers)."""

from collections.abc import Sequence
from functools import cached_property
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from inat_landscape.domain.scoring import MODEL_ID


class SiglipEncoder:
    """Encode images and texts into the shared SigLIP embedding space."""

    def __init__(self, model_id: str = MODEL_ID, batch_size: int = 32) -> None:
        self.model_id = model_id
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @cached_property
    def _model(self) -> Any:
        return AutoModel.from_pretrained(self.model_id).eval().to(self.device)

    @cached_property
    def _processor(self) -> Any:
        return AutoProcessor.from_pretrained(self.model_id)

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
        inputs = self._processor(images=list(images), return_tensors="pt").to(self.device)
        with torch.inference_mode():
            return _pooled(self._model.get_image_features(**inputs))


def _pooled(output: Any) -> np.ndarray:
    tensor = output if isinstance(output, torch.Tensor) else output.pooler_output
    return tensor.float().cpu().numpy()
