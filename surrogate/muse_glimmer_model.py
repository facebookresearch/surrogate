# Copyright (c) 2025 The Authors
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Public Hugging Face adapter for the Muse Glimmer model family."""

from __future__ import annotations

import logging
from typing import Any

import torch
from transformers import AutoTokenizer

from surrogate.transformers_model import TransformersModel


logger: logging.Logger = logging.getLogger(__name__)

DIRECT_RESPONSE_PREFIX: str = " to=user<|message|>"
TEXT_ONLY_DEVICE_MAP: dict[str, str | int] = {
    "model.vision_tower": "cpu",
    "model.vision_adapter": "cpu",
    "model.vision_projection": "cpu",
    "model.perception_emb_norm": "cpu",
    "model.language_model": 0,
    "lm_head": 0,
}


def _image_text_auto_model_class() -> Any:
    """Resolve the loader lazily so older Transformers still import the repo."""
    try:
        from transformers import AutoModelForImageTextToText
    except ImportError as error:
        raise RuntimeError(
            "Muse Glimmer requires transformers>=5.15; install the current "
            "Transformers release before running this optional experiment"
        ) from error
    return AutoModelForImageTextToText


class _ConfigProxy:
    """Expose decoder depth while delegating all other config attributes."""

    def __init__(self, config: Any, num_hidden_layers: int) -> None:
        self._wrapped_config: Any = config
        self.num_hidden_layers: int = num_hidden_layers

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped_config, name)


class _TextInputDeviceProxy:
    """Expose text-only metadata without mutating the model's outer config."""

    def __init__(
        self,
        model: Any,
        input_device: torch.device,
        num_hidden_layers: int,
    ) -> None:
        self._wrapped_model: Any = model
        self._input_device: torch.device = input_device
        self._num_hidden_layers: int = num_hidden_layers

    @property
    def config(self) -> Any:
        return _ConfigProxy(self._wrapped_model.config, self._num_hidden_layers)

    @property
    def device(self) -> torch.device:
        return self._input_device

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # ``torch.no_grad`` decorators do not span the execution of an async
        # function body. The shared attention scorer is async, so enforce
        # inference semantics at this model boundary as well.
        with torch.no_grad():
            return self._wrapped_model(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped_model, name)


class MuseGlimmerTransformersModel(TransformersModel):
    """Adapt Muse Glimmer's public multimodal wrapper to text-only scoring.

    Muse Glimmer exposes its causal decoder at ``model.language_model`` and
    uses an assistant routing header before response content. Forcing the
    official direct-response route makes the next position comparable to the
    answer-token position used for the other benchmark models.
    """

    def load(self) -> MuseGlimmerTransformersModel:
        """Load the public image-text model and expose its text depth."""
        if self._loaded:
            logger.info("Model %s already loaded, skipping", self.model_name)
            return self

        dtype: torch.dtype = getattr(torch, self.torch_dtype, torch.bfloat16)
        device_map: str | dict[str, str | int] = (
            TEXT_ONLY_DEVICE_MAP
            if self.device == "auto" and torch.cuda.is_available()
            else self.device
        )
        logger.info(
            "Loading %s from %s (dtype=%s, device_map=%s)",
            self.model_name,
            self.model_path,
            dtype,
            device_map,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        load_kwargs: dict[str, Any] = {
            "dtype": dtype,
            "device_map": device_map,
            "trust_remote_code": True,
        }
        if self.attn_implementation is not None:
            load_kwargs["attn_implementation"] = self.attn_implementation
        model_class: Any = _image_text_auto_model_class()
        loaded_model: Any = model_class.from_pretrained(
            self.model_path,
            **load_kwargs,
        )
        loaded_model.eval()
        self._model = loaded_model
        decoder: Any = self._text_decoder()
        num_hidden_layers: int = len(decoder.layers)
        first_parameter: torch.Tensor | None = next(decoder.parameters(), None)
        if first_parameter is None:
            raise RuntimeError("Muse Glimmer text decoder has no parameters")
        if isinstance(device_map, dict):
            if first_parameter.device.type != "cuda":
                raise RuntimeError(
                    "Muse Glimmer text decoder was not placed on a CUDA device"
                )
        self._model = _TextInputDeviceProxy(
            loaded_model,
            first_parameter.device,
            num_hidden_layers,
        )
        self._loaded = True
        logger.info("%s loaded on %s", self.model_name, self._model.device)
        return self

    def _text_decoder(self) -> Any:
        """Return the nested causal decoder used for text-only inputs."""
        outer_model: Any = getattr(self._model, "model", None)
        decoder: Any = getattr(outer_model, "language_model", None)
        if decoder is None:
            raise RuntimeError(
                "Muse Glimmer must expose model.language_model for text scoring"
            )
        return decoder

    def _locate_final_norm(self) -> Any:
        """Return the nested decoder's final normalization module."""
        norm: Any = getattr(self._text_decoder(), "norm", None)
        if norm is None:
            raise RuntimeError("Muse Glimmer text decoder has no final norm")
        return norm

    def dialog_to_text(self, dialog: Any) -> str:
        """Render a prompt ending immediately before direct answer content."""
        rendered: str = super().dialog_to_text(dialog)
        if not rendered.endswith("<|start|>assistant"):
            raise RuntimeError(
                "Unexpected Muse Glimmer chat-template suffix; refusing to "
                "score a potentially incorrect token position"
            )
        return rendered + DIRECT_RESPONSE_PREFIX
