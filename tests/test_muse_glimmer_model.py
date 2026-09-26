# Copyright (c) 2025 The Authors
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Tests for the public Muse Glimmer text-scoring adapter."""

from __future__ import annotations

import hashlib
import json
import os
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any
from unittest import TestCase
from unittest.mock import MagicMock, patch

import torch

from benchmark_scripts.run_glimmer_boolq import (
    MODEL_FILE_SHA256,
    MODEL_REVISION,
    _validate_cached_revision,
)
from surrogate.model_types import make_dialog
from surrogate.muse_glimmer_model import (
    DIRECT_RESPONSE_PREFIX,
    TEXT_ONLY_DEVICE_MAP,
    _TextInputDeviceProxy,
    MuseGlimmerTransformersModel,
)


class MuseGlimmerAdapterTest(TestCase):
    """Pin the routing prefix, nested decoder, and model loader contract."""

    def _model(self) -> Any:
        decoder: Any = SimpleNamespace(
            layers=[object() for _ in range(52)],
            norm=object(),
            parameters=lambda: iter([SimpleNamespace(device=torch.device("cuda"))]),
        )
        return SimpleNamespace(
            model=SimpleNamespace(language_model=decoder),
            config=SimpleNamespace(),
            eval=MagicMock(),
            device=torch.device("meta"),
        )

    def test_load_uses_image_text_loader_and_exposes_text_depth(self) -> None:
        loaded_model: Any = self._model()
        loader: Any = MagicMock()
        loader.from_pretrained.return_value = loaded_model
        tokenizer: Any = MagicMock()
        with (
            patch(
                "surrogate.muse_glimmer_model._image_text_auto_model_class",
                return_value=loader,
            ),
            patch(
                "surrogate.muse_glimmer_model.AutoTokenizer.from_pretrained",
                return_value=tokenizer,
            ),
            patch(
                "surrogate.muse_glimmer_model.torch.cuda.is_available",
                return_value=True,
            ),
        ):
            model = MuseGlimmerTransformersModel(
                "muse-glimmer-30b",
                "meta-models/Muse-Glimmer-30B",
                attn_implementation="eager",
            ).load()

        self.assertIs(model._model._wrapped_model, loaded_model)
        self.assertEqual(model._model.config.num_hidden_layers, 52)
        self.assertIs(
            model._locate_final_norm(), loaded_model.model.language_model.norm
        )
        self.assertEqual(model._model.device, torch.device("cuda"))
        loader.from_pretrained.assert_called_once()
        self.assertEqual(
            loader.from_pretrained.call_args.kwargs["device_map"],
            TEXT_ONLY_DEVICE_MAP,
        )
        self.assertEqual(
            loader.from_pretrained.call_args.kwargs["attn_implementation"], "eager"
        )

    def test_dialog_ends_at_direct_answer_content(self) -> None:
        tokenizer: Any = MagicMock()
        tokenizer.apply_chat_template.return_value = "<|start|>assistant"
        model = MuseGlimmerTransformersModel("muse", "unused")
        model._tokenizer = tokenizer
        model._model = self._model()
        model._loaded = True

        rendered: str = model.dialog_to_text(make_dialog("system", "user"))

        self.assertEqual(rendered, "<|start|>assistant" + DIRECT_RESPONSE_PREFIX)

    def test_dialog_rejects_changed_chat_template_boundary(self) -> None:
        tokenizer: Any = MagicMock()
        tokenizer.apply_chat_template.return_value = "unexpected"
        model = MuseGlimmerTransformersModel("muse", "unused")
        model._tokenizer = tokenizer
        model._model = self._model()
        model._loaded = True

        with self.assertRaisesRegex(RuntimeError, "chat-template suffix"):
            model.dialog_to_text(make_dialog("system", "user"))

    def test_proxy_disables_grad_during_async_scorer_forward(self) -> None:
        grad_states: list[bool] = []

        class _CallableModel:
            config: Any = SimpleNamespace()

            def __call__(self) -> str:
                grad_states.append(torch.is_grad_enabled())
                return "output"

        proxy = _TextInputDeviceProxy(
            _CallableModel(), torch.device("cpu"), num_hidden_layers=52
        )
        with torch.enable_grad():
            output: str = proxy()

        self.assertEqual(output, "output")
        self.assertEqual(grad_states, [False])
        self.assertEqual(proxy.config.num_hidden_layers, 52)


class MuseGlimmerRevisionTest(TestCase):
    """Reject caches that cannot be tied to the declared Hub revision."""

    def _write_tree(self, directory: str) -> None:
        tree_dir: str = os.path.join(directory, ".cache", "huggingface", "trees")
        os.makedirs(tree_dir)
        with open(
            os.path.join(tree_dir, f"{MODEL_REVISION}.json"),
            "w",
            encoding="utf-8",
        ) as output:
            json.dump(
                {
                    "format_version": 1,
                    "files": {
                        "config.json": {
                            "size": 2,
                        }
                    },
                },
                output,
            )

    def test_cached_revision_requires_complete_matching_tree(self) -> None:
        with TemporaryDirectory() as directory:
            self._write_tree(directory)
            with open(os.path.join(directory, "config.json"), "wb") as output:
                output.write(b"{}")
            with patch.dict(
                MODEL_FILE_SHA256,
                {"config.json": hashlib.sha256(b"{}").hexdigest()},
                clear=True,
            ):
                _validate_cached_revision(directory)

    def test_cached_revision_rejects_missing_file(self) -> None:
        with TemporaryDirectory() as directory:
            self._write_tree(directory)
            with patch.dict(MODEL_FILE_SHA256, {"config.json": "0" * 64}, clear=True):
                with self.assertRaisesRegex(RuntimeError, "missing"):
                    _validate_cached_revision(directory)

    def test_cached_revision_rejects_wrong_digest(self) -> None:
        with TemporaryDirectory() as directory:
            self._write_tree(directory)
            with open(os.path.join(directory, "config.json"), "wb") as output:
                output.write(b"[]")
            with patch.dict(
                MODEL_FILE_SHA256,
                {"config.json": hashlib.sha256(b"{}").hexdigest()},
                clear=True,
            ):
                with self.assertRaisesRegex(RuntimeError, "wrong_digest"):
                    _validate_cached_revision(directory)
