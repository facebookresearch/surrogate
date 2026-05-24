# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass, field
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class TransformersModel:
    """
    Wraps a HuggingFace transformers model for local logprob-based scoring.

    Loads model weights onto GPU and provides logprob extraction via forward
    passes. Unlike remote APIs that return only top-k logprobs, this gives
    access to the full vocabulary distribution.

    Usage::

        model = TransformersModel(
            model_name="Qwen2.5-7B-Instruct",
            model_path="/local/path/to/weights",
        ).load()

        log_probs = model.get_next_token_log_probs("Hello, world!")
        model.unload()  # free GPU memory before loading next model
    """

    model_name: str  # Display name (e.g., "Qwen2.5-7B-Instruct")
    model_path: str  # Local path to model weights or HuggingFace model ID
    device: str = "auto"
    torch_dtype: str = "bfloat16"  # "bfloat16", "float16", or "float32"
    # HF SDPA backend silently returns None for `output_attentions=True` on
    # Qwen2 / Llama. Set to "eager" to expose attentions for attention scoring.
    attn_implementation: str | None = None  # "eager", "sdpa", or None (auto)

    # Internal state — populated by load(), freed by unload()
    _tokenizer: Any = field(init=False, repr=False, default=None)
    _model: Any = field(init=False, repr=False, default=None)
    _loaded: bool = field(init=False, repr=False, default=False)

    def load(self) -> TransformersModel:
        """Load model and tokenizer onto device. Returns self for chaining."""
        if self._loaded:
            logger.info(f"Model {self.model_name} already loaded, skipping")
            return self

        dtype: torch.dtype = getattr(torch, self.torch_dtype, torch.bfloat16)
        logger.info(
            f"Loading {self.model_name} from {self.model_path} "
            f"(dtype={dtype}, device_map={self.device})"
        )

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        load_kwargs: dict[str, Any] = {
            "dtype": dtype,
            "device_map": self.device,
            "trust_remote_code": True,
        }
        if self.attn_implementation is not None:
            load_kwargs["attn_implementation"] = self.attn_implementation
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            **load_kwargs,
        )
        self._model.eval()
        self._loaded = True

        if hasattr(self._model, "device"):
            logger.info(f"{self.model_name} loaded on {self._model.device}")

        return self

    def unload(self) -> None:
        """Free GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        self._loaded = False
        gc.collect()
        torch.cuda.empty_cache()
        logger.info(f"Unloaded {self.model_name}")

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError(
                f"Model {self.model_name} not loaded. Call .load() first."
            )

    @torch.no_grad()
    def get_next_token_log_probs(self, text: str) -> torch.Tensor:
        """
        Get the full log-probability distribution over the vocabulary for the
        next token, given the input text.

        Returns the exact log-softmax over the entire vocabulary, enabling
        precise logprob lookups for any token without top-k truncation.

        Args:
            text: Input text with chat template already applied.

        Returns:
            1D tensor of shape (vocab_size,) with log-softmax probabilities.
        """
        self._ensure_loaded()

        inputs: dict[str, torch.Tensor] = self._tokenizer(text, return_tensors="pt")
        input_ids: torch.Tensor = inputs["input_ids"].to(self._model.device)
        attention_mask: torch.Tensor | None = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self._model.device)

        outputs: Any = self._model(input_ids=input_ids, attention_mask=attention_mask)
        logits: torch.Tensor = outputs.logits[0, -1]  # (vocab_size,)
        return torch.log_softmax(logits, dim=-1)

    @torch.no_grad()
    def get_next_token_log_probs_batch(self, texts: list[str]) -> list[torch.Tensor]:
        """
        Batched version of get_next_token_log_probs.

        Tokenizes all texts with left-padding, runs a single forward pass,
        and extracts the log-softmax at each sequence's last real token.

        Args:
            texts: List of input texts (chat template already applied).

        Returns:
            List of 1D tensors, each of shape (vocab_size,).
        """
        self._ensure_loaded()
        if not texts:
            return []

        # Left-pad so the last token is the prediction position
        self._tokenizer.padding_side = "left"
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        inputs: Any = self._tokenizer(texts, return_tensors="pt", padding=True)
        input_ids: torch.Tensor = inputs["input_ids"].to(self._model.device)
        attention_mask: torch.Tensor = inputs["attention_mask"].to(self._model.device)

        outputs: Any = self._model(input_ids=input_ids, attention_mask=attention_mask)
        # With left-padding, the last position is always the prediction position
        logits: torch.Tensor = outputs.logits[:, -1, :]  # (batch, vocab_size)
        log_probs: torch.Tensor = torch.log_softmax(logits, dim=-1)

        return [log_probs[i] for i in range(len(texts))]

    @torch.no_grad()
    def get_sequence_log_probs_batch(
        self,
        prompt_texts: list[str],
        target_texts: list[str],
    ) -> list[float]:
        """
        Batched version of get_sequence_log_probs.

        Left-pads prompts+targets and reads per-target-token log-probs at
        each row's known target positions. Left-pad matches the convention
        used by ``get_next_token_log_probs_batch`` and HF's recommendation
        for decoder-only batched inference (real tokens end at the row's
        last index, simplifying any "logit at last real position" lookup).

        Args:
            prompt_texts: List of prompt texts (chat template already applied).
            target_texts: List of target continuations (one per prompt).

        Returns:
            List of summed log-probabilities, one per input.
        """
        self._ensure_loaded()
        if not prompt_texts:
            return []

        self._tokenizer.padding_side = "left"
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Tokenize each prompt and target separately to know boundaries
        prompt_id_lists: list[list[int]] = [
            self._tokenizer.encode(p, add_special_tokens=False) for p in prompt_texts
        ]
        target_id_lists: list[list[int]] = [
            self._tokenizer.encode(t, add_special_tokens=False) for t in target_texts
        ]

        # Build full sequences and left-pad
        full_id_lists: list[list[int]] = [
            p + t for p, t in zip(prompt_id_lists, target_id_lists)
        ]
        max_len: int = max(len(ids) for ids in full_id_lists)
        pad_id: int = self._tokenizer.pad_token_id

        padded: list[list[int]] = []
        attention_masks: list[list[int]] = []
        for ids in full_id_lists:
            n_pad: int = max_len - len(ids)
            padded.append([pad_id] * n_pad + ids)
            attention_masks.append([0] * n_pad + [1] * len(ids))

        input_ids: torch.Tensor = torch.tensor(padded, device=self._model.device)
        attn_mask: torch.Tensor = torch.tensor(
            attention_masks, device=self._model.device
        )

        outputs: Any = self._model(input_ids=input_ids, attention_mask=attn_mask)
        logits: torch.Tensor = outputs.logits  # (batch, seq_len, vocab_size)
        log_probs: torch.Tensor = torch.log_softmax(logits, dim=-1)

        # Left-pad puts every row's real tokens at the end, so target_i's
        # logit (which predicts target_i) is at index max_len - n - 1 + i,
        # where n = len(target). Pad amount and prompt length drop out.
        results: list[float] = []
        for b in range(len(prompt_texts)):
            target_ids: list[int] = target_id_lists[b]
            if not target_ids:
                results.append(-float("inf"))
                continue
            n: int = len(target_ids)
            total: float = 0.0
            for i, tid in enumerate(target_ids):
                pos: int = max_len - n - 1 + i
                total += float(log_probs[b, pos, tid])
            results.append(total)

        return results

    @torch.no_grad()
    def get_last_layer_repr_batch(
        self,
        texts: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extract last-layer hidden states at the last real (non-pad) token.

        Returns both the residual-stream hidden state (pre-final-norm) and
        the post-final-norm hidden state, since ``lm_head`` reads from the
        post-norm representation. The post-norm vector is the one for which
        ``logit_diff = (W[pos] - W[neg]) · z`` holds exactly (linear
        readout), making the Cauchy-Schwarz bound
        ``|Δlogits| ≤ ‖w‖·‖Δz_postnorm‖`` tight.

        Implementation: registers forward pre/post hooks on the model's
        final RMSNorm module to capture both its input (pre-norm residual
        stream) and output (post-norm = ``lm_head`` input). This avoids the
        ``output_hidden_states=True`` ambiguity — recent HF Qwen2/Llama
        versions append the post-norm activation as ``hidden_states[-1]``,
        but earlier versions appended the pre-norm activation. Hooking the
        norm module directly is unambiguous and works across versions.

        Uses left-padding so the prediction position is always the last
        index, matching ``get_next_token_log_probs_batch``.

        Args:
            texts: List of input texts (chat template already applied).

        Returns:
            (prenorm, postnorm), each a tensor of shape (batch, hidden_size).
            If the final norm cannot be located, prenorm == postnorm and a
            warning is logged.
        """
        self._ensure_loaded()
        if not texts:
            empty: torch.Tensor = torch.empty(0)
            return empty, empty

        self._tokenizer.padding_side = "left"
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        inputs: Any = self._tokenizer(texts, return_tensors="pt", padding=True)
        input_ids: torch.Tensor = inputs["input_ids"].to(self._model.device)
        attention_mask: torch.Tensor = inputs["attention_mask"].to(self._model.device)

        norm_module: Any = self._locate_final_norm()
        if norm_module is None:
            # Degenerate fallback: hidden_states[-1] under the assumption
            # used by current HF Qwen2/Llama (post-norm); prenorm == postnorm.
            outputs: Any = self._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            z: torch.Tensor = outputs.hidden_states[-1][:, -1, :]
            return z, z

        captured: dict[str, torch.Tensor] = {}

        def pre_hook(_mod: Any, args: tuple[torch.Tensor, ...]) -> None:
            captured["pre"] = args[0].detach()

        def post_hook(
            _mod: Any, _args: tuple[torch.Tensor, ...], output: torch.Tensor
        ) -> None:
            captured["post"] = output.detach()

        h_pre: Any = norm_module.register_forward_pre_hook(pre_hook)
        h_post: Any = norm_module.register_forward_hook(post_hook)
        try:
            # Call the inner backbone (model.model) to skip lm_head, which would
            # otherwise materialize logits over the full vocabulary (~24 GB on
            # Qwen2.5-14B with batch×seq large enough). The norm hooks still
            # fire, since model.model.forward calls model.model.norm directly.
            backbone: Any = getattr(self._model, "model", self._model)
            backbone(input_ids=input_ids, attention_mask=attention_mask)
        finally:
            h_pre.remove()
            h_post.remove()

        # Last real token = last index (left-padding puts pad on the left).
        return captured["pre"][:, -1, :], captured["post"][:, -1, :]

    def _locate_final_norm(self) -> Any:
        """Return the final RMSNorm/LayerNorm module applied before lm_head,
        or None if not found.

        Llama/Qwen2 expose it as ``model.model.norm``. When None is
        returned, callers fall back to a degenerate path where the
        pre-norm and post-norm hidden states are identical.
        """
        inner: Any = getattr(self._model, "model", None)
        if inner is not None:
            norm: Any = getattr(inner, "norm", None)
            if norm is not None:
                return norm
        logger.warning(
            f"Could not locate final norm on {self.model_name}; "
            "prenorm and postnorm hidden states will be identical."
        )
        return None

    @torch.no_grad()
    def get_unembedding_direction(
        self,
        pos_token_ids: list[int],
        neg_token_ids: list[int],
    ) -> torch.Tensor:
        """
        Build the linear-readout direction `w = Σ W[pos] - Σ W[neg]`.

        Under a strictly linear readout (single token per side, no final
        norm — or equivalently, when reading from the post-final-norm
        hidden state z), the log-odds is `w · z` and so the perturbation
        attribution is `w · (z_orig - z_ablated)` exactly. With multiple
        tokens per side aggregated via logsumexp, this is only an
        approximation (the true gradient is a softmax-weighted combination
        of the W rows). We log it anyway as a sanity-check signal.

        Args:
            pos_token_ids: Token IDs in the positive label set.
            neg_token_ids: Token IDs in the negative label set.

        Returns:
            A 1D tensor of shape (hidden_size,) on the model's device.
        """
        self._ensure_loaded()
        weight: torch.Tensor = self._model.lm_head.weight  # (vocab, hidden)
        pos_sum: torch.Tensor = weight[pos_token_ids].sum(dim=0)
        neg_sum: torch.Tensor = weight[neg_token_ids].sum(dim=0)
        return pos_sum - neg_sum

    @torch.no_grad()
    def get_unembedding_direction_prenorm(
        self,
        pos_token_ids: list[int],
        neg_token_ids: list[int],
    ) -> torch.Tensor:
        """Gamma-scaled unembedding direction for use in pre-norm space.

        For an RMSNorm-style decoder (Llama, Qwen2), the post-norm hidden
        state relates to the pre-norm one by
            z_i = gamma_i * sqrt(d) * z_pre_i / ||z_pre||
        so
            z · w = sqrt(d) * (z_pre · (gamma .* w)) / ||z_pre||

        Thus ``v_eff = gamma .* w`` is the natural pre-norm direction
        such that the change in (z · w) under perturbation can be written
        purely in terms of pre-norm scalars — enabling the additive
        LayerNorm-vs-linear decomposition documented on
        ``representation_scoring.representation_metrics_batch``.

        Falls back to plain ``w`` if the final norm or its ``weight``
        parameter cannot be located (and logs a warning via the
        ``_locate_final_norm`` path).
        """
        self._ensure_loaded()
        w: torch.Tensor = self.get_unembedding_direction(pos_token_ids, neg_token_ids)
        norm_module: Any = self._locate_final_norm()
        if norm_module is None:
            return w
        gamma: Any = getattr(norm_module, "weight", None)
        if gamma is None:
            return w
        return gamma.to(device=w.device, dtype=w.dtype) * w

    def encode_single_token(self, token_text: str) -> int | None:
        """
        Encode a string as a single token ID.

        Returns None if the string maps to multiple tokens in this model's
        vocabulary (e.g., "contradiction" may be multiple BPE tokens).
        """
        self._ensure_loaded()
        token_ids: list[int] = self._tokenizer.encode(
            token_text, add_special_tokens=False
        )
        if len(token_ids) == 1:
            return token_ids[0]
        return None

    def dialog_to_text(self, dialog: Any) -> str:
        """
        Convert a Dialog to text using this model's chat template.

        Applies the tokenizer's ``chat_template`` to produce properly formatted
        input with a generation prompt appended.
        """
        self._ensure_loaded()
        messages: list[dict[str, str]] = []
        for msg in dialog.messages:
            messages.append({"role": msg.role, "content": msg.content})

        result: str = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return result
