# SPDX-License-Identifier: Apache-2.0
"""Qwen3-TTS upstream compatibility shims."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch
from transformers import masking_utils
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.utils import generic

REPO_ROOT = Path(__file__).resolve().parents[3]
# Note (Akazaakane): loaded by path so the module under test never imports sglang.
COMPAT_PATH = REPO_ROOT / "sglang_omni/audio/qwen3_tts_compat.py"
SPEC = importlib.util.spec_from_file_location("qwen3_tts_compat", COMPAT_PATH)
assert SPEC is not None and SPEC.loader is not None
compat = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compat)

# Note (Akazaakane): must be captured before any test applies the patch.
PRISTINE = {
    name: getattr(masking_utils, name) for name in compat._MASK_FACTORY_NAMES
}  # noqa: leading-underscore  # production name


@pytest.fixture(autouse=True)
def restore_transformers_globals():
    """Undo every global ``apply_...()`` touches, so tests cannot leak into
    the rest of the pytest process."""
    saved_masks = {
        name: getattr(masking_utils, name)
        for name in compat._MASK_FACTORY_NAMES  # noqa: leading-underscore  # production name
    }
    saved_check_model_inputs = generic.check_model_inputs
    saved_rope = ROPE_INIT_FUNCTIONS.get("default")
    rope_was_set = "default" in ROPE_INIT_FUNCTIONS
    try:
        yield
    finally:
        for name, original in saved_masks.items():
            setattr(masking_utils, name, original)
        generic.check_model_inputs = saved_check_model_inputs
        if rope_was_set:
            ROPE_INIT_FUNCTIONS["default"] = saved_rope
        else:
            ROPE_INIT_FUNCTIONS.pop("default", None)


def mask_config() -> LlamaConfig:
    config = LlamaConfig(
        hidden_size=8, num_attention_heads=2, num_hidden_layers=1, vocab_size=16
    )
    config._attn_implementation = "eager"  # noqa: leading-underscore  # production name
    config.sliding_window = 2
    return config


def qwen_tts_kwargs(config: LlamaConfig) -> dict:
    return {
        "config": config,
        "input_embeds": torch.zeros(1, 4, 8),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
        "cache_position": torch.arange(4),
        "past_key_values": None,
    }


def supported_kwargs(config: LlamaConfig) -> dict:
    return {
        "config": config,
        "inputs_embeds": torch.zeros(1, 4, 8),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
        "past_key_values": None,
    }


def test_transformers_globals_are_pristine_at_test_start() -> None:
    """Canary for patch leakage between tests in one pytest process."""
    for name, original in PRISTINE.items():
        assert getattr(masking_utils, name) is original
    assert not getattr(
        generic.check_model_inputs, compat._PATCHED_FLAG, False
    )  # noqa: leading-underscore  # production name


@pytest.mark.parametrize(
    "name",
    compat._MASK_FACTORY_NAMES,  # noqa: leading-underscore  # production name
)
def test_unpatched_transformers_rejects_the_qwen_tts_call_shape(name: str) -> None:
    with pytest.raises(TypeError, match="input_embeds"):
        getattr(masking_utils, name)(**qwen_tts_kwargs(mask_config()))


@pytest.mark.parametrize(
    "name",
    compat._MASK_FACTORY_NAMES,  # noqa: leading-underscore  # production name
)
def test_shim_accepts_input_embeds_and_absorbs_cache_position(name: str) -> None:
    config = mask_config()
    compat.apply_qwen_tts_transformers_compatibility_patches()

    shimmed = getattr(masking_utils, name)(**qwen_tts_kwargs(config))
    expected = PRISTINE[name](**supported_kwargs(config))

    assert shimmed is not None
    torch.testing.assert_close(shimmed, expected)


@pytest.mark.parametrize(
    "name",
    compat._MASK_FACTORY_NAMES,  # noqa: leading-underscore  # production name
)
def test_shim_passes_the_supported_call_shape_through(name: str) -> None:
    config = mask_config()
    compat.apply_qwen_tts_transformers_compatibility_patches()

    torch.testing.assert_close(
        getattr(masking_utils, name)(**supported_kwargs(config)),
        PRISTINE[name](**supported_kwargs(config)),
    )


@pytest.mark.parametrize(
    "name",
    compat._MASK_FACTORY_NAMES,  # noqa: leading-underscore  # production name
)
def test_shim_passes_positional_arguments_through(name: str) -> None:
    """The patch is global, so Transformers' own positional callers must work."""
    config = mask_config()
    compat.apply_qwen_tts_transformers_compatibility_patches()

    torch.testing.assert_close(
        getattr(masking_utils, name)(
            config, torch.zeros(1, 4, 8), torch.ones(1, 4, dtype=torch.long), None
        ),
        PRISTINE[name](**supported_kwargs(config)),
    )


def test_shim_prefers_inputs_embeds_when_both_spellings_are_given() -> None:
    config = mask_config()
    compat.apply_qwen_tts_transformers_compatibility_patches()

    mask = masking_utils.create_causal_mask(
        config=config,
        input_embeds=torch.zeros(1, 4, 8),
        inputs_embeds=torch.zeros(1, 2, 8),
        attention_mask=torch.ones(1, 2, dtype=torch.long),
        past_key_values=None,
    )

    assert mask.shape == (1, 1, 2, 2)


def test_shim_is_idempotent() -> None:
    compat.apply_qwen_tts_transformers_compatibility_patches()
    patched = {
        name: getattr(masking_utils, name)
        for name in compat._MASK_FACTORY_NAMES  # noqa: leading-underscore  # production name
    }
    for name, fn in patched.items():
        assert fn is not PRISTINE[name]

    compat.apply_qwen_tts_transformers_compatibility_patches()

    for name, fn in patched.items():
        assert getattr(masking_utils, name) is fn


def test_shim_is_a_no_op_when_transformers_already_accepts_input_embeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def already_compatible(config, input_embeds, cache_position=None):  # noqa: ARG001
        return None

    monkeypatch.setattr(masking_utils, "create_causal_mask", already_compatible)

    compat.apply_qwen_tts_transformers_compatibility_patches()

    assert masking_utils.create_causal_mask is already_compatible
