"""Breeze generation contracts using small real Transformer modules."""

from pathlib import Path
from threading import Event

import pytest
import torch
from pydantic import ValidationError
from safetensors.torch import save_file

from sglang_omni.models.breeze_tts.model import BreezeCheckpointConfig, BreezeModel
from sglang_omni.models.breeze_tts.request import BreezeSpeechRequest
from sglang_omni.models.breeze_tts.runtime import BreezeRuntime
from sglang_omni.models.breeze_tts.sampling import BreezeSamplingParams, sample_token


class CancellingTextRuntime(BreezeRuntime):
    def __init__(self, model: BreezeModel, cancelled: Event) -> None:
        self.model = model
        self.cancelled = cancelled
        self.encoded_segments: int = 0

    def encode_text(self, text: str) -> torch.Tensor:
        self.encoded_segments += 1
        encoded = self.model.encode_text(torch.tensor([[2, 3]]))
        self.cancelled.set()
        return encoded


def test_cancelled_prompt_does_not_start_guidance_branch(model: BreezeModel) -> None:
    cancelled = Event()
    runtime = CancellingTextRuntime(model, cancelled)
    request = BreezeSpeechRequest(
        text="Hello",
        instructions="Calm voice",
        sampling=BreezeSamplingParams(cfg_scale=4),
    )
    with pytest.raises(InterruptedError, match="cancelled"):
        runtime.prepare_prompts(request, cancelled)
    assert runtime.encoded_segments == 1
    with pytest.raises(InterruptedError, match="cancelled"):
        runtime.prepare_prompts(request, cancelled)
    assert runtime.encoded_segments == 1


@pytest.fixture
def model() -> BreezeModel:
    configuration = BreezeCheckpointConfig(
        model_type="breeze",
        backbone_model_type="qwen3",
        text_encoder_proj_type="linear",
        tie_codebooks_embeddings=True,
        hidden_size=16,
        audio_embed_size=16,
        num_codebooks=3,
        vocab_size=11,
        codebook_eos_token_id=0,
        max_position_embeddings=16,
        backbone_config={
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "vocab_size": 11,
        },
        depth_decoder_config={
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "vocab_size": 11,
        },
        text_encoder_config={
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "vocab_size": 32,
            "layer_types": ["sliding_attention", "full_attention"],
            "dropout_rate": 0.0,
        },
        codec_config={"codebook_size": 8, "sampling_rate": 24000},
    )
    with torch.random.fork_rng():
        torch.manual_seed(17)
        network = BreezeModel(configuration).eval()
        torch.nn.init.normal_(network.depth_decoder.codebooks_head.weight, std=0.02)
        network.lm_head.weight.data.zero_()
    return network


def test_frame_limit_and_codebook_ranges(model: BreezeModel) -> None:
    prompt = model.encode_text(torch.tensor([[2, 3, 4]]))
    frames = list(
        model.generate_frames(
            [prompt], BreezeSamplingParams(max_new_tokens=3, temperature=0), Event()
        )
    )
    assert len(frames) == 3
    assert torch.stack(frames).shape == (3, 3)
    assert all(bool(((frame >= 0) & (frame < 8)).all()) for frame in frames)


def test_cancellation_between_frames_and_before_prefill(model: BreezeModel) -> None:
    cancellation = Event()
    prompt = model.encode_text(torch.tensor([[2, 3]]))
    frames = model.generate_frames(
        [prompt], BreezeSamplingParams(max_new_tokens=3, temperature=0), cancellation
    )
    assert next(frames).shape == (3,)
    cancellation.set()
    assert list(frames) == []
    assert list(model.generate_frames([], BreezeSamplingParams(), cancellation)) == []


def test_context_limit_and_empty_generation_window(model: BreezeModel) -> None:
    prompt = model.encode_text(torch.tensor([[2] * 15]))
    params = BreezeSamplingParams(max_new_tokens=3, temperature=0)
    assert len(list(model.generate_frames([prompt], params, Event()))) == 1
    with pytest.raises(ValueError, match="no generation room"):
        list(model.generate_frames([prompt.repeat(1, 2, 1)], params, Event()))


def test_guidance_requests_have_independent_rng_and_cache(model: BreezeModel) -> None:
    prompts = [
        model.encode_text(torch.tensor([[2, 3, 4]])),
        model.encode_text(torch.tensor([[2, 5]])),
    ]
    params = BreezeSamplingParams(max_new_tokens=2, seed=73, cfg_scale=4)
    first = list(model.generate_frames(prompts, params, Event()))
    list(
        model.generate_frames(prompts, params.model_copy(update={"seed": 74}), Event())
    )
    repeated = list(model.generate_frames(prompts, params, Event()))
    assert len(first) == 2
    assert len(first) == len(repeated)
    assert all(torch.equal(left, right) for left, right in zip(first, repeated))


def test_backbone_eos_stops_before_emitting_a_frame(model: BreezeModel) -> None:
    prompt = model.encode_text(torch.tensor([[2, 3, 4]]))
    with torch.no_grad():
        hidden = model.backbone_model(inputs_embeds=prompt).last_hidden_state[0, -1]
        model.lm_head.weight[-1].copy_(hidden)
    assert (
        list(
            model.generate_frames(
                [prompt], BreezeSamplingParams(temperature=0), Event()
            )
        )
        == []
    )


def test_checkpoint_roundtrip_and_missing_weight_rejection(
    model: BreezeModel, tmp_path: Path
) -> None:
    (tmp_path / "config.json").write_text(model.configuration.model_dump_json())
    parameters = model.state_dict()
    save_file(parameters, tmp_path / "model.safetensors")
    restored = BreezeModel.from_checkpoint(tmp_path, torch.device("cpu"), torch.float32)
    input_ids = torch.tensor([[2, 3, 4]])
    params = BreezeSamplingParams(max_new_tokens=2, temperature=0)
    original_frames = list(
        model.generate_frames([model.encode_text(input_ids)], params, Event())
    )
    restored_frames = list(
        restored.generate_frames([restored.encode_text(input_ids)], params, Event())
    )
    assert len(original_frames) == len(restored_frames) == 2
    assert all(
        torch.equal(left, right)
        for left, right in zip(original_frames, restored_frames)
    )
    parameters.pop("lm_head.weight")
    save_file(parameters, tmp_path / "model.safetensors")
    with pytest.raises(RuntimeError, match="lm_head.weight"):
        BreezeModel.from_checkpoint(tmp_path, torch.device("cpu"), torch.float32)


@pytest.mark.parametrize("allow_eos,expected", [(True, 11), (False, 0)])
def test_reserved_tokens_and_backbone_eos(allow_eos: bool, expected: int) -> None:
    logits = torch.zeros(1, 12)
    logits[0, 8:11] = 100
    logits[0, 11] = 50
    token = sample_token(
        logits,
        BreezeSamplingParams(temperature=0),
        torch.Generator(),
        [],
        codebook_size=8,
        allow_eos=allow_eos,
    )
    assert token == expected


def test_repetition_penalty_changes_greedy_selection() -> None:
    logits = torch.tensor([[4.0, 3.0, -1.0]])
    assert (
        sample_token(
            logits,
            BreezeSamplingParams(temperature=0, repetition_penalty=2),
            torch.Generator(),
            [0],
            codebook_size=3,
            allow_eos=False,
        )
        == 1
    )


@pytest.mark.parametrize("temperature", [0.0, 0.9])
def test_nonfinite_logits_fail_instead_of_generating_silent_audio(
    temperature: float,
) -> None:
    with pytest.raises(RuntimeError, match="invalid sampling logits"):
        sample_token(
            torch.tensor([[float("nan"), 1.0]]),
            BreezeSamplingParams(temperature=temperature),
            torch.Generator(),
            [],
            codebook_size=2,
            allow_eos=False,
        )


@pytest.mark.parametrize(
    "values",
    [
        {"temperature": float("nan")},
        {"cfg_scale": float("inf")},
        {"top_p": 0},
        {"max_new_tokens": 0},
        {"seed": -1},
        {"seed": True},
    ],
)
def test_invalid_sampling_parameters(values: dict[str, float | int]) -> None:
    with pytest.raises(ValidationError):
        BreezeSamplingParams.model_validate(values)
