"""Opt-in full-sequence FP32 parity with a separately executed reference."""

import hashlib
import json
import os
from pathlib import Path
from threading import Event

import numpy as np
import pytest
import torch

from sglang_omni.models.breeze_tts.request import BreezeSpeechRequest
from sglang_omni.models.breeze_tts.runtime import BreezeRuntime
from sglang_omni.models.breeze_tts.sampling import BreezeSamplingParams

REFERENCE_DIRECTORY = os.environ.get("BREEZE_REFERENCE_DIRECTORY")
CHECKPOINT = os.environ.get("BREEZE_CHECKPOINT")
pytestmark = pytest.mark.skipif(
    not REFERENCE_DIRECTORY or not CHECKPOINT,
    reason="Set BREEZE_REFERENCE_DIRECTORY and BREEZE_CHECKPOINT",
)


@pytest.fixture(scope="module")
def runtime() -> BreezeRuntime:
    assert CHECKPOINT is not None
    return BreezeRuntime.from_checkpoint(
        Path(CHECKPOINT), torch.device("mps"), torch.float32, 2
    )


@pytest.mark.parametrize(
    "name,expect_eos",
    [("plain", False), ("instruction", True), ("clone", True), ("direction", True)],
)
def test_reference_full_sequence_and_eos(
    runtime: BreezeRuntime, name: str, expect_eos: bool
) -> None:
    assert REFERENCE_DIRECTORY is not None
    directory = Path(REFERENCE_DIRECTORY)
    metadata = json.loads((directory / f"{name}.json").read_text())
    assert CHECKPOINT is not None
    assert metadata["reference_revision"] == "58ec70ce5fa4cc361bdebf77ec40d1365da00ab2"
    assert metadata["dtype"] == "float32"
    assert (
        metadata["checkpoint_config_sha256"]
        == hashlib.sha256((Path(CHECKPOINT) / "config.json").read_bytes()).hexdigest()
    )
    assert (
        metadata["reference_audio_sha256"]
        == hashlib.sha256((directory / "reference.wav").read_bytes()).hexdigest()
    )
    reference = np.load(directory / f"{name}.npy")[0]
    request = metadata["request"]
    sampling = BreezeSamplingParams(
        temperature=0,
        repetition_penalty=metadata["repetition_penalty"],
        cfg_scale=metadata["cfg_scale"],
        max_new_tokens=metadata["max_new_tokens"],
    )
    speech = BreezeSpeechRequest(
        text=request["text"],
        instructions=request.get("instruction", ""),
        ref_audio=(
            str(directory / "reference.wav") if request.get("ref_audio_path") else None
        ),
        ref_text=request.get("ref_text", ""),
        sampling=sampling,
    )
    frames = list(
        runtime.model.generate_frames(
            runtime.prepare_prompts(speech, Event()), sampling, Event()
        )
    )
    actual = torch.stack(frames).cpu().numpy()
    np.save(directory / f"{name}-native.npy", actual)
    eos_rows = np.all(reference == metadata["pad_token_id"], axis=1)
    assert bool(eos_rows.any()) == expect_eos
    if expect_eos:
        eos_frame = int(np.flatnonzero(eos_rows)[0])
        assert eos_frame < sampling.max_new_tokens - 1
        assert eos_rows[eos_frame:].all()
        np.testing.assert_array_equal(actual, reference[:eos_frame])
    else:
        assert len(reference) == sampling.max_new_tokens
        np.testing.assert_array_equal(actual, reference)


@pytest.mark.parametrize("chunk_frames", [1, 2, 7, 32])
@torch.inference_mode()
def test_reference_codes_decode_equally_across_chunk_sizes(
    runtime: BreezeRuntime, chunk_frames: int
) -> None:
    assert REFERENCE_DIRECTORY is not None
    reference = np.load(Path(REFERENCE_DIRECTORY) / "plain.npy")[0]
    codes = torch.from_numpy(reference).to(runtime.device).T.unsqueeze(0)
    expected = runtime.audio_tokenizer.model.decoder(codes).float().cpu()
    state = runtime.codec.init_state(
        batch_size=1, device=runtime.device, dtype=torch.float32
    )
    chunks = [
        runtime.codec.decode(part, state).float().cpu()
        for part in codes.split(chunk_frames, dim=-1)
    ]
    actual = torch.cat(chunks, dim=-1)
    assert actual.shape[-1] == len(reference) * runtime.codec.total_upsample
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)
