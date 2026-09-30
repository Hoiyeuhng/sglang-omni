"""Opt-in real-server checks for Breeze TTS 2 on Apple Silicon."""

import base64
import io
import os
import time
from collections.abc import Iterator

import httpx
import numpy as np
import pytest
import soundfile as sf

BASE_URL = os.environ.get("BREEZE_TEST_BASE_URL")
pytestmark = pytest.mark.skipif(not BASE_URL, reason="Set BREEZE_TEST_BASE_URL")


@pytest.fixture(scope="module")
def client() -> Iterator[httpx.Client]:
    assert BASE_URL is not None
    with httpx.Client(base_url=BASE_URL, timeout=120) as connection:
        yield connection


@pytest.fixture(scope="module")
def reference_audio(client: httpx.Client) -> bytes:
    response = client.post(
        "/v1/audio/speech",
        json={
            "input": "Hello, this is Breeze speaking on a Mac.",
            "instructions": "A warm female voice speaking clearly.",
            "seed": 42,
            "cfg_scale": 4,
            "max_new_tokens": 100,
        },
    )
    response.raise_for_status()
    return response.content


@pytest.mark.parametrize(
    "text,instructions",
    [
        (
            "Hello, this is Breeze speaking on a Mac.",
            "A warm female voice speaking clearly.",
        ),
        (
            "今天天气很好，我们一起去公园散步吧。",
            "一位温柔的女性，语气自然，吐字清晰。",
        ),
    ],
)
def test_streaming_matches_complete_audio(
    client: httpx.Client, text: str, instructions: str
) -> None:
    request = {
        "input": text,
        "instructions": instructions,
        "seed": 42,
        "cfg_scale": 4,
        "max_new_tokens": 100,
    }
    complete = client.post("/v1/audio/speech", json=request)
    complete.raise_for_status()
    waveform, sample_rate = sf.read(io.BytesIO(complete.content), dtype="int16")
    assert sample_rate == 24000
    assert waveform.size > 0
    chunks: list[bytes] = []
    arrival_seconds: list[float] = []
    started = time.monotonic()
    with client.stream(
        "POST",
        "/v1/audio/speech",
        json={**request, "stream": True, "response_format": "pcm"},
    ) as response:
        response.raise_for_status()
        for chunk in response.iter_bytes():
            chunks.append(chunk)
            arrival_seconds.append(time.monotonic() - started)
    assert len(chunks) > 1
    assert arrival_seconds[-1] > arrival_seconds[0]
    np.testing.assert_array_equal(
        waveform, np.frombuffer(b"".join(chunks), dtype="<i2")
    )


@pytest.mark.parametrize("instructions", ["", "Speak softly and slowly."])
def test_reference_clone_and_direction(
    client: httpx.Client, reference_audio: bytes, instructions: str
) -> None:
    response = client.post(
        "/v1/audio/speech",
        json={
            "input": "This is a voice cloning test on Apple Silicon.",
            "ref_audio": "data:audio/wav;base64,"
            + base64.b64encode(reference_audio).decode(),
            "ref_text": "Hello, this is Breeze speaking on a Mac.",
            "instructions": instructions,
            "cfg_scale": 4,
            "seed": 42,
            "max_new_tokens": 150,
        },
    )
    response.raise_for_status()
    waveform, sample_rate = sf.read(io.BytesIO(response.content))
    assert sample_rate == 24000
    assert waveform.size > 0
    assert np.isfinite(waveform).all()
    assert np.abs(waveform).max() > 0


@pytest.mark.parametrize(
    "options",
    [
        {"voice": "missing"},
        {"speed": 2},
        {"ref_text": "No audio"},
        {"language": "French"},
    ],
)
def test_request_errors_are_client_errors(
    client: httpx.Client, options: dict[str, str | int]
) -> None:
    response = client.post("/v1/audio/speech", json={"input": "Hello", **options})
    assert response.status_code == 400, response.text


def test_disconnect_does_not_block_next_request(client: httpx.Client) -> None:
    with client.stream(
        "POST",
        "/v1/audio/speech",
        json={
            "input": "This sentence is repeated to test cancellation. " * 20,
            "instructions": "A calm voice.",
            "stream": True,
            "response_format": "pcm",
            "seed": 42,
        },
    ) as response:
        response.raise_for_status()
        assert next(response.iter_bytes())
    response = client.post(
        "/v1/audio/speech",
        json={"input": "Hello again.", "instructions": "A calm voice.", "seed": 42},
        timeout=30,
    )
    response.raise_for_status()
    waveform, _ = sf.read(io.BytesIO(response.content))
    assert waveform.size > 0
