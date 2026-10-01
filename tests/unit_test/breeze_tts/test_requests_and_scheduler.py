"""Breeze API conversion and scheduler lifecycle without external weights."""

from collections.abc import Generator
from threading import Event, Thread

import numpy as np
import pytest
from numpy.typing import NDArray

from sglang_omni.models.breeze_tts.config import BreezeTTSPipelineConfig
from sglang_omni.models.breeze_tts.request import (
    BreezeRequestError,
    BreezeSpeechRequest,
    parse_request,
)
from sglang_omni.models.breeze_tts.runtime import BreezeRuntime
from sglang_omni.models.breeze_tts.stages import BreezeScheduler
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.scheduling.message import IncomingMessage
from sglang_omni.serve.protocol import CreateSpeechRequest
from sglang_omni.serve.speech_service import SpeechRequestValidator, build_tts_params


@pytest.mark.parametrize("language", ["en", "zh", "English", "Chinese", "auto"])
def test_http_language_hints_reach_breeze(language: str) -> None:
    validator = SpeechRequestValidator(
        default_model="breeze-tts-2",
        additional_speech_languages=BreezeTTSPipelineConfig.additional_speech_languages,
    )
    request = validator.parse_request({"input": "Hello", "language": language})
    stage_payload = payload(request.input)
    stage_payload.request.metadata = {"tts_params": build_tts_params(request)}
    assert parse_request(stage_payload).text == "Hello"


class ControlledRuntime(BreezeRuntime):
    def __init__(self) -> None:
        self.sample_rate_hz = 24000
        self.release: Event = Event()
        self.observed_cancellation: Event = Event()
        self.closed: Event = Event()

    def stream(
        self, request: BreezeSpeechRequest, cancelled: Event
    ) -> Generator[NDArray[np.float32], None, None]:
        try:
            yield np.array([0.1, -0.1], dtype=np.float32)
            if request.text == "wait":
                assert self.release.wait(5), "Test failed to release the generator"
            elif request.text == "fail":
                raise RuntimeError("codec failure")
            else:
                pass
            if cancelled.is_set():
                self.observed_cancellation.set()
                return
            else:
                yield np.array([0.2], dtype=np.float32)
        finally:
            self.closed.set()


def payload(text: str, *, streaming: bool = False) -> StagePayload:
    return StagePayload(
        request_id=text,
        request=OmniRequest(inputs=text, params={"stream": streaming}),
        data={},
    )


def test_http_defaults_do_not_override_model_sampling() -> None:
    request = CreateSpeechRequest(
        input="Hello", instructions="Calm voice", cfg_scale=4, seed=73
    )
    stage_payload = payload(request.input)
    stage_payload.request.params.update(temperature=0.7, top_p=0.7, top_k=20)
    stage_payload.request.metadata = {"tts_params": build_tts_params(request)}
    parsed = parse_request(stage_payload)
    assert parsed.sampling.temperature == 0.9
    assert parsed.sampling.top_p == 1.0
    assert parsed.sampling.top_k == 50
    assert parsed.sampling.cfg_scale == 4
    assert parsed.sampling.seed == 73


def test_explicit_sampling_and_inline_reference() -> None:
    stage_payload = payload("Hello")
    stage_payload.request.inputs = {
        "text": "Hello",
        "references": [
            {"text": "Reference", "data": "AAAA", "media_type": "audio/wav"}
        ],
    }
    stage_payload.request.params["temperature"] = 0.2
    parsed = parse_request(stage_payload)
    assert parsed.sampling.temperature == 0.2
    assert parsed.ref_audio == "data:audio/wav;base64,AAAA"
    assert parsed.ref_text == "Reference"


@pytest.mark.parametrize(
    "options",
    [
        {"ref_audio": "reference.wav"},
        {"ref_text": "missing audio"},
        {"voice": "unavailable"},
        {"speed": 1.5},
        {"language": "French"},
        {"task_type": "CustomVoice"},
        {"cfg_scale": float("nan")},
        {"cfg_scale": 4},
        {"token_count": 10},
        {"stream_codec_output": False},
        {"initial_codec_chunk_frames": 8},
    ],
)
def test_unsupported_or_incomplete_requests(
    options: dict[str, str | float | int]
) -> None:
    stage_payload = payload("Hello")
    stage_payload.request.metadata = {"tts_params": options}
    with pytest.raises(BreezeRequestError):
        parse_request(stage_payload)


def test_non_streaming_output_and_failure_cleanup() -> None:
    runtime = ControlledRuntime()
    scheduler = BreezeScheduler(runtime)
    result = scheduler.compute(payload("Hello"))
    waveform = np.frombuffer(result.data["audio_waveform"], dtype=np.float32)
    np.testing.assert_array_equal(
        waveform, np.array([0.1, -0.1, 0.2], dtype=np.float32)
    )
    assert scheduler.cancel_events == {}
    with pytest.raises(RuntimeError, match="codec failure"):
        scheduler.compute(payload("fail"))
    assert scheduler.cancel_events == {}
    assert runtime.closed.is_set()


def test_abort_releases_active_stream_and_next_request_completes() -> None:
    runtime = ControlledRuntime()
    scheduler = BreezeScheduler(runtime)
    worker = Thread(target=scheduler.start)
    worker.start()
    try:
        scheduler.enqueue(
            IncomingMessage("wait", "new_request", payload("wait", streaming=True))
        )
        first = scheduler.outbox.get(timeout=5)
        assert first.type == "stream"
        assert not runtime.release.is_set()
        assert not runtime.closed.is_set()
        scheduler.abort("wait")
        runtime.release.set()
        assert runtime.observed_cancellation.wait(5)
        scheduler.enqueue(IncomingMessage("next", "new_request", payload("next")))
        result = scheduler.outbox.get(timeout=5)
        assert result.request_id == "next"
        assert result.type == "result"
        assert scheduler.outbox.empty()
    finally:
        runtime.release.set()
        scheduler.stop()
        worker.join(timeout=5)
    assert not worker.is_alive()
    assert scheduler.cancel_events == {}


def test_stream_failure_emits_error_and_worker_recovers() -> None:
    runtime = ControlledRuntime()
    scheduler = BreezeScheduler(runtime)
    worker = Thread(target=scheduler.start)
    worker.start()
    try:
        scheduler.enqueue(
            IncomingMessage("fail", "new_request", payload("fail", streaming=True))
        )
        assert scheduler.outbox.get(timeout=5).type == "stream"
        failure = scheduler.outbox.get(timeout=5)
        assert failure.type == "error"
        assert "codec failure" in str(failure.data)
        assert runtime.closed.wait(5)
        scheduler.enqueue(IncomingMessage("next", "new_request", payload("next")))
        result = scheduler.outbox.get(timeout=5)
        assert result.type == "result"
        assert result.request_id == "next"
    finally:
        scheduler.stop()
        worker.join(timeout=5)
    assert not worker.is_alive()
    assert scheduler.cancel_events == {}


def test_shutdown_cancels_active_generation() -> None:
    runtime = ControlledRuntime()
    scheduler = BreezeScheduler(runtime)
    worker = Thread(target=scheduler.start)
    worker.start()
    try:
        scheduler.enqueue(
            IncomingMessage("wait", "new_request", payload("wait", streaming=True))
        )
        assert scheduler.outbox.get(timeout=5).type == "stream"
        scheduler.stop()
        runtime.release.set()
        assert runtime.observed_cancellation.wait(5)
    finally:
        runtime.release.set()
        scheduler.stop()
        worker.join(timeout=5)
    assert not worker.is_alive()
    assert scheduler.cancel_events == {}
