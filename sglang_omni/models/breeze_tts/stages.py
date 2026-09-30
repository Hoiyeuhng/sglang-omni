# SPDX-License-Identifier: Apache-2.0
"""Serial Breeze generation with cooperative cancellation and streamed audio."""

from _thread import LockType
from contextlib import closing
from pathlib import Path
from threading import Event, Lock
from typing import Literal

import numpy as np
import torch
from numpy.typing import NDArray

from sglang_omni.models.breeze_tts.request import parse_request
from sglang_omni.models.breeze_tts.runtime import BreezeRuntime
from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.message import OutgoingMessage
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.utils.audio_payload import audio_waveform_payload
from sglang_omni.utils.checkpoint import resolve_checkpoint


class BreezeScheduler(SimpleScheduler):
    def __init__(self, runtime: BreezeRuntime) -> None:
        self.runtime: BreezeRuntime = runtime
        self.cancel_events: dict[str, Event] = {}
        self.state_lock: LockType = Lock()
        self.is_stopping: bool = False
        super().__init__(
            self.compute,
            abort_callback=self.cancel_request,
            shutdown_callback=self.cancel_all,
        )

    def cancel_request(self, request_id: str) -> None:
        with self.state_lock:
            cancellation = self.cancel_events.get(request_id)
            if cancellation is not None:
                cancellation.set()
            else:
                pass

    def cancel_all(self) -> None:
        with self.state_lock:
            self.is_stopping = True
            for cancellation in self.cancel_events.values():
                cancellation.set()

    def compute(self, payload: StagePayload) -> StagePayload:
        cancellation = Event()
        with self.state_lock:
            self.cancel_events[payload.request_id] = cancellation
            if self.is_stopping or self.is_aborted(payload.request_id):
                cancellation.set()
            else:
                pass
        try:
            request = parse_request(payload)
            is_streaming = payload.request.params.get("stream", False)
            chunks: list[NDArray[np.float32]] = []
            sample_count = 0
            with closing(self.runtime.stream(request, cancellation)) as audio_stream:
                for waveform in audio_stream:
                    if cancellation.is_set():
                        raise InterruptedError("Breeze request cancelled")
                    else:
                        pass
                    sample_count += waveform.size
                    if is_streaming:
                        self.outbox.put(
                            OutgoingMessage(
                                request_id=payload.request_id,
                                type="stream",
                                data=audio_waveform_payload(
                                    waveform,
                                    sample_rate=self.runtime.sample_rate_hz,
                                    modality="audio",
                                ),
                            )
                        )
                    else:
                        chunks.append(waveform)
            if cancellation.is_set():
                raise InterruptedError("Breeze request cancelled")
            elif sample_count == 0:
                raise RuntimeError("Breeze generated no audio frames")
            else:
                pass
            audio = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float32)
            return StagePayload(
                request_id=payload.request_id,
                request=payload.request,
                data=audio_waveform_payload(
                    audio, sample_rate=self.runtime.sample_rate_hz, modality="audio"
                ),
            )
        finally:
            with self.state_lock:
                self.cancel_events.pop(payload.request_id, None)


def create_executor(
    model_path: str,
    device: Literal["mps", "cpu"],
    dtype: Literal["bfloat16", "float32"],
    chunk_frames: int,
) -> BreezeScheduler:
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Breeze MPS inference requires an available Apple GPU")
    else:
        pass
    torch_dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]
    runtime = BreezeRuntime.from_checkpoint(
        Path(resolve_checkpoint(model_path)),
        torch.device(device),
        torch_dtype,
        chunk_frames,
    )
    return BreezeScheduler(runtime)
