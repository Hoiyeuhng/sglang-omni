# SPDX-License-Identifier: Apache-2.0
"""Check access to model request state after scheduler wrapping."""

from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData
from sglang_omni.scheduling.types import SchedulerRequest


def generation_steps(request: SGLangARRequestData) -> int:
    scheduled = SchedulerRequest(request_id="sample", data=request)
    assert scheduled.data is not None
    return scheduled.data.generation_steps
