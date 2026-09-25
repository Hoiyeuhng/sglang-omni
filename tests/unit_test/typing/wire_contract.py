# SPDX-License-Identifier: Apache-2.0
"""Check ordinary typed dataclass fields declared through wire."""

from dataclasses import dataclass

from sglang_omni.scheduling.pipeline_state import wire


@dataclass
class WireCounter:
    count: int = wire(0, codec="int")
    label: str = wire("", codec="str")
    duration: float = wire(0.0, codec="float")
