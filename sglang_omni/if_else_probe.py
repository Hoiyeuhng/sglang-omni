# SPDX-License-Identifier: Apache-2.0
"""Minimal branching example for exercising the if/else lint hook."""

from typing import Literal


def classify_token_count(
    token_count: int,
) -> Literal["invalid", "empty", "single", "multiple"]:
    if token_count < 0:
        return "invalid"
    elif token_count == 0:
        return "empty"
    elif token_count == 1:
        return "single"
    else:
        return "multiple"
