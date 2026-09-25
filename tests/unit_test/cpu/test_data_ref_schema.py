# SPDX-License-Identifier: Apache-2.0
"""Check schema generation without loading accelerator dependencies."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import msgspec
import pytest


@pytest.fixture(scope="module")
def data_ref_module() -> ModuleType:
    source = Path(__file__).resolve().parents[3] / "sglang_omni/comm/data_ref.py"
    specification = importlib.util.spec_from_file_location("schema_probe", source)
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def test_backend_reference_schema(data_ref_module: ModuleType) -> None:
    assert msgspec.json.schema(data_ref_module.BackendRef)


def test_data_reference_schema(data_ref_module: ModuleType) -> None:
    assert msgspec.json.schema(data_ref_module.DataRef)
