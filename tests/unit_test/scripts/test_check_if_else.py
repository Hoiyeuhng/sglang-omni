# SPDX-License-Identifier: Apache-2.0
"""Contract tests for the if/else lint hook."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
CHECKER_PATH = REPO_ROOT / "scripts" / "check_if_else.py"
PROBE_PACKAGE = "lint_if_else_probe"


def run_checker(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CHECKER_PATH), *args],
        check=False,
        capture_output=True,
        text=True,
    )


@contextmanager
def probe_model_file(source: str) -> Iterator[Path]:
    with TemporaryDirectory(
        prefix=PROBE_PACKAGE, dir=REPO_ROOT / "sglang_omni" / "models"
    ) as directory:
        probe = Path(directory) / "runner.py"
        probe.write_text(source, encoding="utf-8")
        yield probe


def test_current_sglang_omni_tree_is_clean() -> None:
    result = run_checker()
    assert result.returncode == 0, result.stderr


def test_if_elif_else_and_expressions_are_clean() -> None:
    source = """
def choose(flag, items):
    if flag:
        return 1
    elif items:
        return 2
    else:
        return 3

value = 1 if flag else 0
kept = [item for item in items if item]
"""
    with probe_model_file(source) as probe:
        result = run_checker(str(probe))
        assert result.returncode == 0, result.stderr


def test_noqa_comment_does_not_exempt_a_bare_if() -> None:
    source = "def close(session):\n    if session is not None:  # noqa: if-else\n        session.close()\n"
    with probe_model_file(source) as probe:
        result = run_checker(str(probe))
        assert result.returncode == 1
        assert "At least use `else: pass` to fix this lint" in result.stderr
        assert "python scripts/check_if_else.py --fix" in result.stderr


def test_fix_fills_nested_one_line_and_elif() -> None:
    source = (
        "def run(flag, nested):\n"
        "    if flag:\n"
        "        if nested:\n"
        "            return 1\n"
        "    elif nested:\n"
        "        return 2\n"
        "    if flag: return 3\n"
    )
    expected = (
        "def run(flag, nested):\n"
        "    if flag:\n"
        "        if nested:\n"
        "            return 1\n"
        "        else:\n"
        "            pass\n"
        "    elif nested:\n"
        "        return 2\n"
        "    else:\n"
        "        pass\n"
        "    if flag: return 3\n"
        "    else:\n"
        "        pass\n"
    )
    with probe_model_file(source) as probe:
        result = run_checker("--fix", str(probe))
        assert result.returncode == 0, result.stderr
        assert probe.read_text(encoding="utf-8") == expected
        again = run_checker(str(probe))
        assert again.returncode == 0, again.stderr


def test_new_bare_if_fails_the_default_scan() -> None:
    with probe_model_file("def load():\n    if True:\n        return 1\n") as probe:
        result = run_checker()
        assert result.returncode == 1
        assert probe.relative_to(REPO_ROOT).as_posix() in result.stderr
        assert probe.is_file()


@pytest.mark.parametrize(
    ("source", "expected_status"),
    [
        pytest.param(
            "if outer:\n    if inner:\n        pass\n    else:\n        pass\n",
            1,
            id="inner-else-does-not-complete-outer-if",
        ),
        pytest.param(
            "if outer:\n    if inner:\n        pass\nelse:\n    pass\n",
            1,
            id="outer-else-does-not-complete-inner-if",
        ),
        pytest.param(
            "if flag:\n    for value in values:\n        pass\n    else:\n        pass\n",
            1,
            id="loop-else-does-not-complete-if",
        ),
        pytest.param(
            "if first:\n    pass\nelif second:\n    pass\nelif third:\n    pass\n",
            1,
            id="elif-chain-needs-final-else",
        ),
        pytest.param(
            "if flag:\n    raise ValueError('invalid')\n",
            1,
            id="raise-still-needs-else",
        ),
        pytest.param(
            "if outer:\n    if inner:\n        pass\n    else:\n        pass\nelse:\n    pass\n",
            0,
            id="complete-nested-branches",
        ),
        pytest.param(
            "if (\n    flag\n): value = 1\nelse: value = 2\n",
            0,
            id="multiline-condition-with-inline-suites",
        ),
        pytest.param(
            "value = 1 if flag else 2\nvalues = [x for x in items if x]\n",
            0,
            id="expressions-are-not-if-statements",
        ),
    ],
)
def test_check_branch_structure_without_rewriting(
    source: str, expected_status: int
) -> None:
    with probe_model_file(source) as probe:
        original = probe.read_bytes()
        result = run_checker(str(probe))
        assert result.returncode == expected_status, result.stderr
        assert probe.read_bytes() == original


@pytest.mark.parametrize(
    "source",
    [
        pytest.param(
            "if (\n    True\n): print('value')\n",
            id="multiline-condition-with-inline-suite",
        ),
        pytest.param(
            'value = "a\u2028b"\nif True:\n    print(value)\n',
            id="unicode-line-separator-in-string",
        ),
        pytest.param(
            "if True:\n\f    print('value')\n",
            id="form-feed-in-indentation",
        ),
        pytest.param(
            "if True:\n    if True:\n        print('value')\n",
            id="nested-ifs-share-end-line",
        ),
        pytest.param(
            "if True: print('value')",
            id="no-final-newline",
        ),
    ],
)
def test_fix_preserves_execution_and_is_idempotent(source: str) -> None:
    original = subprocess.run(
        [sys.executable, "-c", source],
        check=True,
        capture_output=True,
        text=True,
    )
    with probe_model_file(source) as probe:
        result = run_checker("--fix", str(probe))
        assert result.returncode == 0, result.stderr
        rewritten = probe.read_bytes()
        actual = subprocess.run(
            [sys.executable, str(probe)],
            check=True,
            capture_output=True,
            text=True,
        )
        assert actual.stdout == original.stdout
        assert actual.stderr == original.stderr
        checked = run_checker(str(probe))
        assert checked.returncode == 0, checked.stderr
        again = run_checker("--fix", str(probe))
        assert again.returncode == 0, again.stderr
        assert probe.read_bytes() == rewritten


def test_fix_preserves_crlf_newlines() -> None:
    with probe_model_file("") as probe:
        probe.write_bytes(b"if True:\r\n    print('value')\r\n")
        result = run_checker("--fix", str(probe))
        assert result.returncode == 0, result.stderr
        rewritten = probe.read_bytes()
        assert b"\r\n" in rewritten
        assert b"\n" not in rewritten.replace(b"\r\n", b"")


def test_fix_never_leaves_invalid_python_on_disk() -> None:
    source = "if (\n    True\n): print('value')\n"
    compile(source, "probe.py", "exec")
    with probe_model_file(source) as probe:
        original = probe.read_bytes()
        result = run_checker("--fix", str(probe))
        assert result.returncode in (0, 2), result.stderr
        if result.returncode == 2:
            assert probe.read_bytes() == original
        else:
            compile(probe.read_bytes(), str(probe), "exec")
            checked = run_checker(str(probe))
            assert checked.returncode == 0, checked.stderr
