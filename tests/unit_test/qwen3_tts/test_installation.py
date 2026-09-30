"""Qwen3-TTS installation documentation and runtime hints agree."""

import ast
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
STAGES_PATH = REPO_ROOT / "sglang_omni/models/qwen3_tts/stages.py"
INSTALL_DOCS = (
    REPO_ROOT / "docs/cookbook/qwen3_tts.md",
    REPO_ROOT / "docs/basic_usage/tts.md",
)


def documented_install_commands(markdown: Path) -> list[str]:
    """The `uv pip install` lines from a page's qwen-tts prerequisites block."""
    for block in re.findall(r"```bash\n(.*?)```", markdown.read_text(), re.S):
        if "qwen-tts==" in block:
            return [
                line.strip()
                for line in block.splitlines()
                if line.strip().startswith("uv pip install")
            ]
    raise AssertionError(f"no qwen-tts install block found in {markdown}")


def runtime_install_hint() -> str:
    """_QWEN_TTS_INSTALL_HINT, read without importing stages.py (needs sglang)."""
    for node in ast.parse(STAGES_PATH.read_text()).body:
        if isinstance(node, ast.Assign) and any(
            getattr(target, "id", None) == "_QWEN_TTS_INSTALL_HINT"
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"_QWEN_TTS_INSTALL_HINT not found in {STAGES_PATH}")


def test_documented_install_commands_agree_across_pages() -> None:
    cookbook, basic_usage = (documented_install_commands(p) for p in INSTALL_DOCS)
    assert cookbook == basic_usage


def test_documented_install_commands_keep_no_deps() -> None:
    """Resolving sox or onnxruntime lifts numpy past the numba==0.65.1 ceiling,
    which breaks librosa and so `import qwen_tts`. Dropping --no-deps here has
    regressed twice."""
    for command in documented_install_commands(INSTALL_DOCS[0]):
        assert "--no-deps" in command, command
        assert "onnxruntime" not in command, command


def test_runtime_install_hint_matches_the_documented_commands() -> None:
    hint = runtime_install_hint()
    for command in documented_install_commands(INSTALL_DOCS[0]):
        assert command in hint, f"{command!r} missing from _QWEN_TTS_INSTALL_HINT"
