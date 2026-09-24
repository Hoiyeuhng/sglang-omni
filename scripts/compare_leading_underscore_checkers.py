#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare check_leading_underscore.py with the ast-grep rule and print a Markdown report."""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import statistics
import subprocess
import sys
import tarfile
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[1]
RULE_ID = "leading-underscore"
TIMING_RUNS = 5
SINGLE_FILE = "sglang_omni/models/ming_tts/audio_decode.py"

Site = tuple[str, int, str]


@dataclass(frozen=True, kw_only=True)
class EdgeCase:
    name: str
    source: str
    expected: list[tuple[int, str]]


EDGE_CASES = [
    EdgeCase(
        name="noqa in class body keeps the class name checked",
        source="class _Hidden:\n    def method(self) -> int:\n        return 1  # noqa: leading-underscore\n",
        expected=[(1, "_Hidden")],
    ),
    EdgeCase(
        name="noqa in function body keeps the function name checked",
        source="def _helper() -> int:\n    value = 1  # noqa: leading-underscore\n    return value\n",
        expected=[(1, "_helper")],
    ),
    EdgeCase(
        name="noqa in block body keeps with/if/for headers checked",
        source=(
            "def run(request) -> None:\n"
            "    with request._lock:\n"
            "        is_locked = True  # noqa: leading-underscore\n"
            "    if request._ready:\n"
            "        is_ready = True  # noqa: leading-underscore\n"
        ),
        expected=[(2, "_lock"), (4, "_ready")],
    ),
    EdgeCase(
        name="attribute assignment is reported once",
        source="def attach(request) -> None:\n    request._cache_key = 1\n",
        expected=[(2, "_cache_key")],
    ),
    EdgeCase(
        name="wrapped statement with noqa on the closing line",
        source="value = (\n    request._cache_key\n)  # noqa: leading-underscore\n",
        expected=[],
    ),
]


def run_command(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False)


def export_tree(ref: str, destination: Path) -> Path:
    archive = destination / f"{ref.replace('/', '_')}.tar"
    subprocess.run(
        ["git", "archive", "--output", str(archive), ref], cwd=REPO_ROOT, check=True
    )
    tree = destination / ref.replace("/", "_")
    with tarfile.open(archive) as bundle:
        bundle.extractall(tree, filter="data")
    return tree


def load_checker(tree: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        f"checker_{tree.name}", tree / "scripts" / "check_leading_underscore.py"
    )
    assert spec is not None and spec.loader is not None
    checker = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = checker
    spec.loader.exec_module(checker)
    return checker


def script_sites(checker: ModuleType) -> list[Site]:
    return [
        (checker.repo_relative(path), violation.lineno, violation.name)
        for path in checker.iter_default_files()
        for violation in checker.check_file(path)
    ]


def raw_script_sites(checker: ModuleType) -> list[Site]:
    original_has_noqa, original_allowed = checker.has_noqa, checker.ALLOWED_DEFS
    checker.has_noqa = lambda line: False
    checker.ALLOWED_DEFS = frozenset()
    try:
        return script_sites(checker)
    finally:
        checker.has_noqa, checker.ALLOWED_DEFS = original_has_noqa, original_allowed


def header_only_script_sites(checker: ModuleType) -> list[Site]:
    """Sites the script reports once a compound statement's noqa covers only its header."""

    def statement_has_noqa(self: ast.NodeVisitor, node: ast.AST) -> bool:
        current: ast.AST | None = node
        while current is not None and not isinstance(current, ast.stmt):
            current = getattr(current, "parent", None)
        if current is None:
            return False
        decorators = getattr(current, "decorator_list", [])
        start = min([current.lineno, *(item.lineno for item in decorators)])
        body = getattr(current, "body", None)
        if isinstance(body, list) and body:
            end = body[0].lineno - 1
        else:
            end = current.end_lineno or current.lineno
        return any(
            checker.has_noqa(self.source_lines[index])
            for index in range(start - 1, end)
        )

    visitor_class = checker.LeadingUnderscoreVisitor
    original = visitor_class.statement_has_noqa
    visitor_class.statement_has_noqa = statement_has_noqa
    try:
        return script_sites(checker)
    finally:
        visitor_class.statement_has_noqa = original


def unused_noqa_lines(checker: ModuleType) -> list[tuple[str, int]]:
    """Noqa comments the script never needs: not on a site's line or in its statement."""
    spans: dict[str, list[tuple[int, int, int]]] = {}

    class SpanVisitor(checker.LeadingUnderscoreVisitor):
        def _record_name(
            self, name: str, lineno: int, column: int, kind: str, node: ast.AST
        ) -> None:
            if not checker.is_leading_underscore_name(name):
                return
            current: ast.AST | None = node
            while current is not None and not isinstance(current, ast.stmt):
                current = getattr(current, "parent", None)
            start = current.lineno if current is not None else lineno
            end = (current.end_lineno or start) if current is not None else lineno
            spans.setdefault(checker.repo_relative(self.path), []).append(
                (lineno, start, end)
            )

    unused: list[tuple[str, int]] = []
    for path in checker.iter_default_files():
        source = path.read_text(encoding="utf-8")
        SpanVisitor(path, source.splitlines()).visit(ast.parse(source))
        relative = checker.repo_relative(path)
        file_spans = spans.get(relative, [])
        for index, line in enumerate(source.splitlines(), start=1):
            if not checker.has_noqa(line):
                continue
            if not any(
                index == lineno or start <= index <= end
                for lineno, start, end in file_spans
            ):
                unused.append((relative, index))
    return unused


@dataclass(frozen=True, kw_only=True)
class AstGrepMatch:
    rule_id: str
    file: str
    line: int
    text: str


def ast_grep_scan(binary: str, tree: Path, paths: list[str]) -> list[AstGrepMatch]:
    result = run_command([binary, "scan", "--json=stream", *paths], tree)
    matches = []
    for raw_line in result.stdout.splitlines():
        decoded = json.loads(raw_line)
        assert isinstance(decoded, dict)
        match_range = decoded["range"]
        assert isinstance(match_range, dict) and isinstance(match_range["start"], dict)
        matches.append(
            AstGrepMatch(
                rule_id=str(decoded["ruleId"]),
                file=str(decoded["file"]),
                line=int(match_range["start"]["line"]) + 1,
                text=str(decoded["text"]),
            )
        )
    return matches


def ast_grep_split(matches: list[AstGrepMatch]) -> tuple[list[Site], int]:
    """Split matches into rule sites and unused ignore comments."""
    sites = [
        (match.file, match.line, match.text)
        for match in matches
        if match.rule_id == RULE_ID and not match.text.startswith("#")
    ]
    return sites, len(matches) - len(sites)


def median_seconds(command: list[str], cwd: Path) -> float:
    durations = []
    for _ in range(TIMING_RUNS):
        started = time.perf_counter()
        run_command(command, cwd)
        durations.append(time.perf_counter() - started)
    return statistics.median(durations)


def edge_case_results(
    checker: ModuleType, binary: str, rule_tree: Path, workspace: Path
) -> list[tuple[str, bool, bool]]:
    probe_root = workspace / "edge"
    (probe_root / "sglang_omni").mkdir(parents=True)
    for name in ["sgconfig.yml", ".ast-grep"]:
        run_command(["cp", "-R", str(rule_tree / name), str(probe_root)], workspace)
    results: list[tuple[str, bool, bool]] = []
    for case in EDGE_CASES:
        script_probe = workspace / "script_probe.py"
        script_probe.write_text(case.source, encoding="utf-8")
        found = sorted(
            (violation.lineno, violation.name)
            for violation in checker.check_file(script_probe)
        )
        script_ok = found == case.expected
        grep_probe = probe_root / "sglang_omni" / "probe.py"
        grep_probe.write_text(
            case.source.replace(
                "# noqa: leading-underscore", "# ast-grep-ignore: leading-underscore"
            ),
            encoding="utf-8",
        )
        sites, _ = ast_grep_split(ast_grep_scan(binary, probe_root, []))
        grep_ok = sorted((lineno, name) for _, lineno, name in sites) == case.expected
        results.append((case.name, script_ok, grep_ok))
    return results


def count_lines(path: Path) -> int:
    return sum(
        1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    )


def rule_added_commit() -> str:
    result = subprocess.run(
        ["git", "log", "--format=%H", "--diff-filter=A", "-1", "--", "sgconfig.yml"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--before", help="Ref with noqa comments (default: before the rule was added)"
    )
    parser.add_argument(
        "--after", default="HEAD", help="Ref with the ast-grep rule and ignores"
    )
    parser.add_argument("--ast-grep", default="ast-grep", help="ast-grep binary")
    args = parser.parse_args()
    before_ref = args.before or f"{rule_added_commit()}^"

    with tempfile.TemporaryDirectory() as workspace_name:
        workspace = Path(workspace_name)
        before = export_tree(before_ref, workspace)
        after = export_tree(args.after, workspace)
        for name in ["sgconfig.yml", ".ast-grep"]:
            run_command(["cp", "-R", str(after / name), str(before)], workspace)
        checker = load_checker(before)

        raw_script = raw_script_sites(checker)
        raw_grep, _ = ast_grep_split(ast_grep_scan(args.ast_grep, before, []))
        normal_script = script_sites(checker)
        hidden_by_scope = header_only_script_sites(checker)
        unused_noqa = unused_noqa_lines(checker)
        after_grep, after_unused = ast_grep_split(
            ast_grep_scan(args.ast_grep, after, [])
        )
        script_on_after = script_sites(load_checker(after))
        edge_results = edge_case_results(checker, args.ast_grep, after, workspace)

        script_command = [sys.executable, "scripts/check_leading_underscore.py"]
        timing = {
            "full tree": (
                median_seconds(script_command, before),
                median_seconds([args.ast_grep, "scan"], after),
            ),
            f"one file ({SINGLE_FILE})": (
                median_seconds([*script_command, SINGLE_FILE], before),
                median_seconds([args.ast_grep, "scan", SINGLE_FILE], after),
            ),
        }
        script_lines = count_lines(before / "scripts" / "check_leading_underscore.py")
        rule_lines = count_lines(
            after / ".ast-grep" / "rules" / "leading-underscore.yml"
        )

    unique_script, unique_grep = set(raw_script), set(raw_grep)
    print(f"Compared `{before_ref}` (noqa) with `{args.after}` (ast-grep).\n")
    print("### Detection, all suppressions off\n")
    print("| | script | ast-grep |\n|---|---:|---:|")
    print(f"| reports | {len(raw_script)} | {len(raw_grep)} |")
    print(f"| unique (file, line, name) | {len(unique_script)} | {len(unique_grep)} |")
    print(
        f"| duplicate reports | {len(raw_script) - len(unique_script)} | {len(raw_grep) - len(unique_grep)} |"
    )
    print(
        f"| found by both | {len(unique_script & unique_grep)} | {len(unique_script & unique_grep)} |"
    )
    print(
        f"| found only by this tool | {len(unique_script - unique_grep)} | {len(unique_grep - unique_script)} |\n"
    )
    print("### With suppressions\n")
    print("| | count |\n|---|---:|")
    print(f"| script on its own tree (noqa) | {len(normal_script)} |")
    print(f"| ast-grep on its own tree (ast-grep-ignore) | {len(after_grep)} |")
    print(f"| unused ast-grep-ignore comments | {after_unused} |")
    print(
        f"| real sites the script hides because a noqa in a block body covers the header | {len(hidden_by_scope)} |"
    )
    print(f"| noqa comments in the tree that cover nothing | {len(unused_noqa)} |")
    print(
        f"| script on the ast-grep tree (it can't read ast-grep-ignore) | {len(script_on_after)} |\n"
    )
    if hidden_by_scope:
        print("<details><summary>Sites hidden by the noqa scope</summary>\n")
        for path, lineno, name in sorted(set(hidden_by_scope)):
            print(f"- `{path}:{lineno}` `{name}`")
        print("\n</details>\n")
    if unused_noqa:
        print("<details><summary>noqa comments that cover nothing</summary>\n")
        for path, lineno in unused_noqa:
            print(f"- `{path}:{lineno}`")
        print("\n</details>\n")
    print(f"### Speed (median of {TIMING_RUNS} runs)\n")
    print("| | script | ast-grep | speedup |\n|---|---:|---:|---:|")
    for label, (script_seconds, grep_seconds) in timing.items():
        print(
            f"| {label} | {script_seconds:.2f}s | {grep_seconds:.2f}s | {script_seconds / grep_seconds:.1f}x |"
        )
    print("\n### Edge cases\n")
    print("| case | script | ast-grep |\n|---|:---:|:---:|")
    for name, script_ok, grep_ok in edge_results:
        print(
            f"| {name} | {'pass' if script_ok else 'FAIL'} | {'pass' if grep_ok else 'FAIL'} |"
        )
    print("\n### Size\n")
    print(
        f"script: {script_lines} non-blank lines (check and --fix); rule: {rule_lines} non-blank lines (check only)."
    )


if __name__ == "__main__":
    main()
