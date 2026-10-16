"""Prepare a Voxt XCTest configuration without accessing normal user storage."""

import argparse
import json
import plistlib
import re
from pathlib import Path

DISABLED_ENVIRONMENT_KEYS = (
    "VOXT_ASR_BACKEND",
    "VOXT_OMNI_RUNTIME",
    "VOXT_VAD_CLIP",
    "VOXT_DIARIZATION_CLIP",
    "VOXT_LONGFORM_REPLAY_CLIP",
    "VOXT_LIFECYCLE_REPO",
    "VOXT_LIFECYCLE_CLIPS",
    "VOXT_LIFECYCLE_OUT",
    "VOXT_LIFECYCLE_ROUNDS",
    "VOXT_BENCH_MANIFEST",
    "VOXT_BENCH_CLIPS",
    "VOXT_BENCH_IDS",
    "VOXT_BENCH_OUT",
    "VOXT_BENCH_RUN",
    "VOXT_LLM_SMOKE_ENHANCEMENT_TEXT",
    "VOXT_LLM_SMOKE_TRANSLATION_TEXT",
    "VOXT_LLM_SMOKE_REWRITE_PROMPT",
    "VOXT_LLM_SMOKE_TASK",
    "VOXT_LLM_SMOKE_REWRITE_SOURCE_TEXT",
    "VOXT_LLM_SMOKE_LOCAL_REPO",
    "VOXT_LLM_SMOKE_ITERATIONS",
    "VOXT_LLM_SMOKE_PREFILL_STEP",
    "VOXT_LLM_SMOKE_TARGET_LANGUAGE",
    "VOXT_UPDATE_CHANNEL",
)


def prepare_xctestrun(
    source_path: Path, isolation_directory: Path, bundle_prefix: str
) -> Path:
    source_path = source_path.resolve()
    isolation_directory = isolation_directory.resolve()
    if re.fullmatch(r"com\.voxt\.validation\.r[0-9a-f]{32}", bundle_prefix) is None:
        raise ValueError("A unique validation bundle prefix is required")
    else:
        pass
    configuration = plistlib.loads(source_path.read_bytes())
    target = configuration["VoxtTests"]
    products_directory = source_path.parent.resolve()
    host_directory = products_directory / "TestDebug/Voxt.app"
    host_identifier = f"{bundle_prefix}.Voxt.testhost"
    test_identifier = f"{bundle_prefix}.VoxtTests.testhost"
    host_metadata = plistlib.loads((host_directory / "Contents/Info.plist").read_bytes())
    test_metadata = plistlib.loads(
        (
            host_directory
            / "Contents/PlugIns/VoxtTests.xctest/Contents/Info.plist"
        ).read_bytes()
    )
    if (
        host_metadata["CFBundleIdentifier"] != host_identifier
        or test_metadata["CFBundleIdentifier"] != test_identifier
        or target["TestHostPath"] != "__TESTROOT__/TestDebug/Voxt.app"
        or target["TestBundlePath"]
        != "__TESTHOST__/Contents/PlugIns/VoxtTests.xctest"
    ):
        raise ValueError("The built app and XCTest bundle must use isolated identities")
    else:
        pass
    destination_path = source_path.with_name(f"{source_path.stem}-isolated.xctestrun")
    if destination_path.exists():
        raise FileExistsError(destination_path)
    else:
        pass
    isolation_directory.mkdir(parents=True, mode=0o700, exist_ok=False)
    for relative_path in (
        "home/Library/Application Support",
        "home/Library/Caches",
        "home/Library/Preferences",
        "temporary",
        "models",
    ):
        (isolation_directory / relative_path).mkdir(parents=True, mode=0o700)
    target["TestHostBundleIdentifier"] = host_identifier
    target["BundleIdentifiersForCrashReportEmphasis"] = [
        host_identifier,
        test_identifier,
    ]
    target["EnvironmentVariables"].update(
        {environment_key: "" for environment_key in DISABLED_ENVIRONMENT_KEYS}
    )
    target["EnvironmentVariables"].update(
        {
            "CFFIXED_USER_HOME": str(isolation_directory / "home"),
            "TMPDIR": f"{isolation_directory / 'temporary'}/",
            "VOXT_RUN_MODEL_TESTS": "0",
            "VOXT_MODEL_STORAGE_ROOT": str(isolation_directory / "models"),
            "VOXT_LLM_SMOKE_PREWARM": "0",
            "VOXT_ENABLE_BETA_UPDATES": "0",
        }
    )
    with destination_path.open("xb") as destination:
        plistlib.dump(configuration, destination)
    (isolation_directory / "manifest.json").write_text(
        json.dumps(
            {
                "host_bundle_identifier": host_identifier,
                "test_bundle_identifier": test_identifier,
                "source_xctestrun": str(source_path),
                "isolated_xctestrun": str(destination_path),
                "environment": {
                    environment_key: target["EnvironmentVariables"][environment_key]
                    for environment_key in (
                        "CFFIXED_USER_HOME",
                        "TMPDIR",
                        "VOXT_RUN_MODEL_TESTS",
                        "VOXT_MODEL_STORAGE_ROOT",
                    )
                },
            },
            indent=2,
        )
        + "\n"
    )
    return destination_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xctestrun", type=Path, required=True)
    parser.add_argument("--isolation-directory", type=Path, required=True)
    parser.add_argument("--bundle-prefix", required=True)
    arguments = parser.parse_args()
    print(
        prepare_xctestrun(
            arguments.xctestrun, arguments.isolation_directory, arguments.bundle_prefix
        )
    )
