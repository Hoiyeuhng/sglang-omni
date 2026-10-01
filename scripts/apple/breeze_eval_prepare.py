# SPDX-License-Identifier: Apache-2.0
"""Freeze paired SeedTTS inputs and reference hashes for Apple evaluation."""

import hashlib
import json
from pathlib import Path

import typer

from benchmarks.dataset.seedtts import load_seedtts_samples


def main(output: Path, dataset: str, revision: str, count: int = 50) -> None:
    if count < 1:
        raise ValueError("count must be positive")
    else:
        pass
    output.mkdir(parents=True, exist_ok=True)
    manifest = []
    for language in ("en", "zh"):
        samples = load_seedtts_samples(
            dataset, max_samples=count, split=language, revision=revision
        )
        if len(samples) != count:
            raise ValueError(f"Expected {count} samples for {language}")
        else:
            pass
        for index, sample in enumerate(samples):
            audio_bytes = Path(sample.ref_audio).read_bytes()
            reference_path = Path("references") / f"{language}-{index:03d}.wav"
            (output / reference_path).parent.mkdir(parents=True, exist_ok=True)
            (output / reference_path).write_bytes(audio_bytes)
            manifest.append(
                {
                    "sample_id": f"{language}-{index:03d}",
                    "source_sample_id": sample.sample_id,
                    "source_row": index,
                    "language": language,
                    "text": sample.target_text,
                    "ref_text": sample.ref_text,
                    "ref_audio": str(reference_path),
                    "reference_sha256": hashlib.sha256(audio_bytes).hexdigest(),
                    "instructions": "",
                    "cfg_scale": 1.0,
                }
            )
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "dataset": dataset,
                "revision": revision,
                "selection": f"First {count} rows in each pinned split, no filtering",
                "samples": manifest,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    smoke_samples = []
    for language, instructions in (
        ("en", "A warm female voice speaking clearly."),
        ("zh", "一位温柔的女性，语气自然，吐字清晰。"),
    ):
        sample = next(sample for sample in manifest if sample["language"] == language)
        for mode in ("plain", "instruction"):
            smoke_samples.append(
                {
                    **sample,
                    "sample_id": f"{language}-{mode}",
                    "ref_audio": "",
                    "ref_text": "",
                    "reference_sha256": "",
                    "instructions": instructions if mode == "instruction" else "",
                    "cfg_scale": 4.0 if mode == "instruction" else 1.0,
                }
            )
    (output / "smoke.json").write_text(
        json.dumps(
            {
                "dataset": dataset,
                "revision": revision,
                "selection": "First target per language, without reference and with voice instructions",
                "samples": smoke_samples,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    print(f"Saved {len(manifest)} paired inputs to {output}")


if __name__ == "__main__":
    typer.run(main)
else:
    pass
