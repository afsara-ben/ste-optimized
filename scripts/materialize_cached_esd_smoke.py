#!/usr/bin/env python3
"""Materialize the fixed, minimal angry ESD smoke set from the HF cache.

This script is deliberately local-only: it resolves ``mimba/esd-en`` from an
existing Hugging Face Hub cache, reads the cached Parquet shards with PyArrow,
and never attempts a network download.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import wave
from dataclasses import asdict, dataclass
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

from ste_optimized.config import SPEAKER_PARTITION
from ste_optimized.data import build_manifest, read_jsonl, speaker_split


DATASET_ID = "mimba/esd-en"


@dataclass(frozen=True)
class Selection:
    source_id: str
    speaker: str
    emotion: str
    text: str
    csv_index: str
    role: str


# Text is intentionally checked byte-for-byte after Parquet decoding.  These
# fixed rows give one lexically matched angry contrast, four training neutral
# bases, and four speaker-held-out validation neutral bases.
SELECTIONS = (
    Selection("0011_000254", "0011", "Neutral", "She laughed.", "000254",
              "contrast_neutral"),
    Selection("0011_000604", "0011", "Angry", "She laughed.", "000254",
              "contrast_angry"),
    Selection("0014_000278", "0014", "Neutral", "I know you .", "000278",
              "train_base"),
    Selection("0014_000212", "0014", "Neutral", "No thank you.", "000212",
              "train_base"),
    Selection("0014_000202", "0014", "Neutral", "that sounds good.", "000202",
              "train_base"),
    Selection("0014_000098", "0014", "Neutral", "Said the witch.", "000098",
              "train_base"),
    Selection("0012_000332", "0012", "Neutral", "where are you going?", "000332",
              "validation_base"),
    Selection("0012_000296", "0012", "Neutral", "You woke me up!", "000296",
              "validation_base"),
    Selection("0012_000236", "0012", "Neutral", "How rash you are!", "000236",
              "validation_base"),
    Selection("0012_000047", "0012", "Neutral", "please excuse me.", "000047",
              "validation_base"),
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/esd-smoke-source"),
        help="directory receiving WAV files, source.csv, and selection.json",
    )
    parser.add_argument(
        "--prepared-dir",
        type=Path,
        default=Path("data/esd-smoke-prepared"),
        help="directory receiving build_manifest outputs",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="optional Hugging Face Hub cache root",
    )
    return parser.parse_args()


def _validate_selection_definition() -> None:
    ids = [row.source_id for row in SELECTIONS]
    if len(ids) != len(set(ids)):
        raise ValueError("selection contains duplicate source ids")

    contrast = [row for row in SELECTIONS if row.role.startswith("contrast_")]
    if len(contrast) != 2:
        raise ValueError("selection must contain exactly two contrast rows")
    if {row.emotion for row in contrast} != {"Neutral", "Angry"}:
        raise ValueError("contrast must contain one Neutral and one Angry row")
    if len({row.speaker for row in contrast}) != 1:
        raise ValueError("contrast rows must have the same speaker")
    if len({row.csv_index for row in contrast}) != 1:
        raise ValueError("contrast rows must map to the same generic CSV index")
    if contrast[0].text != contrast[1].text:
        raise ValueError("contrast rows must have exactly matching text")

    for row in SELECTIONS:
        expected_split = {
            "contrast_neutral": "train",
            "contrast_angry": "train",
            "train_base": "train",
            "validation_base": "validation",
        }[row.role]
        actual_split = speaker_split(row.speaker)
        if actual_split != expected_split:
            raise ValueError(
                f"{row.source_id} is in {actual_split}, expected {expected_split}"
            )


def _resolve_snapshot(cache_dir: Path | None) -> Path:
    snapshot = Path(
        snapshot_download(
            repo_id=DATASET_ID,
            repo_type="dataset",
            cache_dir=str(cache_dir) if cache_dir is not None else None,
            local_files_only=True,
        )
    )
    parquet_files = sorted((snapshot / "data").glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"no cached Parquet shards under {snapshot / 'data'}")
    return snapshot


def _read_selected_rows(snapshot: Path) -> dict[str, dict]:
    """Read only the three row groups containing selected audio payloads."""
    expected = {row.source_id: row for row in SELECTIONS}
    locations: dict[tuple[Path, int], list[tuple[int, Selection, dict]]] = {}
    seen: set[str] = set()

    for parquet_path in sorted((snapshot / "data").glob("*.parquet")):
        parquet = pq.ParquetFile(parquet_path)
        for row_group in range(parquet.num_row_groups):
            metadata = parquet.read_row_group(
                row_group, columns=["id", "speaker_id", "emotion", "text"]
            )
            for offset, row in enumerate(metadata.to_pylist()):
                source_id = row["id"]
                spec = expected.get(source_id)
                if spec is None:
                    continue
                if source_id in seen:
                    raise ValueError(f"duplicate dataset row for {source_id}")
                seen.add(source_id)
                actual = {
                    "speaker": row["speaker_id"],
                    "emotion": row["emotion"],
                    "text": row["text"],
                }
                wanted = {
                    "speaker": spec.speaker,
                    "emotion": spec.emotion,
                    "text": spec.text,
                }
                if actual != wanted:
                    raise ValueError(
                        f"metadata mismatch for {source_id}: {actual!r} != {wanted!r}"
                    )
                locations.setdefault((parquet_path, row_group), []).append(
                    (offset, spec, row)
                )

    missing = sorted(set(expected) - seen)
    if missing:
        raise ValueError(f"selected dataset rows missing from cache: {missing}")

    selected: dict[str, dict] = {}
    for (parquet_path, row_group), hits in sorted(
        locations.items(), key=lambda item: (str(item[0][0]), item[0][1])
    ):
        parquet = pq.ParquetFile(parquet_path)
        audio_column = parquet.read_row_group(row_group, columns=["audio"])["audio"]
        for offset, spec, metadata in hits:
            audio = audio_column[offset].as_py()
            wav_bytes = audio.get("bytes") if audio is not None else None
            if not isinstance(wav_bytes, bytes):
                raise ValueError(f"{spec.source_id} has no embedded WAV bytes")
            _validate_wav(spec.source_id, wav_bytes)
            selected[spec.source_id] = {
                **metadata,
                "wav_bytes": wav_bytes,
                "parquet": parquet_path.name,
                "row_group": row_group,
            }
    return selected


def _validate_wav(source_id: str, wav_bytes: bytes) -> None:
    if wav_bytes[:4] != b"RIFF" or wav_bytes[8:12] != b"WAVE":
        raise ValueError(f"{source_id} payload is not a RIFF/WAVE file")
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
            if wav.getnframes() <= 0 or wav.getframerate() <= 0:
                raise ValueError(f"{source_id} WAV contains no audio frames")
    except wave.Error as exc:
        raise ValueError(f"invalid WAV payload for {source_id}: {exc}") from exc


def _write_source(output_root: Path, selected: dict[str, dict], revision: str) -> Path:
    wav_dir = output_root / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_root / "source.csv"
    csv_rows = []
    selection_rows = []
    for spec in SELECTIONS:
        source = selected[spec.source_id]
        wav_path = (wav_dir / f"{spec.source_id}.wav").resolve()
        wav_path.write_bytes(source["wav_bytes"])
        csv_rows.append(
            {
                "path": str(wav_path),
                "speaker": spec.speaker,
                "emotion": spec.emotion.lower(),
                "text": spec.text,
                "index": spec.csv_index,
            }
        )
        selection_rows.append(
            {
                **asdict(spec),
                "wav": str(wav_path.relative_to(output_root.resolve())),
                "wav_bytes": len(source["wav_bytes"]),
                "wav_sha256": hashlib.sha256(source["wav_bytes"]).hexdigest(),
                "parquet": source["parquet"],
                "row_group": source["row_group"],
            }
        )

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["path", "speaker", "emotion", "text", "index"]
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    selection_document = {
        "schema": "ste-optimized/esd-smoke-selection/v1",
        "dataset_id": DATASET_ID,
        "snapshot_revision": revision,
        "rows": selection_rows,
    }
    (output_root / "selection.json").write_text(
        json.dumps(selection_document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return csv_path


def _verify_manifest(prepared_dir: Path, manifest: dict) -> None:
    expected_counts = {split: 0 for split in SPEAKER_PARTITION}
    expected_counts["train"] = 1
    expected_base_counts = {split: 0 for split in SPEAKER_PARTITION}
    expected_base_counts.update({"train": 4, "validation": 4})
    if manifest["counts"] != expected_counts:
        raise ValueError(
            f"unexpected pair counts: {manifest['counts']} != {expected_counts}"
        )
    if manifest["base_counts"] != expected_base_counts:
        raise ValueError(
            "unexpected base counts: "
            f"{manifest['base_counts']} != {expected_base_counts}"
        )

    pairs = read_jsonl(prepared_dir / "pairs.jsonl")
    if len(pairs) != 1 or pairs[0]["pair_id"] != "0011:000254":
        raise ValueError(f"unexpected prepared contrast pair: {pairs!r}")
    if pairs[0]["text"] != "She laughed." or pairs[0]["emotion"] != "angry":
        raise ValueError(f"prepared contrast metadata changed: {pairs[0]!r}")

    bases = read_jsonl(prepared_dir / "bases.jsonl")
    expected_bases = {
        "0014:000278",
        "0014:000212",
        "0014:000202",
        "0014:000098",
        "0012:000332",
        "0012:000296",
        "0012:000236",
        "0012:000047",
    }
    actual_bases = {row["base_id"] for row in bases}
    if actual_bases != expected_bases:
        raise ValueError(f"unexpected prepared bases: {sorted(actual_bases)}")
    for row in bases:
        if row["index"] == row["reference_index"]:
            raise ValueError(f"base reuses target as reference: {row['base_id']}")


def main() -> None:
    args = _arguments()
    _validate_selection_definition()
    output_root = args.output_root.resolve()
    prepared_dir = args.prepared_dir.resolve()
    if output_root == prepared_dir:
        raise ValueError("--output-root and --prepared-dir must be different")

    snapshot = _resolve_snapshot(args.cache_dir)
    selected = _read_selected_rows(snapshot)
    output_root.mkdir(parents=True, exist_ok=True)
    source_csv = _write_source(output_root, selected, snapshot.name)
    manifest = build_manifest(
        source=source_csv,
        emotion="angry",
        out_dir=prepared_dir,
        bases_per_speaker=4,
    )
    _verify_manifest(prepared_dir, manifest)

    summary = {
        "dataset_id": DATASET_ID,
        "snapshot_revision": snapshot.name,
        "source_csv": str(source_csv),
        "prepared_dir": str(prepared_dir),
        "wav_files": len(SELECTIONS),
        "pair_counts": manifest["counts"],
        "base_counts": manifest["base_counts"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
