#!/usr/bin/env python3
"""Materialize a deterministic angry/neutral ESD subset from HF Arrow shards.

The cached ``mimba/esd-en`` dataset stores WAV bytes directly in Arrow.  This
utility selects matched neutral/angry renditions per canonical speaker split,
writes only the selected WAVs, and emits the generic CSV accepted by
``ste-optimized build-data``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc

from ste_optimized.config import SPEAKER_PARTITION


DEFAULT_PER_SPEAKER = {
    "train": 50,       # 4 speakers -> 200 contrasts
    "validation": 20,  # 2 speakers -> 40 contrasts
    "test": 20,        # 2 speakers -> 40 contrasts
    "reserve": 10,     # held back from model selection
}


def _batches(path: Path):
    with pa.memory_map(str(path), "r") as source:
        yield from ipc.open_stream(source)


def _stable_rank(seed: int, speaker: str, index: str) -> bytes:
    return hashlib.sha256(f"{seed}:{speaker}:{index}".encode()).digest()


def _normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9']+", " ", text.casefold()).strip()


def materialize(arrow_dir: Path, output: Path, seed: int) -> Path:
    shards = sorted(arrow_dir.glob("*.arrow"))
    if not shards:
        raise FileNotFoundError(f"no Arrow shards under {arrow_dir}")

    wanted_speakers = {s for speakers in SPEAKER_PARTITION.values() for s in speakers}
    variants: dict[tuple[str, str], dict[str, dict[str, str]]] = {}
    for shard in shards:
        for batch in _batches(shard):
            ids = batch.column(batch.schema.get_field_index("id")).to_pylist()
            speakers = batch.column(batch.schema.get_field_index("speaker_id")).to_pylist()
            emotions = batch.column(batch.schema.get_field_index("emotion")).to_pylist()
            texts = batch.column(batch.schema.get_field_index("text")).to_pylist()
            for utterance_id, speaker, emotion, text in zip(ids, speakers, emotions, texts):
                emotion = emotion.lower()
                if speaker not in wanted_speakers or emotion not in {"neutral", "angry"}:
                    continue
                normalized = _normalize_text(text)
                if not normalized:
                    continue
                slot = variants.setdefault((speaker, normalized), {})
                # ESD normally has one recording per speaker/text/emotion.  If
                # a cache contains duplicates, choose the lexicographically
                # first ID so selection remains reproducible.
                candidate = {"id": utterance_id, "text": text.strip()}
                if emotion not in slot or utterance_id < slot[emotion]["id"]:
                    slot[emotion] = candidate

    selected: dict[tuple[str, str], dict[str, str]] = {}
    for split, speakers in SPEAKER_PARTITION.items():
        limit = DEFAULT_PER_SPEAKER[split]
        for speaker in speakers:
            paired = [key for key, values in variants.items()
                      if key[0] == speaker and {"neutral", "angry"} <= values.keys()]
            paired.sort(key=lambda key: _stable_rank(seed, key[0], key[1]))
            if len(paired) < limit:
                raise RuntimeError(
                    f"speaker {speaker} has {len(paired)} matched pairs; need {limit}"
                )
            for key in paired[:limit]:
                values = variants[key]
                # Use the neutral raw suffix as the shared generic-CSV index;
                # build_manifest will therefore pair the two different source
                # IDs under one canonical pair ID.
                shared_index = values["neutral"]["id"].rsplit("_", 1)[-1]
                selected[key] = {
                    "index": shared_index,
                    "neutral_id": values["neutral"]["id"],
                    "angry_id": values["angry"]["id"],
                    "text": values["neutral"]["text"],
                }

    wav_root = output / "wav"
    wav_root.mkdir(parents=True, exist_ok=True)
    csv_rows: list[dict[str, str]] = []
    written: set[tuple[str, str, str]] = set()
    for shard in shards:
        for batch in _batches(shard):
            records = batch.to_pylist()
            for record in records:
                speaker = record["speaker_id"]
                emotion = record["emotion"].lower()
                key = (speaker, _normalize_text(record["text"]))
                if key not in selected or emotion not in {"neutral", "angry"}:
                    continue
                selection = selected[key]
                if record["id"] != selection[f"{emotion}_id"]:
                    continue
                index = selection["index"]
                row_key = (speaker, index, emotion)
                if row_key in written:
                    raise RuntimeError(f"duplicate selected utterance: {row_key}")
                audio_bytes = record["audio"]["bytes"]
                if not audio_bytes:
                    raise RuntimeError(f"missing WAV bytes for {record['id']} {emotion}")
                wav_path = (wav_root / f"{record['id']}_{emotion}.wav").resolve()
                wav_path.write_bytes(audio_bytes)
                csv_rows.append({
                    "path": str(wav_path),
                    "speaker": speaker,
                    "emotion": emotion,
                    # Both renditions receive the same canonical transcript;
                    # the source occasionally differs only in punctuation or
                    # case, which is still a valid lexical match.
                    "text": selection["text"],
                    "index": index,
                })
                written.add(row_key)

    expected = 2 * len(selected)
    if len(csv_rows) != expected:
        raise RuntimeError(f"wrote {len(csv_rows)} rows; expected {expected}")
    csv_rows.sort(key=lambda row: (row["speaker"], row["index"], row["emotion"]))
    csv_path = output / "source.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["path", "speaker", "emotion", "text", "index"]
        )
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"selected_pairs={len(selected)} wavs={len(csv_rows)} csv={csv_path}")
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arrow-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    materialize(args.arrow_dir, args.output, args.seed)


if __name__ == "__main__":
    main()
