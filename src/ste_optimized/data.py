"""Dataset construction from raw ESD-style audio. No prior extraction assumed.

Input layouts supported by `build_manifest`:
1. ESD official layout:  root/<speaker>/<Emotion>/<speaker>_<index>.wav with a
   per-speaker transcript file root/<speaker>/<speaker>.txt
   (tab/space-separated: filename, text, emotion).
2. Generic CSV: columns path,speaker,emotion,text,index.

Outputs (all JSONL + one hashed manifest.json):
- pairs.jsonl: one row per (speaker, index) with neutral+emotional audio paths
  and shared text, tagged with its canonical split.
- bases.jsonl: neutral voice-clone bases (per speaker, from neutral rows),
  reference = a DIFFERENT utterance of the same speaker than any target it is
  used with (enforced at sampling time).

Leakage checks FAIL CLOSED (plan §3 data prep): partition sets disjoint,
every row's speaker inside its declared split, pair ids unique per split,
no pair id across splits.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import SPEAKER_PARTITION


@dataclass
class PairRecord:
    pair_id: str          # "<speaker>:<index>"
    speaker: str
    index: str
    text: str
    split: str
    neutral_audio: str
    emotional_audio: str
    emotion: str


@dataclass
class BaseRecord:
    """A neutral rollout base: target text from one utterance, ICL reference
    from a DIFFERENT neutral utterance of the same speaker (plan: the
    reference must never be the target utterance itself)."""

    base_id: str          # "<speaker>:<target index>"
    speaker: str
    index: str            # target utterance index
    target_text: str
    reference_index: str
    reference_text: str
    reference_audio: str  # neutral recording of reference_index
    split: str


def speaker_split(speaker: str) -> str:
    for split, speakers in SPEAKER_PARTITION.items():
        if speaker in speakers:
            return split
    raise ValueError(f"speaker {speaker!r} not in the canonical partition")


def _read_esd_transcripts(speaker_dir: Path) -> dict[str, tuple[str, str]]:
    """filename-stem -> (text, emotion) from the per-speaker ESD text file."""
    out: dict[str, tuple[str, str]] = {}
    candidates = list(speaker_dir.glob("*.txt"))
    if not candidates:
        raise FileNotFoundError(f"no transcript file in {speaker_dir}")
    for line in candidates[0].read_text(encoding="utf-8-sig").splitlines():
        parts = [p for p in line.replace("\t", " ").split(" ") if p]
        if len(parts) < 3:
            continue
        stem, emotion = parts[0], parts[-1]
        text = " ".join(parts[1:-1])
        out[stem] = (text, emotion)
    return out


def _rows_from_esd(root: Path) -> list[dict[str, str]]:
    rows = []
    for speaker_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        speaker = speaker_dir.name
        transcripts = _read_esd_transcripts(speaker_dir)
        for emotion_dir in sorted(p for p in speaker_dir.iterdir() if p.is_dir()):
            for wav in sorted(emotion_dir.rglob("*.wav")):
                stem = wav.stem
                text = transcripts.get(stem, ("", ""))[0]
                index = stem.split("_")[-1]
                rows.append({"path": str(wav), "speaker": speaker,
                             "emotion": emotion_dir.name.lower(),
                             "text": text, "index": index})
    return rows


def _rows_from_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as fh:
        return [dict(r) for r in csv.DictReader(fh)]


def build_manifest(source: str | Path, emotion: str, out_dir: str | Path,
                   bases_per_speaker: int = 5) -> dict:
    """Pair neutral/emotional renditions of the same text per speaker, choose
    neutral bases, run leakage checks, write hashed manifest."""
    source = Path(source)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = _rows_from_csv(source) if source.is_file() else _rows_from_esd(source)
    emotion = emotion.lower()

    by_key: dict[tuple[str, str], dict[str, dict]] = {}
    for r in rows:
        if not r.get("text"):
            continue
        key = (r["speaker"], r["index"])
        by_key.setdefault(key, {})[r["emotion"].lower()] = r

    pairs: list[PairRecord] = []
    neutral_rows: list[dict] = []
    for (speaker, index), variants in sorted(by_key.items()):
        try:
            split = speaker_split(speaker)
        except ValueError:
            continue  # speakers outside the canonical partition are ignored
        if "neutral" in variants:
            neutral_rows.append({**variants["neutral"], "split": split})
        if "neutral" in variants and emotion in variants:
            neu, emo = variants["neutral"], variants[emotion]
            if neu["text"].strip().lower() != emo["text"].strip().lower():
                continue  # lexical match required
            pairs.append(PairRecord(
                pair_id=f"{speaker}:{index}", speaker=speaker, index=index,
                text=neu["text"], split=split, neutral_audio=neu["path"],
                emotional_audio=emo["path"], emotion=emotion))

    bases: list[BaseRecord] = []
    by_speaker: dict[str, list[dict]] = {}
    for r in sorted(neutral_rows, key=lambda x: (x["speaker"], x["index"])):
        by_speaker.setdefault(r["speaker"], []).append(r)
    for speaker, rows_s in by_speaker.items():
        if len(rows_s) < 2:
            continue
        for i, r in enumerate(rows_s[:bases_per_speaker]):
            ref = rows_s[(i + 1) % len(rows_s)]  # different neutral utterance
            bases.append(BaseRecord(
                base_id=f"{speaker}:{r['index']}", speaker=speaker,
                index=r["index"], target_text=r["text"],
                reference_index=ref["index"], reference_text=ref["text"],
                reference_audio=ref["path"], split=r["split"]))

    _leakage_checks(pairs, bases)

    pairs_path = out / "pairs.jsonl"
    bases_path = out / "bases.jsonl"
    _write_jsonl(pairs_path, [asdict(p) for p in pairs])
    _write_jsonl(bases_path, [asdict(b) for b in bases])
    manifest = {
        "schema": "ste-optimized/dataset/v1",
        "emotion": emotion,
        "speaker_partition": {k: list(v) for k, v in SPEAKER_PARTITION.items()},
        "counts": {split: sum(1 for p in pairs if p.split == split)
                   for split in SPEAKER_PARTITION},
        "base_counts": {split: sum(1 for b in bases if b.split == split)
                        for split in SPEAKER_PARTITION},
        "files": {"pairs": _sha256(pairs_path), "bases": _sha256(bases_path)},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def _leakage_checks(pairs: list[PairRecord], bases: list[BaseRecord]) -> None:
    all_splits = list(SPEAKER_PARTITION)
    for a in all_splits:
        for b in all_splits:
            if a < b and set(SPEAKER_PARTITION[a]) & set(SPEAKER_PARTITION[b]):
                raise ValueError(f"partition overlap between {a} and {b}")
    seen: dict[str, str] = {}
    for p in pairs:
        if speaker_split(p.speaker) != p.split:
            raise ValueError(f"pair {p.pair_id} speaker outside split {p.split}")
        if p.pair_id in seen and seen[p.pair_id] != p.split:
            raise ValueError(f"pair id {p.pair_id} appears in two splits")
        seen[p.pair_id] = p.split
    for b in bases:
        if speaker_split(b.speaker) != b.split:
            raise ValueError(f"base {b.base_id} speaker outside split {b.split}")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_pairs(dataset_dir: str | Path, split: str) -> list[PairRecord]:
    rows = read_jsonl(Path(dataset_dir) / "pairs.jsonl")
    return [PairRecord(**r) for r in rows if r["split"] == split]


def load_bases(dataset_dir: str | Path, split: str) -> list[BaseRecord]:
    rows = read_jsonl(Path(dataset_dir) / "bases.jsonl")
    return [BaseRecord(**r) for r in rows if r["split"] == split]
