#!/usr/bin/env python3
"""Evaluate a Whisper checkpoint on a local, explicitly selected manifest.

Manifest CSV columns: audio_path,reference
Paths are resolved relative to the manifest file.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from pathlib import Path
from typing import Iterable


_ARABIC_TO_PERSIAN = str.maketrans({"ي": "ی", "ى": "ی", "ك": "ک"})
_DIACRITICS = re.compile(r"[ً-ٰٟـ]")
_WHITESPACE = re.compile(r"\s+")


def normalize_persian(text: str) -> str:
    """Apply a small, explicit normalization before word error rate scoring."""
    text = unicodedata.normalize("NFC", text).translate(_ARABIC_TO_PERSIAN)
    text = text.replace("\u200c", " ")
    text = _DIACRITICS.sub("", text)
    text = "".join(" " if unicodedata.category(char).startswith("P") else char for char in text)
    return _WHITESPACE.sub(" ", text).strip()


def read_manifest(path: Path, audio_column: str, text_column: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {audio_column, text_column}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest is missing columns: {', '.join(sorted(missing))}")
        for line_number, row in enumerate(reader, start=2):
            audio_path = (row.get(audio_column) or "").strip()
            reference = (row.get(text_column) or "").strip()
            if not audio_path or not reference:
                raise ValueError(f"Empty audio path or reference at CSV line {line_number}")
            resolved = Path(audio_path)
            if not resolved.is_absolute():
                resolved = path.parent / resolved
            if not resolved.is_file():
                raise FileNotFoundError(f"Audio file not found at CSV line {line_number}: {resolved}")
            rows.append({
                "sample_id": (row.get("sample_id") or str(line_number - 1)).strip(),
                "audio_path": str(resolved),
                "reference": reference,
            })
    if not rows:
        raise ValueError("Manifest contains no evaluation rows")
    return rows


def batched(items: list[dict[str, str]], size: int) -> Iterable[list[dict[str, str]]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Hugging Face model ID or local checkpoint path")
    parser.add_argument("--manifest", required=True, type=Path, help="CSV manifest containing audio_path and reference")
    parser.add_argument("--output", type=Path, default=Path("whisper-evaluation.json"))
    parser.add_argument("--audio-column", default="audio_path")
    parser.add_argument("--text-column", default="reference")
    parser.add_argument("--device", default="cpu", help="Transformers pipeline device, e.g. cpu or cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--chunk-length-s", type=int, default=30)
    parser.add_argument("--stride-length-s", type=int, default=5)
    args = parser.parse_args()
    if args.batch_size < 1 or args.chunk_length_s < 1 or args.stride_length_s < 0:
        parser.error("batch size and chunk length must be positive; stride cannot be negative")

    try:
        from jiwer import wer
        from transformers import pipeline
    except ImportError as exc:
        raise SystemExit("Install evaluation dependencies with: pip install -r requirements-eval.txt") from exc

    manifest = args.manifest.resolve()
    rows = read_manifest(manifest, args.audio_column, args.text_column)
    asr = pipeline("automatic-speech-recognition", model=args.model, device=args.device)

    predictions: list[dict[str, str]] = []
    for batch in batched(rows, args.batch_size):
        outputs = asr(
            [row["audio_path"] for row in batch],
            batch_size=args.batch_size,
            chunk_length_s=args.chunk_length_s,
            stride_length_s=args.stride_length_s,
        )
        if isinstance(outputs, dict):
            outputs = [outputs]
        for row, output in zip(batch, outputs, strict=True):
            predictions.append({
                "sample_id": row["sample_id"],
                "audio_path": row["audio_path"],
                "reference": row["reference"],
                "prediction": output["text"],
                "normalized_reference": normalize_persian(row["reference"]),
                "normalized_prediction": normalize_persian(output["text"]),
            })

    score = wer(
        [item["normalized_reference"] for item in predictions],
        [item["normalized_prediction"] for item in predictions],
    )
    result = {
        "model": args.model,
        "manifest": str(manifest),
        "sample_count": len(predictions),
        "wer": score,
        "wer_percent": round(score * 100, 4),
        "normalization": "NFC; Arabic yeh/kaf mapping; ZWNJ and diacritics/punctuation removed; whitespace collapsed",
        "chunk_length_s": args.chunk_length_s,
        "stride_length_s": args.stride_length_s,
        "predictions": predictions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Samples: {len(predictions)}")
    print(f"WER: {score * 100:.4f}%")
    print(f"Results: {args.output.resolve()}")


if __name__ == "__main__":
    main()
