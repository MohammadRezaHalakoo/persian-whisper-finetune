# Persian Whisper Fine-Tuning

Fine-tuning and evaluation notes for Persian automatic speech recognition with Whisper Large v3.

## Reported experiment checkpoints

The following figures are historical results reported for this project. The original repository did not include the exact held-out evaluation manifest, so these values are not independently reproducible from the repository alone.

| Run | Reported result | How to interpret it |
| --- | --- | --- |
| Final fine-tuning run (7 epochs) | WER 34.60% → 26.85% | Reported before/after result; held-out split and scoring normalization need to be confirmed. |
| Learning-rate sweep (3 epochs) | WER 14.86% at a reported learning rate of 1e-7/1e-8 | A separate run, not the final checkpoint result above. The original notes do not make the exact LR/checkpoint association or evaluation split sufficiently clear. |

Do not compare these numbers as if they came from the same checkpoint or test set. To publish a verified headline score, evaluate a fixed, untouched test manifest and commit the manifest schema, command, normalization, sample count, and JSON output (without redistributing audio or transcripts unless their licenses permit it).

### Previously reported setup

- Dataset: Persian Common Voice 17.0; project notes report 368,172 samples.
- The project notes describe about 40% of the data as noisy; the noise-labeling method was not recorded in the repository.
- Reported training setup: 2 × H100 PCIe 80 GB, about 10.6 days, learning rate 1e-7, batch size 20, mixed precision fp16.
- These are project-reported settings, not independently verified benchmark claims.

## Reproducible local evaluation

The repository includes a local evaluator that reads a CSV manifest and writes per-sample predictions plus aggregate WER. It does not download or expose dataset audio.

Install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-eval.txt
```

Create a local `manifest.csv` (keep audio and transcripts private unless you have permission to publish them):

```csv
sample_id,audio_path,reference
sample-001,audio/001.mp3,متن مرجع فارسی
```

Run the evaluator against the actual model ID or a local checkpoint:

```bash
python evaluate_whisper.py \
  --model "/path/to/model-or-hugging-face-id" \
  --manifest manifest.csv \
  --device cuda:0 \
  --batch-size 4 \
  --output results/whisper-evaluation.json
```

Use the same fixed manifest, model revision, chunking settings, and scoring rules for comparisons. The evaluator normalizes Unicode to NFC, maps Arabic yeh/kaf to Persian forms, replaces ZWNJ with a space, removes Persian/Arabic diacritics and punctuation, and collapses whitespace before WER scoring.

Run the lightweight normalization tests with:

```bash
python -m unittest discover -s tests -v
```

The evaluator intentionally writes predictions locally. Review the output before publishing it: it contains recognized text and local audio paths.

## Project links

- [GitHub profile](https://github.com/MohammadRezaHalakoo)
- [Hugging Face profile](https://huggingface.co/MohammadReza-Halakoo)
- [LinkedIn profile](https://www.linkedin.com/in/mohammadreza-halakoo/)
