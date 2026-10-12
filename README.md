# Persian Whisper Fine-Tuning

Fine-tuning and evaluation notes for Persian automatic speech recognition with Whisper Large v3.

[View the published fine-tuned model on Hugging Face](https://huggingface.co/MohammadReza-Halakoo/1-persian-whisper-large-v3-train-validation-tested-100-percent-17-0-7-epoch)

## Reported experiment checkpoints

These are historical project results, not verified held-out benchmark scores. The model card metadata for the 7-epoch result lists the evaluated split as `train+validation+validated+test`, which includes training data. Therefore, its WER should not be presented as a generalization score on an untouched test set.

| Run | Reported result | Available detail |
| --- | --- | --- |
| Final fine-tuning run (7 epochs) | WER 34.60% → 26.85% | Model card metadata lists `train+validation+validated+test`; the exact manifest and scoring normalization are not included. |
| Learning-rate sweep (3 epochs) | WER 14.86% at 1e-7 and 1e-8 | A separate run in the project notes. The exact checkpoint, split, manifest, and scoring normalization are not documented. |

These figures come from separate runs and should not be compared as if they used the same checkpoint or test set. To publish a verified headline score, evaluate a fixed, untouched test manifest that was excluded from training and model selection. Record the model revision, manifest version, normalization, sample count, command, and JSON output. Do not redistribute audio or transcripts unless their licenses permit it.

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

Run the evaluator against the published model or a local checkpoint:

```bash
python evaluate_whisper.py \
  --model "MohammadReza-Halakoo/1-persian-whisper-large-v3-train-validation-tested-100-percent-17-0-7-epoch" \
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
- [Hugging Face model](https://huggingface.co/MohammadReza-Halakoo/1-persian-whisper-large-v3-train-validation-tested-100-percent-17-0-7-epoch)
- [LinkedIn profile](https://www.linkedin.com/in/mohammadreza-halakoo/)
