# Multimodal fake-news detection — V1

V1 of my graduation project. Binary out-of-context detector: given
(text, image), predict whether the image matches the text.

Model is FND-CLIP (Zhou et al., ICME 2023) — ResNet-50 + BERT + frozen
CLIP, fused with modality attention. Code under `src/`, configs under
`config/`, runnable scripts under `scripts/`.

## Run

```
pip install -r requirements-v1.txt
python scripts/build_balanced_dataset.py
python -m src.train --config config/v1_ooc.yaml
python scripts/fndclip_ooc_test_eval.py
```

Data (DGM4, NewsCLIPpings) and checkpoints are not in the repo.

## Results

Test set n=1,800. Accuracy 0.9456, F1 0.9449, AUC 0.9874.

- `outputs/v1_results/V1_OOC_FINAL.md` — full write-up (training curves,
  confusion matrix, LLaVA-1.5-7B zero-shot comparison).
- `outputs/v1_results/fndclip_ooc_predictions.csv` — per-sample predictions.
- `outputs/v1_results/fndclip_ooc_predictions_metrics.yaml` — metrics dump.
- `outputs/v1_results/v1_ooc_comparison.md` — head-to-head vs LLaVA.
- `outputs/v1_results/diagrams/` — figures.

To regenerate the metrics YAMLs:

```
python scripts/fndclip_ooc_test_eval.py       # FND-CLIP on the test split
python scripts/llava_ooc_eval.py --split test # LLaVA zero-shot baseline
python scripts/compare_fndclip_llava.py       # head-to-head report
```
