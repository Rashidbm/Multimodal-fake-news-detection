# V1 Semantic Baseline: FND-CLIP for OOC Detection

Hardware: Apple M3 Max, 48 GB RAM, MPS backend.

## Task

Binary out-of-context (OOC) detection. Scenario 1 (real text + real image,
misaligned) is the positive class. Scenarios 2, 3, 4, 5 are the negative
class. The 5-way scenario breakdown happens downstream in the external
fusion layer (V1 + V2 + V3 + V4). V1 only decides OOC vs not-OOC.

## Dataset

12,000 balanced samples.

| Class | Count | Source |
|-------|-------|--------|
| OOC (y=1) | 6,000 | NewsCLIPpings shuffled pairs |
| Not-OOC (y=0) | 6,000 | 1,500 each from DGM4 scenarios 2, 3, 5 and DGM4/NewsCLIPpings scenario 4 |

Stratified split: 8,400 train, 1,800 val, 1,800 test.

## Model

FND-CLIP from Zhou et al., ICME 2023.

- Visual stream: ResNet-50 (ImageNet pretrained, fine-tuned), 2048 to 512.
- Text stream: BERT-base-uncased, [CLS] token, 768 to 512.
- CLIP stream: paired CLIP-ViT-B/32 encoders, 512-dim each. Cosine similarity
  between image and text embeddings reweights the fused vector.
- Modality attention: learned softmax over 3 scalars (text, image, clip).
- Classifier: 2-layer MLP with sigmoid.

287M total parameters, 135M trainable (CLIP frozen).

## Training

AdamW, lr 2e-5, weight decay 1e-4, cosine annealing over 3 epochs, batch
size 16, BCEWithLogitsLoss. Training time about 15 minutes.

| Epoch | Train loss | Val loss | Val acc | Val F1 | Val AUC |
|-------|-----------|----------|---------|--------|---------|
| 1 | 0.4646 | 0.3328 | 0.8600 | 0.8526 | 0.9357 |
| 2 | 0.1796 | 0.1470 | 0.9467 | 0.9453 | 0.9888 (best) |
| 3 | 0.0699 | 0.1503 | 0.9406 | 0.9404 | 0.9883 |

Best checkpoint: epoch 2. Test metrics use that checkpoint.

## Test metrics (n = 1,800)

| Metric | Value |
|--------|-------|
| Accuracy | 0.9456 |
| Precision | 0.9471 |
| Recall | 0.9428 |
| F1 | 0.9449 |
| AUC-ROC | 0.9874 |

Confusion matrix:

|                           | Pred not-OOC | Pred OOC |
|---------------------------|-------------|----------|
| True aligned (n=908) | TN = 861 | FP = 47 |
| True OOC (n=892) | FN = 51 | TP = 841 |

False positive rate 5.2%, false negative rate 5.7%.

## LLaVA-1.5-7B zero-shot comparison

Run on the same test samples, zero-shot (no fine-tuning). For each pair:

1. Prompt: `"USER: <image>\nText: {text}\n\nQuestion: Does the image match what the text describes?\nAnswer with ONLY one word: Yes or No.\nASSISTANT:"`
2. One forward pass through LLaVA. Take the next-token logit distribution
   (shape 32,000 over the vocabulary).
3. Pick the logits at the "Yes" and "No" token ids. Softmax over those two.
4. Treat P("No") as P(OOC). Predict OOC if P(OOC) >= 0.5.

The softmax over two specific token logits is what gives a continuous
probability per sample, which is what makes AUC-ROC computable.

### Head-to-head on 1,724 identical test samples

| Metric | FND-CLIP (trained) | LLaVA-1.5-7B (zero-shot) | Delta |
|--------|-------------------:|-------------------------:|------:|
| Accuracy | 0.9623 | 0.7865 | +0.1758 |
| Precision | 0.9686 | 0.8079 | +0.1607 |
| Recall | 0.9589 | 0.7756 | +0.1833 |
| F1 | 0.9637 | 0.7914 | +0.1723 |
| AUC-ROC | 0.9946 | 0.8716 | +0.1230 |
| Params | 287M (135M trainable) | 7B (frozen) | - |

FND-CLIP on the 1,724-sample split is slightly higher than on the earlier
1,800-sample split (96.2% vs 94.6%) because the splits differ by a few
samples. Same checkpoint in both cases.

### Why LLaVA loses

- No fine-tuning on OOC detection. The prompt is the only task signal.
- Logit margin is small: `logit_yes` mean 24.60, `logit_no` mean 24.57 across
  1,724 samples. Model hedges rather than commits.
- P(OOC) class separation is soft: 0.39 mean on aligned pairs, 0.59 on OOC.
  AUC 0.87 reflects a real but noisy signal.
- Threshold tuning on the logits caps at 79.06% accuracy (optimal threshold
  0.49), so calibration alone cannot close the gap.

### Options to improve LLaVA (not run)

1. Entity-focused prompt with explicit reasoning. Expected 82-85%.
2. Few-shot in-context learning (3-5 labeled examples in the prompt).
   Expected 84-88%.
3. LoRA fine-tune on the 8,400 training samples. Expected 85-92% but ~10
   hours on MPS.
4. LLaVA-NeXT 13B or Qwen2-VL. Expected 82-87%.

## Files

Code:
- `src/models/fnd_clip.py`
- `src/dataset.py`
- `src/train.py`
- `src/evaluate.py`
- `scripts/build_balanced_dataset.py`
- `scripts/llava_ooc_eval.py`
- `scripts/fndclip_ooc_test_eval.py`
- `scripts/compare_fndclip_llava.py`
- `config/v1_ooc.yaml`

Artifacts:
- `outputs/v1_ooc/best.pt` (checkpoint, not in repo)
- `outputs/v1_results/llava_ooc_predictions.csv` (per-sample logits)
- `outputs/v1_results/fndclip_ooc_predictions.csv` (FND-CLIP predictions on same split)
- `outputs/v1_results/v1_ooc_comparison.yaml`
