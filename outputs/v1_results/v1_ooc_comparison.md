## FND-CLIP (trained) vs LLaVA-1.5-7B (zero-shot) — V1 OOC

Both evaluated on the same **1724 test samples**.

| Metric | FND-CLIP (trained) | LLaVA-1.5-7B (zero-shot) | Δ (FND − LLaVA) |
|--------|-------------------:|-------------------------:|----------------:|
| accuracy | 0.9623 | 0.7865 | +0.1758 |
| precision | 0.9686 | 0.8079 | +0.1607 |
| recall | 0.9589 | 0.7756 | +0.1833 |
| f1 | 0.9637 | 0.7914 | +0.1723 |
| auc_roc | 0.9946 | 0.8716 | +0.1230 |

### Confusion matrices

**FND-CLIP:**

| | Pred aligned | Pred OOC |
|---|---|---|
| True aligned (n=824) | TN=796 | FP=28 |
| True OOC (n=900) | FN=37 | TP=863 |

**LLaVA-1.5-7B zero-shot:**

| | Pred aligned | Pred OOC |
|---|---|---|
| True aligned (n=824) | TN=658 | FP=166 |
| True OOC (n=900) | FN=202 | TP=698 |
