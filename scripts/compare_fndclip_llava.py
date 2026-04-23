"""Head-to-head FND-CLIP vs LLaVA-1.5-7B on the OOC test split."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)


def metrics_from(df, prob_col="p_ooc", pred_col="y_pred", true_col="y_true"):
    y_true = df[true_col].values.astype(int)
    y_pred = df[pred_col].values.astype(int)
    y_score = df[prob_col].values.astype(float)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(df)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_true, y_score)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fnd",
                   default="outputs/v1_results/fndclip_ooc_predictions.csv")
    p.add_argument("--llava",
                   default="outputs/v1_results/llava_ooc_predictions.csv")
    p.add_argument("--out_yaml",
                   default="outputs/v1_results/v1_ooc_comparison.yaml")
    p.add_argument("--out_md",
                   default="outputs/v1_results/v1_ooc_comparison.md")
    args = p.parse_args()

    fnd = pd.read_csv(args.fnd).drop_duplicates("sample_id", keep="first")
    llv = pd.read_csv(args.llava).drop_duplicates("sample_id", keep="first")

    # Align on sample_id so both models are evaluated on the same rows
    common = sorted(set(fnd["sample_id"]) & set(llv["sample_id"]))
    print(f"FND-CLIP rows: {len(fnd)}")
    print(f"LLaVA rows:    {len(llv)}")
    print(f"Intersection:  {len(common)}")

    fnd = fnd[fnd["sample_id"].isin(common)].sort_values("sample_id").reset_index(drop=True)
    llv = llv[llv["sample_id"].isin(common)].sort_values("sample_id").reset_index(drop=True)

    assert (fnd["y_true"].values == llv["y_true"].values).all(), \
        "Ground-truth labels disagree between files"

    m_fnd = metrics_from(fnd)
    m_llv = metrics_from(llv)

    out = {
        "n_samples": len(common),
        "fnd_clip": m_fnd,
        "llava_zero_shot": m_llv,
        "deltas_fnd_minus_llava": {
            k: round(m_fnd[k] - m_llv[k], 4)
            for k in ["accuracy", "precision", "recall", "f1", "auc_roc"]
        },
    }

    print("\n=== FND-CLIP (trained) vs LLaVA-1.5-7B (zero-shot) ===")
    print(f"Evaluated on n = {len(common)} identical test samples")
    print()
    print(f"{'metric':<12} {'FND-CLIP':>10} {'LLaVA':>10} {'Δ':>10}")
    for k in ["accuracy", "precision", "recall", "f1", "auc_roc"]:
        print(f"{k:<12} {m_fnd[k]:>10.4f} {m_llv[k]:>10.4f} "
              f"{m_fnd[k] - m_llv[k]:>+10.4f}")

    Path(args.out_yaml).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_yaml, "w") as f:
        yaml.safe_dump(out, f, sort_keys=False)
    print(f"\nSaved: {args.out_yaml}")

    md_lines = [
        "## FND-CLIP (trained) vs LLaVA-1.5-7B (zero-shot) — V1 OOC",
        "",
        f"Both evaluated on the same **{len(common)} test samples**.",
        "",
        "| Metric | FND-CLIP (trained) | LLaVA-1.5-7B (zero-shot) | Δ (FND − LLaVA) |",
        "|--------|-------------------:|-------------------------:|----------------:|",
    ]
    for k in ["accuracy", "precision", "recall", "f1", "auc_roc"]:
        md_lines.append(
            f"| {k} | {m_fnd[k]:.4f} | {m_llv[k]:.4f} | {m_fnd[k] - m_llv[k]:+.4f} |"
        )
    md_lines += [
        "",
        "### Confusion matrices",
        "",
        "**FND-CLIP:**",
        "",
        f"| | Pred aligned | Pred OOC |",
        f"|---|---|---|",
        f"| True aligned (n={m_fnd['tn']+m_fnd['fp']}) | TN={m_fnd['tn']} | FP={m_fnd['fp']} |",
        f"| True OOC (n={m_fnd['fn']+m_fnd['tp']}) | FN={m_fnd['fn']} | TP={m_fnd['tp']} |",
        "",
        "**LLaVA-1.5-7B zero-shot:**",
        "",
        f"| | Pred aligned | Pred OOC |",
        f"|---|---|---|",
        f"| True aligned (n={m_llv['tn']+m_llv['fp']}) | TN={m_llv['tn']} | FP={m_llv['fp']} |",
        f"| True OOC (n={m_llv['fn']+m_llv['tp']}) | FN={m_llv['fn']} | TP={m_llv['tp']} |",
        "",
    ]
    with open(args.out_md, "w") as f:
        f.write("\n".join(md_lines))
    print(f"Saved: {args.out_md}")


if __name__ == "__main__":
    main()
