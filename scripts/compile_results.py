"""Compile the final V1 results report from the pipeline outputs."""

import argparse
import os
from pathlib import Path

import pandas as pd
import yaml


def load_yaml(path):
    if not path or not os.path.exists(path):
        return None
    with open(path) as f:
        return yaml.safe_load(f)


def format_metrics(m):
    if not m:
        return "_(not available)_"
    keys = ["accuracy", "precision", "recall", "f1", "auc_roc"]
    rows = []
    for k in keys:
        v = m.get(k)
        if v is not None:
            rows.append(f"| {k.upper():10s} | {v:.4f} |")
    return "| Metric | Value |\n|--------|-------|\n" + "\n".join(rows)


def compute_llava_metrics(csv_path):
    if not csv_path or not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    if "gt_answers" not in df.columns:
        return None
    df["gt"] = df["gt_answers"].str.lower().isin(["fake", "false"]).astype(int)
    df["pred"] = (df["llava_prediction"].str.lower() == "yes").astype(int)
    from sklearn.metrics import (accuracy_score, f1_score,
                                 precision_score, recall_score, roc_auc_score)
    return {
        "accuracy": accuracy_score(df["gt"], df["pred"]),
        "precision": precision_score(df["gt"], df["pred"], zero_division=0),
        "recall": recall_score(df["gt"], df["pred"], zero_division=0),
        "f1": f1_score(df["gt"], df["pred"], zero_division=0),
        "auc_roc": roc_auc_score(df["gt"], df["p_yes"]) if df["gt"].nunique() > 1 else float("nan"),
        "n_samples": len(df),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fndclip-metrics", help="FND-CLIP test metrics YAML")
    p.add_argument("--fndclip-transfer", help="FND-CLIP MMFakeBench transfer YAML")
    p.add_argument("--llava-csv", help="LLaVA logits CSV")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    fnd_test = load_yaml(args.fndclip_metrics)
    fnd_transfer = load_yaml(args.fndclip_transfer)
    llava_metrics = compute_llava_metrics(args.llava_csv)

    md = []
    md.append("# MultiGuard V1 — Results Report\n")
    md.append("**Pipeline:** FND-CLIP (Zhou et al., ICME 2023) — Semantic Baseline\n")
    md.append("## 1. FND-CLIP on DGM4 Test Set\n")
    md.append("Training data: balanced 5-scenario subset of DGM4.\n")
    md.append(format_metrics(fnd_test))
    md.append("")
    md.append("## 2. FND-CLIP Transferred to MMFakeBench (zero-shot benchmark)\n")
    md.append(format_metrics(fnd_transfer))
    md.append("")
    md.append("## 3. LLaVA-1.5-7B Zero-Shot on MMFakeBench\n")
    if llava_metrics:
        n = llava_metrics.pop("n_samples", "?")
        md.append(f"Samples evaluated: {n}\n")
        md.append(format_metrics(llava_metrics))
    else:
        md.append("_(not available)_")
    md.append("")
    md.append("## Summary\n")
    md.append("| Model | Evaluated On | Accuracy | F1 | AUC-ROC |")
    md.append("|-------|--------------|----------|------|---------|")
    if fnd_test:
        md.append(f"| FND-CLIP (V1) | DGM4 test | {fnd_test.get('accuracy', 0):.3f} | "
                  f"{fnd_test.get('f1', 0):.3f} | {fnd_test.get('auc_roc', 0):.3f} |")
    if fnd_transfer:
        md.append(f"| FND-CLIP (V1) | MMFakeBench val | {fnd_transfer.get('accuracy', 0):.3f} | "
                  f"{fnd_transfer.get('f1', 0):.3f} | {fnd_transfer.get('auc_roc', 0):.3f} |")
    if llava_metrics:
        md.append(f"| LLaVA-1.5-7B (zero-shot) | MMFakeBench val | {llava_metrics.get('accuracy', 0):.3f} | "
                  f"{llava_metrics.get('f1', 0):.3f} | {llava_metrics.get('auc_roc', 0):.3f} |")

    out = "\n".join(md)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        f.write(out)
    print(out)
    print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
