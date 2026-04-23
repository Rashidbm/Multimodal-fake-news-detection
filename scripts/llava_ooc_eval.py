"""LLaVA-1.5-7B zero-shot OOC baseline via next-token Yes/No logits."""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from tqdm import tqdm


PROMPT = """USER: <image>
Text: {text}

Question: Does the image match what the text describes?
Answer with ONLY one word: Yes or No.
ASSISTANT:"""


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float16
    if torch.backends.mps.is_available():
        return torch.device("mps"), torch.float16
    return torch.device("cpu"), torch.float32


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/processed/balanced_dataset.csv")
    p.add_argument("--split", default="test")
    p.add_argument("--limit", type=int, default=0,
                   help="0 = full split")
    p.add_argument("--model", default="llava-hf/llava-1.5-7b-hf")
    p.add_argument("--output",
                   default="outputs/v1_results/llava_ooc_predictions.csv")
    args = p.parse_args()

    from transformers import LlavaForConditionalGeneration, AutoProcessor

    device, dtype = pick_device()
    print(f"Device: {device} | dtype: {dtype}")

    df = pd.read_csv(args.csv)
    df = df[df["split"] == args.split].reset_index(drop=True)
    # Shuffle so classes are mixed across periodic saves
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    if args.limit and args.limit > 0:
        df = df.iloc[:args.limit].reset_index(drop=True)

    # y_true: scenario 1 (OOC) -> 1, scenario 4 (aligned) -> 0
    df["y_true"] = (df["scenario"] == 1).astype(int)
    print(f"Evaluating {len(df)} samples ({args.split} split)")
    print(f"  OOC (y=1):     {(df['y_true']==1).sum()}")
    print(f"  aligned (y=0): {(df['y_true']==0).sum()}")

    print(f"\nLoading {args.model}...")
    processor = AutoProcessor.from_pretrained(args.model)
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=dtype, low_cpu_mem_usage=True,
    ).to(device).eval()

    # Llama tokenizer prefixes a leading-space token; take the last id.
    yes_ids = processor.tokenizer("Yes", add_special_tokens=False).input_ids
    no_ids = processor.tokenizer("No", add_special_tokens=False).input_ids
    yes_id, no_id = yes_ids[-1], no_ids[-1]
    print(f"Yes token id: {yes_id}, No token id: {no_id}")
    assert yes_id != no_id, "Yes/No token ids must differ"

    rows = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="LLaVA-OOC"):
        try:
            pil = Image.open(row["image_path"]).convert("RGB")
        except Exception:
            continue

        prompt = PROMPT.format(text=str(row["text"])[:400])
        inputs = processor(images=pil, text=prompt,
                           return_tensors="pt").to(device, dtype)

        with torch.no_grad():
            out = model(**inputs)
        next_logits = out.logits[0, -1, :].float().cpu()

        logit_yes = float(next_logits[yes_id].item())
        logit_no = float(next_logits[no_id].item())

        two = torch.tensor([logit_no, logit_yes])
        probs = torch.softmax(two, dim=0).numpy()
        p_ooc = float(probs[0])
        p_aligned = float(probs[1])

        pred = 1 if p_ooc > 0.5 else 0

        rows.append({
            "sample_id": row["sample_id"],
            "scenario": int(row["scenario"]),
            "y_true": int(row["y_true"]),
            "logit_yes": logit_yes,
            "logit_no": logit_no,
            "p_ooc": p_ooc,
            "p_aligned": p_aligned,
            "y_pred": pred,
        })

        if (idx + 1) % 50 == 0:
            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
            pd.DataFrame(rows).to_csv(args.output, index=False)

    df_out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df_out.to_csv(args.output, index=False)

    from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                                 recall_score, roc_auc_score,
                                 confusion_matrix)
    y_true = df_out["y_true"].values
    y_pred = df_out["y_pred"].values
    y_score = df_out["p_ooc"].values

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics = {
        "n_samples": int(len(df_out)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_true, y_score)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "n_predicted_ooc": int((y_pred == 1).sum()),
        "n_predicted_aligned": int((y_pred == 0).sum()),
    }

    print("\nLLaVA-1.5-7B zero-shot OOC metrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    out_yaml = args.output.replace(".csv", "_metrics.yaml")
    with open(out_yaml, "w") as f:
        yaml.safe_dump(metrics, f)
    print(f"\nSaved: {args.output}")
    print(f"Saved: {out_yaml}")


if __name__ == "__main__":
    main()
