"""LLaVA-1.5-7B zero-shot 5-class variant (not part of V1 binary; kept for reference)."""

import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from tqdm import tqdm


PROMPT = """USER: <image>
Text: {text}

You are looking at a news article with the above text and image. Classify which of the 5 scenarios this belongs to:
1. Real text with real image but MISMATCHED (out-of-context)
2. Fake/manipulated text with real image
3. Real text with fake/manipulated image
4. Real text with real image (GENUINE)
5. Fake text with fake image (both manipulated)

Answer with ONLY the digit 1, 2, 3, 4, or 5.
ASSISTANT:"""


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float16
    if torch.backends.mps.is_available():
        return torch.device("mps"), torch.float16
    return torch.device("cpu"), torch.float32


def parse_answer(text):
    """Extract digit 1-5 from LLaVA's response."""
    m = re.search(r"[1-5]", text)
    return int(m.group()) if m else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/processed/balanced_dataset.csv")
    p.add_argument("--split", default="test")
    p.add_argument("--limit", type=int, default=500)
    p.add_argument("--model", default="llava-hf/llava-1.5-7b-hf")
    p.add_argument("--output", default="outputs/v1_multiclass/llava_predictions.csv")
    args = p.parse_args()

    from transformers import LlavaForConditionalGeneration, AutoProcessor

    device, dtype = pick_device()
    print(f"Device: {device} | dtype: {dtype}")

    df = pd.read_csv(args.csv)
    df = df[df["split"] == args.split].reset_index(drop=True)
    if args.limit:
        df = df.iloc[:args.limit]
    print(f"Evaluating {len(df)} {args.split} samples")

    print(f"Loading {args.model}...")
    processor = AutoProcessor.from_pretrained(args.model)
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=dtype, low_cpu_mem_usage=True,
    ).to(device).eval()

    # Token ids for digits 1-5 (use LAST token — LLaVA tokenizer prefixes a space token)
    digit_ids = {}
    for d in range(1, 6):
        ids = processor.tokenizer(str(d), add_special_tokens=False).input_ids
        digit_ids[d] = ids[-1]  # last token = actual digit
    print(f"Digit token ids: {digit_ids}")
    assert len(set(digit_ids.values())) == 5, f"Digit tokens not unique! {digit_ids}"

    rows = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="LLaVA"):
        try:
            pil = Image.open(row["image_path"]).convert("RGB")
        except Exception as e:
            continue

        prompt = PROMPT.format(text=str(row["text"])[:400])
        inputs = processor(images=pil, text=prompt, return_tensors="pt").to(device, dtype)

        with torch.no_grad():
            # Get logits for next token (the digit)
            out = model(**inputs)
        next_logits = out.logits[0, -1, :]  # (vocab,)

        # Logits for 1-5 token ids
        class_logits = torch.tensor([next_logits[digit_ids[d]].item() for d in range(1, 6)])
        class_probs = torch.softmax(class_logits, dim=0).numpy()
        pred_scenario = int(class_logits.argmax().item()) + 1  # 1-5

        rows.append({
            "sample_id": row["sample_id"],
            "text": str(row["text"])[:120],
            "true_scenario": int(row["scenario"]),
            "pred_scenario": pred_scenario,
            "logit_s1": float(class_logits[0]), "prob_s1": float(class_probs[0]),
            "logit_s2": float(class_logits[1]), "prob_s2": float(class_probs[1]),
            "logit_s3": float(class_logits[2]), "prob_s3": float(class_probs[2]),
            "logit_s4": float(class_logits[3]), "prob_s4": float(class_probs[3]),
            "logit_s5": float(class_logits[4]), "prob_s5": float(class_probs[4]),
        })

        # Periodic save
        if (idx + 1) % 50 == 0:
            pd.DataFrame(rows).to_csv(args.output, index=False)

    df_out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df_out.to_csv(args.output, index=False)

    # Compute metrics
    from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                                 recall_score, roc_auc_score, confusion_matrix)
    true = df_out["true_scenario"].values - 1  # 0-4
    pred = df_out["pred_scenario"].values - 1
    probs = df_out[[f"prob_s{i}" for i in range(1, 6)]].values

    metrics = {
        "n_samples": len(df_out),
        "accuracy": accuracy_score(true, pred),
        "precision_macro": precision_score(true, pred, average="macro", zero_division=0),
        "recall_macro": recall_score(true, pred, average="macro", zero_division=0),
        "f1_macro": f1_score(true, pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(true, pred, average="weighted", zero_division=0),
    }
    try:
        ohe = np.eye(5)[true]
        metrics["auc_roc_macro"] = roc_auc_score(ohe, probs, average="macro", multi_class="ovr")
    except Exception:
        metrics["auc_roc_macro"] = float("nan")

    cm = confusion_matrix(true, pred, labels=[0, 1, 2, 3, 4])
    metrics["confusion_matrix"] = cm.tolist()

    print("\n=== LLaVA 5-class Metrics ===")
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    out_yaml = args.output.replace(".csv", "_metrics.yaml")
    with open(out_yaml, "w") as f:
        yaml.safe_dump({k: (float(v) if isinstance(v, (int, float)) else v)
                        for k, v in metrics.items()}, f)
    print(f"\nSaved: {args.output} + {out_yaml}")


if __name__ == "__main__":
    main()
