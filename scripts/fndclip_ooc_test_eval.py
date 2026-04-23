"""Run the trained FND-CLIP checkpoint on the CSV test split, save predictions + metrics."""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dataset import MultiGuardDataset  # noqa: E402
from evaluate import compute_metrics    # noqa: E402
from models.fnd_clip import FNDCLIP     # noqa: E402


def collate_batch(batch):
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch]) for k in keys}


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/processed/balanced_dataset.csv")
    p.add_argument("--checkpoint", default="outputs/v1_ooc/best.pt")
    p.add_argument("--bert", default="bert-base-uncased")
    p.add_argument("--clip", default="openai/clip-vit-base-patch32")
    p.add_argument("--output",
                   default="outputs/v1_results/fndclip_ooc_predictions.csv")
    p.add_argument("--batch_size", type=int, default=16)
    args = p.parse_args()

    device = pick_device()
    print(f"Device: {device}")

    df = pd.read_csv(args.csv)
    test_idx = df.index[df["split"] == "test"].tolist()
    print(f"Test split: {len(test_idx)} samples")

    ds = MultiGuardDataset(
        csv_path=args.csv, train=False,
        bert_model=args.bert, clip_model=args.clip, max_length=128,
    )
    test_subset = Subset(ds, test_idx)

    loader = DataLoader(
        test_subset, batch_size=args.batch_size, shuffle=False,
        num_workers=2, collate_fn=collate_batch,
    )

    model = FNDCLIP(feat_dim=512, num_classes=1).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded checkpoint from {args.checkpoint} "
          f"(best val AUC {ckpt.get('val_metrics', {}).get('auc_roc', 'n/a')})")

    all_probs, all_labels, all_scenarios = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="FND-CLIP eval"):
            out = model(
                image=batch["image"].to(device),
                bert_ids=batch["bert_ids"].to(device),
                bert_mask=batch["bert_mask"].to(device),
                clip_pixels=batch["clip_pixels"].to(device),
                clip_ids=batch["clip_ids"].to(device),
                clip_mask=batch["clip_mask"].to(device),
            )
            logits = out["logits"].squeeze(-1)
            all_probs.append(torch.sigmoid(logits).float().cpu())
            all_labels.append(batch["label"].cpu())
            all_scenarios.append(batch["scenario"].cpu())

    probs = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy().astype(int)
    scenarios = torch.cat(all_scenarios).numpy().astype(int)
    preds = (probs >= 0.5).astype(int)

    sample_ids = df.loc[test_idx, "sample_id"].values
    pd.DataFrame({
        "sample_id": sample_ids,
        "scenario": scenarios,
        "y_true": labels,
        "p_ooc": probs,
        "y_pred": preds,
    }).to_csv(args.output, index=False)

    metrics = compute_metrics(labels, probs)
    print("\n=== FND-CLIP on current test split ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    out_yaml = args.output.replace(".csv", "_metrics.yaml")
    with open(out_yaml, "w") as f:
        yaml.safe_dump({k: float(v) if isinstance(v, (int, float))
                        else v for k, v in metrics.items()}, f)
    print(f"\nSaved: {args.output}")
    print(f"Saved: {out_yaml}")


if __name__ == "__main__":
    main()
