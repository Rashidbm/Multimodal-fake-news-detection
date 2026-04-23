"""Transfer eval on MMFakeBench val/test using a trained FND-CLIP checkpoint."""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from transformers import BertTokenizer, CLIPProcessor

sys.path.insert(0, str(Path(__file__).parent))
from models.fnd_clip import FNDCLIP
from evaluate import compute_metrics


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class MMFakeBenchEvalDataset(Dataset):
    def __init__(self, root, split="val",
                 bert_model="bert-base-uncased",
                 clip_model="openai/clip-vit-base-patch32"):
        from datasets import load_from_disk
        root = Path(root)
        hf_path = root / "liuxuannan___mm_fake_bench" / f"MMFakeBench_{split}"
        # Walk to the latest version/hash dir
        for v in hf_path.iterdir():
            if v.is_dir():
                for h in v.iterdir():
                    if h.is_dir() and (h / "dataset_info.json").exists():
                        hf_path = h
                        break
                break
        ds = load_from_disk(str(hf_path))
        if "train" in ds:
            ds = ds["train"]
        self.samples = list(ds)
        self.images_root = root / "images" / f"MMFakeBench_{split}"
        self.bert_tok = BertTokenizer.from_pretrained(bert_model)
        self.clip_proc = CLIPProcessor.from_pretrained(clip_model)
        self.image_tf = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        text = item["text"]
        rel = item["image_path"].lstrip("/")
        img_path = self.images_root / rel
        try:
            pil = Image.open(img_path).convert("RGB")
        except Exception:
            pil = Image.new("RGB", (224, 224), 0)

        image = self.image_tf(pil)
        bert = self.bert_tok(text, padding="max_length", truncation=True,
                             max_length=128, return_tensors="pt")
        clip = self.clip_proc(images=pil, text=text, return_tensors="pt",
                              padding="max_length", truncation=True, max_length=77)

        gt = item.get("gt_answers", "True")
        label = 0 if gt.lower() in ("true", "real") else 1

        return {
            "image": image,
            "bert_ids": bert["input_ids"].squeeze(0),
            "bert_mask": bert["attention_mask"].squeeze(0),
            "clip_pixels": clip["pixel_values"].squeeze(0),
            "clip_ids": clip["input_ids"].squeeze(0),
            "clip_mask": clip["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.float32),
            "fake_cls": item.get("fake_cls", ""),
        }


def collate(batch):
    keys = [k for k in batch[0].keys() if k != "fake_cls"]
    out = {k: torch.stack([b[k] for b in batch]) for k in keys}
    out["fake_cls"] = [b["fake_cls"] for b in batch]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--mmfakebench", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--output", required=True)
    p.add_argument("--batch-size", type=int, default=8)
    args = p.parse_args()

    device = pick_device()
    print(f"Device: {device}")

    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt.get("config", {})
    feat_dim = cfg.get("model", {}).get("feat_dim", 512)
    model = FNDCLIP(feat_dim=feat_dim, num_classes=1).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    ds = MMFakeBenchEvalDataset(args.mmfakebench, split=args.split)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate, num_workers=2)
    print(f"Eval samples: {len(ds)}")

    all_probs, all_labels, all_fake_cls = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval"):
            out = model(
                image=batch["image"].to(device),
                bert_ids=batch["bert_ids"].to(device),
                bert_mask=batch["bert_mask"].to(device),
                clip_pixels=batch["clip_pixels"].to(device),
                clip_ids=batch["clip_ids"].to(device),
                clip_mask=batch["clip_mask"].to(device),
            )
            probs = torch.sigmoid(out["logits"].squeeze(-1)).cpu()
            all_probs.append(probs)
            all_labels.append(batch["label"])
            all_fake_cls.extend(batch["fake_cls"])

    probs = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy()
    metrics = compute_metrics(labels, probs)

    print("\n=== FND-CLIP on MMFakeBench ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        yaml.safe_dump({k: float(v) for k, v in metrics.items()}, f)

    # Per-class breakdown
    df = pd.DataFrame({
        "prob": probs, "label": labels, "fake_cls": all_fake_cls,
        "pred": (probs >= 0.5).astype(int),
    })
    print("\nPer fake_cls accuracy:")
    for cls, sub in df.groupby("fake_cls"):
        acc = (sub["pred"] == sub["label"]).mean()
        print(f"  {cls:40s} n={len(sub):4d}  acc={acc:.3f}")


if __name__ == "__main__":
    main()
