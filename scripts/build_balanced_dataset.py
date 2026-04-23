"""Build balanced_dataset.csv from DGM4 + NewsCLIPpings (5-scenario layout)."""

import argparse
import hashlib
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd

SEED = 42
random.seed(SEED)

SCENARIOS = {
    1: "real_text_real_image_ooc",
    2: "fake_text_real_image",
    3: "real_text_fake_image",
    4: "real_text_real_image_genuine",
    5: "fake_text_fake_image",
}


# ---------------------------------------------------------------------------
# Dataset loaders - adjust paths/fields to match each dataset's actual schema
# ---------------------------------------------------------------------------

def load_mmfakebench(root, split="val"):
    """
    MMFakeBench (liuxuannan/MMFakeBench) real schema:
        text, image_path (relative, starts with / like '/real/bbc_val_50/BBC_val_0.png'),
        text_source, image_source, gt_answers ('True'|'Fake'),
        fake_cls (e.g. 'original', 'textual_veracity_distortion',
                  'visual_veracity_distortion', 'cross_modal_inconsistency',
                  'textual+visual', 'textual+cross_modal', ...)

    We use it mostly for EVALUATION comparing against LLaVA baseline.
    Images must be extracted under root/images/MMFakeBench_{split}/...
    """
    from datasets import load_from_disk
    root = Path(root)
    hf_path = root / "liuxuannan___mm_fake_bench" / f"MMFakeBench_{split}"
    # Find the specific version dir
    if hf_path.exists():
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

    images_root = root / "images" / f"MMFakeBench_{split}"
    rows = []
    for idx, item in enumerate(ds):
        text = item["text"]
        rel = item["image_path"].lstrip("/")
        img_path = images_root / rel

        fake_cls = item.get("fake_cls", "original")
        # Map MMFakeBench fake_cls -> our text/image labels
        text_fake = "textual" in fake_cls
        image_fake = "visual" in fake_cls
        is_ooc = "cross_modal" in fake_cls
        # 'original' is the only fully real case
        is_original = fake_cls == "original"

        rows.append({
            "sample_id": f"mmfb_{split}_{idx}",
            "text": text,
            "image_path": str(img_path),
            "text_label": "fake" if text_fake else "real",
            "image_label": "fake" if image_fake else "real",
            "is_ooc": is_ooc,
            "gt_answers": item.get("gt_answers"),
            "fake_cls": fake_cls,
            "source": f"MMFakeBench_{split}",
        })
    return pd.DataFrame(rows)


def load_newsclippings(root):
    """NewsCLIPpings: real news with genuine or out-of-context pairings."""
    root = Path(root)
    ann = root / "news_clippings" / "data" / "merged_balanced" / "train.json"
    if not ann.exists():
        ann = root / "train.json"
    with open(ann) as f:
        data = json.load(f)

    annotations = data.get("annotations", data)
    rows = []
    for item in annotations:
        text = item.get("caption", "")
        image = item.get("image_path", "")
        falsified = item.get("falsified", False)
        rows.append({
            "sample_id": item.get("id", f"nclip_{len(rows)}"),
            "text": text,
            "image_path": str(root / image),
            "text_label": "real",
            "image_label": "real",
            "is_ooc": falsified,
            "source": "NewsCLIPpings",
        })
    return pd.DataFrame(rows)


def load_dgm4(root, split="train"):
    """
    DGM4 (real schema verified from rshaojimmy/DGM4 HuggingFace):
    - fake_cls values:
        'orig'                          -> real text + real image  (scenario 4)
        'face_swap', 'face_attribute'   -> image-only manipulation (scenario 3)
        'text_swap', 'text_attribute'   -> text-only manipulation  (scenario 2)
        combinations with '&'           -> both manipulated        (scenario 5)

    Image paths in metadata are relative like 'DGM4/origin/usa_today/0048/120.jpg'
    Actual extracted images end up at root/origin/usa_today/0048/120.jpg, so
    we strip the leading 'DGM4/' prefix.
    """
    root = Path(root)
    ann = root / "metadata" / f"{split}.json"
    with open(ann) as f:
        data = json.load(f)

    rows = []
    for item in data:
        text = item.get("text", "")
        image_rel = item.get("image", "")
        if image_rel.startswith("DGM4/"):
            image_rel = image_rel[len("DGM4/"):]
        image_path = str(root / image_rel)

        fake_cls = item.get("fake_cls", "orig")
        text_fake = "text" in fake_cls
        image_fake = "face" in fake_cls

        rows.append({
            "sample_id": f"dgm4_{item.get('id', len(rows))}",
            "text": text,
            "image_path": image_path,
            "text_label": "fake" if text_fake else "real",
            "image_label": "fake" if image_fake else "real",
            "is_ooc": False,
            "source": "DGM4",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Scenario assignment
# ---------------------------------------------------------------------------

def assign_scenario(row):
    text_real = row["text_label"] == "real"
    image_real = row["image_label"] == "real"
    is_ooc = row.get("is_ooc", False)

    if text_real and image_real and is_ooc:
        return 1  # real text + real image, out of context
    if not text_real and image_real:
        return 2  # fake text + real image
    if text_real and not image_real:
        return 3  # real text + fake image
    if text_real and image_real and not is_ooc:
        return 4  # both real, genuine
    if not text_real and not image_real:
        return 5  # both fake
    return None


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def hash_row(row):
    text = str(row.get("text", "")).strip().lower()
    img = str(row.get("image_path", "")).strip()
    return hashlib.md5(f"{text}||{img}".encode("utf-8")).hexdigest()


def deduplicate(df):
    before = len(df)
    df["_hash"] = df.apply(hash_row, axis=1)
    df = df.drop_duplicates(subset="_hash").drop(columns="_hash")
    print(f"Deduplication: {before} -> {len(df)} ({before - len(df)} removed)")
    return df


# ---------------------------------------------------------------------------
# Balancing
# ---------------------------------------------------------------------------

def balance_classes(df):
    counts = df["scenario"].value_counts().to_dict()
    print("\nClass counts before balancing:")
    for s in sorted(SCENARIOS):
        print(f"  Class {s} ({SCENARIOS[s]}): {counts.get(s, 0)}")

    present = [s for s in SCENARIOS if counts.get(s, 0) > 0]
    if len(present) < 5:
        missing = set(SCENARIOS) - set(present)
        print(f"\nWARNING: No samples for classes: {sorted(missing)}")
        print("Pipeline will need additional sources for these classes.")

    min_count = min(counts[s] for s in present)
    print(f"\nBalancing all classes to: {min_count} samples")

    balanced = []
    for s in present:
        subset = df[df["scenario"] == s].sample(n=min_count, random_state=SEED)
        balanced.append(subset)
    return pd.concat(balanced, ignore_index=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dgm4", default=None)
    p.add_argument("--newsclippings", default=None)
    p.add_argument("--mmfakebench", default=None, help="Reserved for eval, not training")
    p.add_argument("--output", default="data/processed/balanced_dataset.csv")
    args = p.parse_args()

    dfs = []
    # Primary: DGM4 (covers scenarios 2, 3, 4, 5)
    if args.dgm4 and os.path.exists(args.dgm4):
        print("Loading DGM4 (primary)...")
        dfs.append(load_dgm4(args.dgm4, split="val"))

    if args.newsclippings and os.path.exists(args.newsclippings):
        print("Loading NewsCLIPpings (for Scenario 1 OOC)...")
        dfs.append(load_newsclippings(args.newsclippings))

    if args.mmfakebench and os.path.exists(args.mmfakebench):
        print("Loading MMFakeBench val (held out for benchmark eval)...")
        # We DON'T include this in training set - reserved for eval
        pass  # build_balanced_dataset is for training; use eval pipeline for MMFB

    combined = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal samples loaded: {len(combined)}")

    combined = deduplicate(combined)

    combined["scenario"] = combined.apply(assign_scenario, axis=1)
    before = len(combined)
    combined = combined.dropna(subset=["scenario"])
    combined["scenario"] = combined["scenario"].astype(int)
    if before != len(combined):
        print(f"Dropped {before - len(combined)} samples that did not fit any scenario")

    balanced = balance_classes(combined)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    balanced.to_csv(args.output, index=False)

    print(f"\nSaved: {args.output}")
    print(f"Final dataset size: {len(balanced)}")
    print("\nFinal class distribution:")
    for s, count in balanced["scenario"].value_counts().sort_index().items():
        print(f"  Class {s} ({SCENARIOS[s]}): {count}")
    print("\nSource distribution:")
    for src, count in balanced["source"].value_counts().items():
        print(f"  {src}: {count}")


if __name__ == "__main__":
    main()
