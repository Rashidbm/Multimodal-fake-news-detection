"""Extract LLaVA Yes/No logits on MMFakeBench pairs."""

import argparse
import os
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm


PROMPT = (
    "USER: <image>\n{text}\n\n"
    "Is the above text-image pair misinformation? Answer with only 'Yes' or 'No'.\n"
    "ASSISTANT:"
)


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float16
    if torch.backends.mps.is_available():
        return torch.device("mps"), torch.float16
    return torch.device("cpu"), torch.float32


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Path to MMFakeBench dataset (HF arrow format)")
    p.add_argument("--images", required=True, help="Path to extracted images directory")
    p.add_argument("--output", required=True, help="Output CSV path")
    p.add_argument("--model", default="llava-hf/llava-1.5-7b-hf")
    p.add_argument("--limit", type=int, default=None, help="Limit samples for quick test")
    args = p.parse_args()

    from datasets import load_from_disk
    from transformers import LlavaForConditionalGeneration, AutoProcessor

    device, dtype = pick_device()
    print(f"Device: {device} | dtype: {dtype}")

    print(f"Loading dataset from {args.input}")
    ds = load_from_disk(args.input)
    if "train" in ds:
        ds = ds["train"]
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    print(f"Samples: {len(ds)}")

    print(f"Loading model {args.model}")
    processor = AutoProcessor.from_pretrained(args.model)
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=dtype, low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    # Get 'Yes' and 'No' token ids (take first token of each word)
    yes_id = processor.tokenizer("Yes", add_special_tokens=False).input_ids[0]
    no_id = processor.tokenizer("No", add_special_tokens=False).input_ids[0]
    print(f"Yes token id: {yes_id}, No token id: {no_id}")

    images_root = Path(args.images)
    rows = []
    for idx, sample in enumerate(tqdm(ds, desc="Running LLaVA")):
        text = sample["text"]
        rel_path = sample["image_path"].lstrip("/")
        img_path = images_root / rel_path
        if not img_path.exists():
            # Try without the root prefix
            img_path = images_root / Path(rel_path).name
        if not img_path.exists():
            continue
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        prompt = PROMPT.format(text=text[:500])  # truncate very long text
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, dtype)

        with torch.no_grad():
            out = model(**inputs)
        # Logits for the next token after the prompt (the answer)
        next_logits = out.logits[0, -1, :]  # (vocab,)
        yes_logit = float(next_logits[yes_id].item())
        no_logit = float(next_logits[no_id].item())
        probs = torch.softmax(torch.tensor([yes_logit, no_logit]), dim=0)
        p_yes = float(probs[0].item())
        prediction = "Yes" if p_yes > 0.5 else "No"

        rows.append({
            "sample_id": f"mmfb_{idx}",
            "text": text[:200],
            "image_path": str(img_path),
            "gt_answers": sample.get("gt_answers"),
            "fake_cls": sample.get("fake_cls"),
            "yes_logit": yes_logit,
            "no_logit": no_logit,
            "p_yes": p_yes,
            "llava_prediction": prediction,
        })

        # Periodic save in case of interrupt
        if (idx + 1) % 100 == 0:
            pd.DataFrame(rows).to_csv(args.output, index=False)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nSaved {len(df)} rows to {args.output}")
    print(f"\nQuick agreement stats (LLaVA prediction vs gt_answers):")
    if "gt_answers" in df.columns:
        df["gt_bin"] = df["gt_answers"].str.lower().isin(["fake", "false"]).astype(int)
        df["pred_bin"] = (df["llava_prediction"] == "Yes").astype(int)
        correct = (df["gt_bin"] == df["pred_bin"]).sum()
        print(f"  Accuracy: {correct / len(df):.3f} ({correct}/{len(df)})")


if __name__ == "__main__":
    main()
