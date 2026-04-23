"""Sanity-check one forward+backward pass of FND-CLIP on MPS."""
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from models.fnd_clip import FNDCLIP


def main():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    print("Building FND-CLIP...")
    t0 = time.time()
    model = FNDCLIP(feat_dim=512, num_classes=1).to(device)
    print(f"  built in {time.time() - t0:.1f}s")
    print(f"  params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    B = 2
    batch = {
        "image": torch.randn(B, 3, 224, 224, device=device),
        "bert_ids": torch.randint(0, 30000, (B, 128), device=device),
        "bert_mask": torch.ones(B, 128, dtype=torch.long, device=device),
        "clip_pixels": torch.randn(B, 3, 224, 224, device=device),
        "clip_ids": torch.randint(0, 49000, (B, 77), device=device),
        "clip_mask": torch.ones(B, 77, dtype=torch.long, device=device),
        "label": torch.randint(0, 2, (B,), device=device).float(),
    }

    print("\nForward pass...")
    t0 = time.time()
    model.train()
    out = model(
        image=batch["image"],
        bert_ids=batch["bert_ids"],
        bert_mask=batch["bert_mask"],
        clip_pixels=batch["clip_pixels"],
        clip_ids=batch["clip_ids"],
        clip_mask=batch["clip_mask"],
    )
    print(f"  forward in {time.time() - t0:.2f}s")
    print(f"  logits shape: {out['logits'].shape}")
    print(f"  clip_sim: {out['clip_sim'].detach().cpu().tolist()}")
    print(f"  attn_weights[0]: {out['attn_weights'][0].detach().cpu().tolist()}")

    print("\nBackward pass...")
    t0 = time.time()
    logits = out["logits"].squeeze(-1)
    loss = nn.BCEWithLogitsLoss()(logits, batch["label"])
    loss.backward()
    print(f"  backward in {time.time() - t0:.2f}s")
    print(f"  loss: {loss.item():.4f}")

    print("\nMPS smoke test PASSED")


if __name__ == "__main__":
    main()
