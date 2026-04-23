"""Generate PNG diagrams for the LLaVA OOC evaluation."""

import os
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path("outputs/v1_results/diagrams")
OUT.mkdir(parents=True, exist_ok=True)

# Color palette
C_IMG = "#d5e8f0"   # light blue  — image side
C_TXT = "#f5e1c0"   # sand        — text / LLM side
C_OUT = "#d8e7c4"   # light green — outputs
C_RED = "#fbdddd"   # light red   — soft/weak signal
C_EDGE = "#2f2f2f"


def box(ax, xy, w, h, text, face=C_IMG, fontsize=9, weight="normal"):
    x, y = xy
    bbox = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.2, edgecolor=C_EDGE, facecolor=face,
    )
    ax.add_patch(bbox)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, fontweight=weight, wrap=True)


def arrow(ax, start, end, label=None, curve=0.0, fontsize=8, color=C_EDGE):
    a = FancyArrowPatch(
        start, end,
        connectionstyle=f"arc3,rad={curve}",
        arrowstyle="-|>", mutation_scale=14,
        linewidth=1.3, color=color,
    )
    ax.add_patch(a)
    if label:
        mx = (start[0] + end[0]) / 2
        my = (start[1] + end[1]) / 2
        ax.text(mx, my + 0.08, label, ha="center", va="bottom",
                fontsize=fontsize, style="italic", color="#444")


def setup_ax(ax, xlim=(0, 14), ylim=(0, 10), title=None):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)


# =============================================================================
# Diagram 1: What LLaVA actually is inside
# =============================================================================
fig, ax = plt.subplots(figsize=(14, 7))
setup_ax(ax, (0, 14), (0, 8.5),
         "Diagram 1 — What LLaVA-1.5-7B actually is inside")

# Image path (top row)
box(ax, (0.3, 6.3), 2.2, 1.0, "Input image\n336x336 RGB", face=C_IMG, fontsize=9)
box(ax, (3.0, 6.3), 2.4, 1.0,
    "CLIP ViT-L/14\nvision encoder\n~300M params", face=C_IMG, fontsize=8)
box(ax, (5.9, 6.3), 2.3, 1.0,
    "MLP projector\n2-layer\nvision to text\nembedding space",
    face=C_IMG, fontsize=8)
box(ax, (8.7, 6.3), 2.3, 1.0,
    "576 image tokens\neach a 4096-dim\nvector", face=C_IMG, fontsize=8)

arrow(ax, (2.5, 6.8), (3.0, 6.8))
arrow(ax, (5.4, 6.8), (5.9, 6.8))
arrow(ax, (8.2, 6.8), (8.7, 6.8))

# Text path (bottom row)
box(ax, (0.3, 3.3), 2.2, 1.3,
    "Text prompt:\n'USER: <image>\nDoes image match\ntext? Yes or No.'",
    face=C_TXT, fontsize=8)
box(ax, (3.0, 3.3), 2.4, 1.3,
    "Llama tokenizer\nbreaks text into\nsub-word token ids",
    face=C_TXT, fontsize=8)
box(ax, (5.9, 3.3), 2.3, 1.3,
    "Text tokens\neach a 4096-dim\nvector",
    face=C_TXT, fontsize=8)

arrow(ax, (2.5, 3.95), (3.0, 3.95))
arrow(ax, (5.4, 3.95), (5.9, 3.95))

# Merge into LLM
box(ax, (8.7, 3.8), 2.3, 1.8,
    "Interleave\nimage + text\ntokens into one\nsequence", face=C_OUT, fontsize=9)
arrow(ax, (8.2, 3.95), (8.7, 4.4))
arrow(ax, (11.0, 6.8), (12.0, 5.2), curve=-0.2)  # from image tokens down
arrow(ax, (11.0, 4.7), (12.0, 5.2))  # from merge box to LLM

# LLM
box(ax, (12.0, 4.2), 1.8, 2.3,
    "Llama-2-7B\ndecoder\n32 layers\n7B params\n(all frozen)",
    face=C_TXT, fontsize=9, weight="bold")

# Outputs
box(ax, (11.5, 1.5), 2.7, 1.5,
    "Next-token logits\nover 32,000\nvocabulary words",
    face=C_OUT, fontsize=9)
arrow(ax, (12.9, 4.2), (12.9, 3.0))

box(ax, (8.2, 1.5), 2.8, 1.5,
    "We read logit of\n'Yes' and 'No' only\n-> softmax -> P(OOC)",
    face=C_OUT, fontsize=9, weight="bold")
arrow(ax, (11.5, 2.25), (11.0, 2.25))

# Legend / note
ax.text(7.0, 0.5,
        "Key: BLUE = vision path,  ORANGE = text / LLM path,  GREEN = outputs",
        ha="center", fontsize=9, style="italic")
ax.text(7.0, 0.1,
        "LLaVA was trained for general visual Q&A — never on news misinformation.",
        ha="center", fontsize=9, color="#666")

plt.tight_layout()
plt.savefig(OUT / "1_llava_architecture.png", dpi=160, bbox_inches="tight")
plt.close()
print(f"wrote {OUT/'1_llava_architecture.png'}")


# =============================================================================
# Diagram 2: Our evaluation pipeline — what WE did
# =============================================================================
fig, ax = plt.subplots(figsize=(12, 10))
setup_ax(ax, (0, 12), (0, 14),
         "Diagram 2 — Our evaluation pipeline (what we did)")

# Loop header
ax.text(6, 13.2, "For each of the 1,724 test samples:",
        ha="center", fontsize=11, style="italic", color="#444")
rect = patches.Rectangle((0.5, 2.0), 11, 10.5, linewidth=1.5,
                         edgecolor="#888", facecolor="none", linestyle="--")
ax.add_patch(rect)

# Steps in a vertical flow
steps = [
    (11.0, "CSV row -> text + image_path + true label", C_IMG),
    (10.0, "Load image (PIL), truncate text to 400 chars", C_IMG),
    (9.0,  "Format prompt template:\n'USER: <image>\\nText: {caption}\\nDoes the image match what the text describes?\\nAnswer with ONLY one word: Yes or No.\\nASSISTANT:'", C_TXT),
    (7.6,  "LlavaProcessor: tokenize text + preprocess image (336x336, normalize)", C_TXT),
    (6.6,  "model.forward(inputs) — single pass, NO generation", C_TXT),
    (5.6,  "Get logits[0, -1, :]  -> shape (32000,)  next-token distribution", C_OUT),
    (4.6,  "Index two specific tokens:\nlogit_yes = logits[ id('Yes') ]   logit_no = logits[ id('No') ]", C_OUT),
    (3.3,  "softmax([logit_no, logit_yes])  ->  P(OOC) = P(No),  P(aligned) = P(Yes)", C_OUT),
    (2.3,  "Predict OOC if P(OOC) >= 0.5, else aligned", C_OUT),
]
for i, (y, txt, color) in enumerate(steps):
    box(ax, (1.0, y), 10.0, 0.7, txt, face=color, fontsize=9)
    if i > 0:
        prev_y = steps[i - 1][0]
        arrow(ax, (6.0, prev_y), (6.0, y + 0.7))

# After loop
box(ax, (1.0, 0.8), 10.0, 0.8,
    "Aggregate all 1,724 predictions -> compute accuracy, precision, recall, F1, AUC-ROC",
    face="#e6d8ee", fontsize=10, weight="bold")
arrow(ax, (6.0, 2.0), (6.0, 1.6))

plt.tight_layout()
plt.savefig(OUT / "2_our_pipeline.png", dpi=160, bbox_inches="tight")
plt.close()
print(f"wrote {OUT/'2_our_pipeline.png'}")


# =============================================================================
# Diagram 3: logit extraction
# =============================================================================
fig, ax = plt.subplots(figsize=(14, 6.5))
setup_ax(ax, (0, 14), (0, 7),
         "Diagram 3 — LLaVA next-token logit extraction")

box(ax, (0.2, 3.0), 2.4, 1.4,
    "Llama final\nhidden state\n(last position)\n1 x 4096 vector",
    face=C_TXT, fontsize=9)

box(ax, (3.2, 3.0), 2.4, 1.4,
    "LM head:\nLinear(4096, 32000)", face=C_TXT, fontsize=9)

box(ax, (6.2, 3.0), 2.4, 1.4,
    "Logits vector:\n32,000 numbers\n(one per\nvocab token)", face=C_OUT, fontsize=9)

arrow(ax, (2.6, 3.7), (3.2, 3.7))
arrow(ax, (5.6, 3.7), (6.2, 3.7))

# Zoom-in on vocab indices
box(ax, (9.2, 5.0), 4.6, 0.6,
    "id(' Yes') = 8241 -> logit = 24.60", face=C_OUT, fontsize=9, weight="bold")
box(ax, (9.2, 4.2), 4.6, 0.6,
    "id(' No')  = 3782 -> logit = 24.57", face=C_OUT, fontsize=9, weight="bold")
box(ax, (9.2, 3.4), 4.6, 0.6,
    "(the other 31,998 logits — ignored)", face="#f0f0f0", fontsize=8)

arrow(ax, (8.6, 3.9), (9.2, 4.5))
arrow(ax, (8.6, 3.7), (9.2, 5.3))

# Softmax + interpretation
box(ax, (9.2, 1.6), 4.6, 1.4,
    "softmax([logit_no, logit_yes])\n= [0.49, 0.51]\n\nP(OOC) = P('No') = 0.49",
    face="#e6d8ee", fontsize=9, weight="bold")
arrow(ax, (11.5, 3.4), (11.5, 3.0))

# Why it matters (left side bottom)
box(ax, (0.2, 0.3), 8.5, 1.6,
    "WHY THIS MATTERS:\n"
    "- If we let LLaVA just generate a word, we get only a hard Yes/No per sample.\n"
    "- By reading the raw logits for 'Yes' and 'No', we get a continuous probability.\n"
    "- A continuous probability lets us compute AUC-ROC (the threshold-independent metric).\n"
    "- AUC-ROC is what actually compares properly vs FND-CLIP's sigmoid output.",
    face=C_IMG, fontsize=9)

plt.tight_layout()
plt.savefig(OUT / "3_logit_extraction.png", dpi=160, bbox_inches="tight")
plt.close()
print(f"wrote {OUT/'3_logit_extraction.png'}")


# =============================================================================
# Diagram 4: FND-CLIP vs LLaVA side by side
# =============================================================================
fig, ax = plt.subplots(figsize=(15, 9))
setup_ax(ax, (0, 15), (0, 11),
         "Diagram 4 — FND-CLIP (trained) vs LLaVA (zero-shot), side by side")

# LLaVA side (LEFT)
ax.add_patch(patches.Rectangle((0.1, 0.5), 7.0, 9.8, linewidth=1.5,
                               edgecolor="#c07070", facecolor="#faf3f3"))
ax.text(3.6, 9.9, "LLaVA-1.5-7B  (zero-shot)",
        ha="center", fontsize=12, fontweight="bold", color="#702020")

box(ax, (0.6, 8.5), 2.6, 0.9, "Text", face=C_TXT, fontsize=9)
box(ax, (3.7, 8.5), 2.6, 0.9, "Image", face=C_IMG, fontsize=9)

box(ax, (1.2, 6.8), 4.7, 1.2,
    "Prompt template wraps both\n(one fixed string, same for every sample)",
    face=C_TXT, fontsize=9)
arrow(ax, (1.9, 8.5), (2.5, 8.0))
arrow(ax, (5.0, 8.5), (4.5, 8.0))

box(ax, (1.5, 5.0), 4.1, 1.4,
    "Llama-2-7B decoder\n(7 billion params, frozen)",
    face=C_TXT, fontsize=10, weight="bold")
arrow(ax, (3.6, 6.8), (3.6, 6.4))

box(ax, (1.5, 3.4), 4.1, 1.1,
    "Read 2 logits: 'Yes', 'No'\n-> softmax -> P(OOC)",
    face=C_OUT, fontsize=9, weight="bold")
arrow(ax, (3.6, 5.0), (3.6, 4.5))

box(ax, (1.0, 1.0), 5.1, 1.9,
    "Output: narrow margin\n"
    "P(OOC|aligned) mean = 0.39\n"
    "P(OOC|ooc)     mean = 0.59\n\n"
    "acc = 0.79   AUC = 0.87",
    face=C_RED, fontsize=10, weight="bold")
arrow(ax, (3.6, 3.4), (3.6, 2.9))

# FND-CLIP side (RIGHT)
ax.add_patch(patches.Rectangle((7.9, 0.5), 7.0, 9.8, linewidth=1.5,
                               edgecolor="#70a070", facecolor="#f3faf3"))
ax.text(11.4, 9.9, "FND-CLIP  (trained)",
        ha="center", fontsize=12, fontweight="bold", color="#205020")

box(ax, (8.2, 8.5), 2.1, 0.9, "Text", face=C_TXT, fontsize=9)
box(ax, (12.5, 8.5), 2.1, 0.9, "Image", face=C_IMG, fontsize=9)

box(ax, (8.2, 7.0), 1.9, 1.1, "BERT\n[CLS]\n-> 512", face=C_TXT, fontsize=8)
box(ax, (10.3, 7.0), 2.4, 1.1, "CLIP text\n+ CLIP vision\n-> 512 each", face="#e0e8ee", fontsize=8)
box(ax, (12.8, 7.0), 1.9, 1.1, "ResNet-50\n-> 2048 -> 512", face=C_IMG, fontsize=8)
arrow(ax, (9.2, 8.5), (9.2, 8.1))
arrow(ax, (11.5, 8.5), (11.5, 8.1))
arrow(ax, (13.5, 8.5), (13.5, 8.1))

box(ax, (9.4, 5.5), 4.1, 1.1,
    "CLIP cosine similarity\nreweights fused vector\n(the FND-CLIP 'key idea')",
    face="#e0e8ee", fontsize=9)
arrow(ax, (11.5, 7.0), (11.5, 6.6))

box(ax, (8.5, 4.0), 5.9, 1.1,
    "Modality-wise attention\n3 scalar weights over (text, image, clip)",
    face="#dff0d8", fontsize=9, weight="bold")
arrow(ax, (9.2, 7.0), (10.0, 5.1), curve=-0.2)
arrow(ax, (11.5, 5.5), (11.5, 5.1))
arrow(ax, (13.5, 7.0), (13.0, 5.1), curve=0.2)

box(ax, (9.4, 2.6), 4.1, 1.1,
    "2-layer MLP + sigmoid\n-> direct P(OOC)",
    face=C_OUT, fontsize=9, weight="bold")
arrow(ax, (11.5, 4.0), (11.5, 3.7))

box(ax, (8.8, 1.0), 5.2, 1.2,
    "Output: crisp, confident\n"
    "acc = 0.96   AUC = 0.995",
    face="#d8e7c4", fontsize=10, weight="bold")
arrow(ax, (11.5, 2.6), (11.5, 2.2))

# Comparison arrow at bottom
ax.annotate("", xy=(11.3, 0.2), xytext=(3.7, 0.2),
            arrowprops=dict(arrowstyle="<-", color="#404040", lw=1.3))
ax.text(7.5, -0.0, "+17.6 accuracy points, +0.12 AUC",
        ha="center", fontsize=10, fontweight="bold", color="#404040")

plt.tight_layout()
plt.savefig(OUT / "4_fndclip_vs_llava.png", dpi=160, bbox_inches="tight")
plt.close()
print(f"wrote {OUT/'4_fndclip_vs_llava.png'}")

print("\nAll 4 diagrams generated in outputs/v1_results/diagrams/")
