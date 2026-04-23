# Dataset Download Guide

Download each dataset into `data/raw/`. All three require filling forms or accepting licenses.

## 1. MFakeBench
- Paper: arXiv 2406.08772
- Primary source for fake/manipulated classes
- Repo: https://github.com/ (check paper for current link)
- Expected path: `data/raw/MFakeBench/`

## 2. NewsCLIPpings
- Paper: arXiv 2104.05893
- Source for real samples and out-of-context pairs
- Repo: https://github.com/g-luo/news_clippings
- Download script in the repo
- Expected path: `data/raw/NewsCLIPpings/`

## 3. DGM4
- Paper: arXiv 2304.02556
- Source for Class 5 (fake text + fake image)
- Repo: https://github.com/rshaojimmy/MultiModal-DeepFake
- HuggingFace: huggingface.co/datasets/rshaojimmy/DGM4
- Expected path: `data/raw/DGM4/`

## After downloading

```bash
# Build balanced dataset
python scripts/build_balanced_dataset.py \
    --mfakebench data/raw/MFakeBench \
    --newsclippings data/raw/NewsCLIPpings \
    --dgm4 data/raw/DGM4 \
    --output data/processed/balanced_dataset.csv

# Train FND-CLIP
python src/train.py --config config/v1.yaml
```
