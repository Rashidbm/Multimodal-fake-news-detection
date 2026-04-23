"""Dataset loader: reads the balanced CSV, prepares ResNet/BERT/CLIP inputs."""

import os
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import BertTokenizer, CLIPProcessor


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_image_transform(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class MultiGuardDataset(Dataset):
    # CSV columns: sample_id, text, image_path, text_label, image_label, scenario, source
    # V1 binary label: 0 if scenario == 4 (genuine pair) else 1.

    def __init__(self, csv_path, train=True,
                 bert_model="bert-base-uncased",
                 clip_model="openai/clip-vit-base-patch32",
                 max_length=128):
        self.df = pd.read_csv(csv_path)
        self.max_length = max_length
        self.image_transform = build_image_transform(train=train)
        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model)

    def __len__(self):
        return len(self.df)

    def _load_image(self, path):
        try:
            return Image.open(path).convert("RGB")
        except Exception:
            return Image.new("RGB", (224, 224), color=0)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row["text"])
        pil_image = self._load_image(row["image_path"])

        image = self.image_transform(pil_image)

        bert = self.bert_tokenizer(
            text, padding="max_length", truncation=True,
            max_length=self.max_length, return_tensors="pt",
        )

        clip = self.clip_processor(
            images=pil_image, text=text, return_tensors="pt",
            padding="max_length", truncation=True, max_length=77,
        )

        scenario = int(row["scenario"])
        label = 0 if scenario == 4 else 1

        return {
            "image": image,
            "bert_ids": bert["input_ids"].squeeze(0),
            "bert_mask": bert["attention_mask"].squeeze(0),
            "clip_pixels": clip["pixel_values"].squeeze(0),
            "clip_ids": clip["input_ids"].squeeze(0),
            "clip_mask": clip["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.float32),
            "scenario": torch.tensor(scenario, dtype=torch.long),
        }
