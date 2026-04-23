"""FND-CLIP (Zhou et al., ICME 2023): ResNet + BERT + CLIP with modality attention."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from transformers import BertModel, CLIPModel, CLIPProcessor


class VisualStream(nn.Module):
    def __init__(self, out_dim=512, pretrained=True):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.project = nn.Linear(2048, out_dim)

    def forward(self, image):
        x = self.features(image).flatten(1)
        return self.project(x)


class TextStream(nn.Module):
    def __init__(self, out_dim=512, pretrained="bert-base-uncased"):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained)
        self.project = nn.Linear(768, out_dim)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.project(cls)


class CLIPStream(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(model_name)
        for p in self.clip.parameters():
            p.requires_grad = False

    def forward(self, pixel_values, input_ids, attention_mask):
        vision_out = self.clip.vision_model(pixel_values=pixel_values)
        img_pooled = vision_out.pooler_output if hasattr(vision_out, "pooler_output") \
            else vision_out.last_hidden_state[:, 0]
        img_feat = self.clip.visual_projection(img_pooled)

        text_out = self.clip.text_model(input_ids=input_ids, attention_mask=attention_mask)
        txt_pooled = text_out.pooler_output if hasattr(text_out, "pooler_output") \
            else text_out.last_hidden_state[:, 0]
        txt_feat = self.clip.text_projection(txt_pooled)

        img_feat = F.normalize(img_feat, dim=-1)
        txt_feat = F.normalize(txt_feat, dim=-1)

        sim = (img_feat * txt_feat).sum(dim=-1)
        fused = torch.cat([img_feat, txt_feat], dim=-1)
        return fused, sim


class ModalityAttention(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(feat_dim * 3, feat_dim),
            nn.Tanh(),
            nn.Linear(feat_dim, 3),
        )

    def forward(self, text_feat, image_feat, clip_feat):
        stack = torch.stack([text_feat, image_feat, clip_feat], dim=1)
        concat = stack.flatten(1)
        weights = F.softmax(self.scorer(concat), dim=-1)
        weighted = (stack * weights.unsqueeze(-1)).sum(dim=1)
        return weighted, weights


class FNDCLIP(nn.Module):
    def __init__(self, feat_dim=512, num_classes=5):
        super().__init__()
        self.visual = VisualStream(out_dim=feat_dim)
        self.text = TextStream(out_dim=feat_dim)
        self.clip = CLIPStream()
        self.clip_project = nn.Linear(1024, feat_dim)
        self.attention = ModalityAttention(feat_dim)
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(feat_dim // 2, num_classes),
        )

    def forward(self, image, bert_ids, bert_mask,
                clip_pixels, clip_ids, clip_mask):
        image_feat = self.visual(image)
        text_feat = self.text(bert_ids, bert_mask)
        clip_fused, clip_sim = self.clip(clip_pixels, clip_ids, clip_mask)

        clip_feat = self.clip_project(clip_fused)
        clip_feat = clip_feat * clip_sim.unsqueeze(-1)

        weighted, weights = self.attention(text_feat, image_feat, clip_feat)
        logits = self.classifier(weighted)

        return {
            "logits": logits,
            "clip_sim": clip_sim,
            "attn_weights": weights,
        }
