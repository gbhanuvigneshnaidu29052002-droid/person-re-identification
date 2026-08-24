"""
Deep Feature Embedding Architecture for Person Re-Identification
Author: Bhanu Vignesh Naidu Ganeshna
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights, mobilenet_v2, MobileNet_V2_Weights

class ReIDBackbone(nn.Module):
    """
    Deep CNN Feature Embedding Extractor for Person Re-ID.
    Supports ResNet-50 and MobileNetV2 backbones with L2 embedding normalization.
    """
    def __init__(self, backbone='resnet50', embedding_dim=512, pretrained=True):
        super(ReIDBackbone, self).__init__()
        self.backbone_name = backbone
        self.embedding_dim = embedding_dim
        
        if backbone == 'resnet50':
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            base_model = resnet50(weights=weights)
            in_features = base_model.fc.in_features
            base_model.fc = nn.Identity()
            self.base = base_model
        elif backbone == 'mobilenet_v2':
            weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
            base_model = mobilenet_v2(weights=weights)
            in_features = base_model.classifier[1].in_features
            base_model.classifier = nn.Identity()
            self.base = base_model
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
            
        self.embedding_head = nn.Sequential(
            nn.Linear(in_features, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(embedding_dim, embedding_dim)
        )

    def extract_features(self, x):
        """
        Extracts L2-normalized feature vectors.
        """
        features = self.base(x)
        embeddings = self.embedding_head(features)
        normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
        return normalized_embeddings

    def forward(self, x):
        return self.extract_features(x)
