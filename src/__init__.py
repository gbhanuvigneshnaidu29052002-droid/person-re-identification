"""
Person Re-Identification (ReID) Deep Feature Embedding Package
Author: Bhanu Vignesh Naidu Ganeshna
"""

from .models import ReIDBackbone
from .dataset import ReIDDataset, generate_synthetic_reid_dataset
from .evaluator import ReIDEvaluator

__all__ = ["ReIDBackbone", "ReIDDataset", "generate_synthetic_reid_dataset", "ReIDEvaluator"]
