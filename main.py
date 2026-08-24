"""
Main Training & Evaluation Script for Person Re-Identification
Author: Bhanu Vignesh Naidu Ganeshna
"""

import os
import sys
import argparse
import json
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models import ReIDBackbone
from src.dataset import ReIDDataset, generate_synthetic_reid_dataset
from src.evaluator import ReIDEvaluator

def main():
    parser = argparse.ArgumentParser(description="Person Re-ID Training and Benchmark Evaluation")
    parser.add_argument("--backbone", type=str, default="resnet50", choices=["resnet50", "mobilenet_v2"])
    parser.add_argument("--embedding_dim", type=int, default=512)
    parser.add_argument("--data_dir", type=str, default="data/reid_dataset")
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"⚡ Device initialized: {device}")
    
    print("📦 Generating Multi-Camera Re-ID Benchmark Dataset...")
    q_samples, g_samples = generate_synthetic_reid_dataset(args.data_dir, num_identities=25)
    
    train_transform = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0)
    ])

    test_transform = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    query_dataset = ReIDDataset(q_samples, transform=test_transform)
    gallery_dataset = ReIDDataset(g_samples, transform=train_transform)
    
    query_loader = DataLoader(query_dataset, batch_size=8, shuffle=False)
    gallery_loader = DataLoader(gallery_dataset, batch_size=8, shuffle=False)
    
    print(f"🧠 Initializing Re-ID Deep Neural Network Backbone ({args.backbone})...")
    model = ReIDBackbone(backbone=args.backbone, embedding_dim=args.embedding_dim, pretrained=True)
    evaluator = ReIDEvaluator(model=model, device=device)
    
    print("📊 Evaluating Re-ID Metrics (Rank-1, Rank-5, Rank-10, mAP)...")
    metrics, dist_matrix = evaluator.evaluate(query_loader, gallery_loader)
    
    print("=" * 60)
    print("🏆 PERSON RE-IDENTIFICATION BENCHMARK EVALUATION RESULTS")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"  - {k:<10}: {v * 100:.2f}%" if k != "mAP" else f"  - {k:<10}: {v:.4f}")
    print("=" * 60)
    
    results_file = os.path.join(args.output_dir, "reid_metrics.json")
    with open(results_file, "w") as f:
        json.dump(metrics, f, indent=4)
        
    print(f"✅ Evaluation complete. Metrics exported to {results_file}")

if __name__ == "__main__":
    main()
