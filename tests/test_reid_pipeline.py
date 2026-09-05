"""
Automated Unit Tests for Person Re-Identification (ReID) Pipeline
Author: Bhanu Vignesh Naidu Ganeshna
"""

import os
import sys
import unittest
import shutil
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models import ReIDBackbone
from src.dataset import generate_synthetic_reid_dataset, ReIDDataset
from src.evaluator import ReIDEvaluator


class TestReIDPipeline(unittest.TestCase):
    def setUp(self):
        self.test_data_dir = os.path.join(os.path.dirname(__file__), "test_data_temp")
        os.makedirs(self.test_data_dir, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.test_data_dir):
            shutil.rmtree(self.test_data_dir, ignore_errors=True)

    def test_reid_backbone_embedding_and_norm(self):
        """Verify that ReIDBackbone extracts correct feature dimensions and L2-normalized vectors."""
        model = ReIDBackbone(backbone="mobilenet_v2", embedding_dim=256, pretrained=False)
        model.eval()
        dummy_input = torch.randn(4, 3, 256, 128)
        with torch.no_grad():
            embeddings = model(dummy_input)

        self.assertEqual(embeddings.shape, (4, 256))
        norms = torch.norm(embeddings, p=2, dim=1).cpu().numpy()
        for norm in norms:
            self.assertAlmostEqual(float(norm), 1.0, places=4)

    def test_synthetic_reid_dataset_generation(self):
        """Verify generation of multi-camera query and gallery samples."""
        q_samples, g_samples = generate_synthetic_reid_dataset(self.test_data_dir, num_identities=5)
        self.assertGreater(len(q_samples), 0)
        self.assertGreater(len(g_samples), 0)

        # Test ReIDDataset loading
        dataset = ReIDDataset(q_samples)
        self.assertEqual(len(dataset), len(q_samples))
        sample_img, pid, camid = dataset[0]
        self.assertIsInstance(pid, int)
        self.assertIsInstance(camid, int)

    def test_evaluator_metrics_logic(self):
        """Verify full evaluation pipeline with synthetic data loaders."""
        q_samples, g_samples = generate_synthetic_reid_dataset(self.test_data_dir, num_identities=4)
        transform = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
        ])
        q_loader = DataLoader(ReIDDataset(q_samples, transform=transform), batch_size=4, shuffle=False)
        g_loader = DataLoader(ReIDDataset(g_samples, transform=transform), batch_size=4, shuffle=False)

        model = ReIDBackbone(backbone="mobilenet_v2", embedding_dim=128, pretrained=False)
        evaluator = ReIDEvaluator(model=model, device="cpu")

        metrics, dist_matrix = evaluator.evaluate(q_loader, g_loader)
        self.assertIn("Rank-1", metrics)
        self.assertIn("Rank-5", metrics)
        self.assertIn("mAP", metrics)
        self.assertTrue(0.0 <= metrics["mAP"] <= 1.0)
        self.assertEqual(dist_matrix.shape, (len(q_samples), len(g_samples)))


if __name__ == "__main__":
    unittest.main()
