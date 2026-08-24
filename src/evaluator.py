"""
ReID Benchmark Evaluator: Rank-1, Rank-5, Rank-10 & mAP Metrics Calculation
Author: Bhanu Vignesh Naidu Ganeshna
"""

import numpy as np
import torch
import torch.nn.functional as F

class ReIDEvaluator:
    """
    Computes CMC (Cumulative Matching Characteristics) and mAP for Person Re-ID.
    """
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.model.eval()

    def extract_features_loader(self, dataloader):
        """
        Extracts features and metadata from dataloader.
        """
        features_list = []
        pids_list = []
        cams_list = []
        
        with torch.no_grad():
            for imgs, pids, cams in dataloader:
                imgs = imgs.to(self.device)
                feats = self.model(imgs)
                features_list.append(feats.cpu().numpy())
                pids_list.extend(pids.numpy())
                cams_list.extend(cams.numpy())
                
        features = np.concatenate(features_list, axis=0)
        pids = np.array(pids_list)
        cams = np.array(cams_list)
        return features, pids, cams

    def evaluate(self, query_loader, gallery_loader):
        """
        Computes Rank-1, Rank-5, Rank-10 accuracy and Mean Average Precision (mAP).
        """
        q_feats, q_pids, q_cams = self.extract_features_loader(query_loader)
        g_feats, g_pids, g_cams = self.extract_features_loader(gallery_loader)
        
        # Cosine distance matrix: 1 - cosine_similarity
        dist_matrix = 1.0 - np.dot(q_feats, g_feats.T)
        
        num_q, num_g = dist_matrix.shape
        all_cmc = []
        all_ap = []
        
        for i in range(num_q):
            q_pid = q_pids[i]
            q_cam = q_cams[i]
            
            # Remove gallery samples from the same camera and identity
            order = np.argsort(dist_matrix[i])
            remove = (g_pids[order] == q_pid) & (g_cams[order] == q_cam)
            keep = np.invert(remove)
            
            orig_cmc = matches = (g_pids[order][keep] == q_pid).astype(np.int32)
            if not np.any(orig_cmc):
                continue
                
            # Compute Average Precision (AP)
            num_rel = orig_cmc.sum()
            tmp_cmc = orig_cmc.cumsum()
            tmp_cmc = [x / (i + 1.0) for i, x in enumerate(tmp_cmc)]
            tmp_cmc = np.array(tmp_cmc) * orig_cmc
            ap = tmp_cmc.sum() / num_rel
            all_ap.append(ap)
            
            # Compute Cumulative Matching Characteristics (CMC)
            cmc = orig_cmc.cumsum()
            cmc[cmc > 1] = 1
            all_cmc.append(cmc[:10])
            
        all_cmc = np.asarray(all_cmc).astype(np.float32)
        cmc = all_cmc.mean(axis=0)
        mAP = np.mean(all_ap)
        
        metrics = {
            'Rank-1': float(cmc[0]),
            'Rank-5': float(cmc[4]) if len(cmc) >= 5 else float(cmc[-1]),
            'Rank-10': float(cmc[9]) if len(cmc) >= 10 else float(cmc[-1]),
            'mAP': float(mAP)
        }
        
        # Export CMC Curve plot
        import os
        import matplotlib.pyplot as plt
        os.makedirs("results", exist_ok=True)
        
        ranks = np.arange(1, len(cmc) + 1)
        plt.figure(figsize=(7, 5))
        plt.plot(ranks, cmc * 100, marker='o', color='#3498DB', lw=2.5, label=f'ResNet-50 Baseline (mAP = {mAP:.4f})')
        plt.xlabel('Rank Position', fontsize=11)
        plt.ylabel('Cumulative Matching Probability (%)', fontsize=11)
        plt.title('Person Re-ID: Cumulative Matching Characteristics (CMC)', fontsize=12, fontweight='bold')
        plt.xticks(ranks)
        plt.ylim([0, 105])
        plt.grid(True, alpha=0.3)
        plt.legend(loc='lower right', fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join("results", "cmc_curve.png"), dpi=300)
        plt.close()
        
        return metrics, dist_matrix
