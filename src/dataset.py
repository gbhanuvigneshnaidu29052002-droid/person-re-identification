"""
ReID Dataset Loader and Synthetic Multi-View Person Generator
Author: Bhanu Vignesh Naidu Ganeshna
"""

import os
import torch
import numpy as np
from PIL import Image, ImageDraw
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class ReIDDataset(Dataset):
    """
    Dataset class for Person Re-ID query and gallery samples.
    """
    def __init__(self, samples, transform=None):
        self.samples = samples  # List of (img_path_or_tensor, person_id, camera_id)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, person_id, cam_id = self.samples[idx]
        if isinstance(img, str):
            img = Image.open(img).convert('RGB')
        elif isinstance(img, np.ndarray):
            img = Image.fromarray(img)
            
        if self.transform:
            img = self.transform(img)
            
        return img, person_id, cam_id

def generate_synthetic_reid_dataset(output_dir, num_identities=20, imgs_per_cam=4, img_size=(128, 64)):
    """
    Generates a structured multi-camera person Re-ID benchmark dataset.
    """
    os.makedirs(os.path.join(output_dir, "query"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "gallery"), exist_ok=True)
    
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0),
        (0, 128, 128), (128, 128, 0), (128, 0, 0), (0, 128, 0)
    ]
    
    query_samples = []
    gallery_samples = []
    
    for pid in range(num_identities):
        base_color = colors[pid % len(colors)]
        
        # Query image (Camera 1)
        q_img = Image.new('RGB', (img_size[1], img_size[0]), color=(240, 240, 240))
        draw = ImageDraw.Draw(q_img)
        draw.rectangle([10, 10, 54, 30], fill=base_color) # Torso
        draw.rectangle([15, 30, 49, 118], fill=(50, 50, 50)) # Legs
        q_path = os.path.join(output_dir, "query", f"person_{pid:03d}_cam1_001.jpg")
        q_img.save(q_path)
        query_samples.append((q_path, pid, 1))
        
        # Gallery images (Camera 2 & 3 with noise/variations)
        for c in range(2, 4):
            for i in range(imgs_per_cam):
                g_img = Image.new('RGB', (img_size[1], img_size[0]), color=(220 + np.random.randint(-15, 15), 220, 220))
                draw = ImageDraw.Draw(g_img)
                # Slightly varied torso color to simulate illumination changes
                var_color = tuple(max(0, min(255, c_val + np.random.randint(-30, 30))) for c_val in base_color)
                draw.rectangle([10, 10, 54, 30], fill=var_color)
                draw.rectangle([15, 30, 49, 118], fill=(50, 50, 50))
                
                g_path = os.path.join(output_dir, "gallery", f"person_{pid:03d}_cam{c}_{i:02d}.jpg")
                g_img.save(g_path)
                gallery_samples.append((g_path, pid, c))
                
    return query_samples, gallery_samples
