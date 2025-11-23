# rebuttal_project/src/blending/detector.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from torchvision import transforms
import timm
import os
from tqdm import tqdm

class BlendingDetector:
    """Encapsulates loading and inference for the Blending detector.

    Supports batched inference to improve GPU utilization (default batch_size=64).
    """

    def __init__(self, model_name, weights_path, img_size, num_class, device):
        """
        Initialize the BlendingDetector.

        Args:
            model_name (str): Model name in timm.
            weights_path (str): Absolute path to weights file.
            img_size (int): Input image size.
            num_class (int): Number of classes for classification.
            device (str): PyTorch device.
        """
        self.device = device
        self.img_size = img_size
        
        print(f"--- Initializing BlendingDetector: Loading Model from {weights_path} ---")
        self.model = self._load_network(model_name, weights_path, num_class).to(self.device)
        self.model.eval()
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass
        
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        print("--- BlendingDetector Initialized Successfully ---")

    def _load_network(self, model_name, save_filename, num_class):
        """Load network and handle potential DataParallel 'module.' prefixes."""
        model = timm.create_model(model_name, pretrained=False, num_classes=num_class)
        state_dict = torch.load(save_filename, map_location='cpu')
        
        # Handle models trained under DataParallel or DDP
        if next(iter(state_dict)).startswith('module.'):
            state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
            
        model.load_state_dict(state_dict)
        return model

    def _load_image(self, image_path):
        """Load image with OpenCV and convert to RGB."""
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None: return None
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def infer(self, image_paths, batch_size: int = 64, num_workers: int = 4, pin_memory: bool = True):
        """
        Run batched inference on a list of images and return 'fake' scores.

        Args:
            image_paths (List[str]): List of image paths.
            batch_size (int): Batch size (default 64).
            num_workers (int): Dataloader worker count (default 4).
            pin_memory (bool): Use pinned memory to speed up H2D (default True).

        Returns:
            dict: Mapping {image_path: {"score": value}}.
        """
        all_results = {}

        class _ImageDataset(Dataset):
            def __init__(self, paths):
                self.paths = list(paths)

            def __len__(self):
                return len(self.paths)

            def __getitem__(self, idx):
                p = self.paths[idx]
                img = self._load_image(p)
                if img is None:
                    return None, p
                tensor = self.preprocess(img)
                return tensor, p

            # bind outer methods/attrs
            _load_image = self._load_image
            preprocess = self.preprocess

        def _collate(samples):
            ok_tensors = []
            ok_paths = []
            failed_paths = []
            for t, p in samples:
                if t is None:
                    failed_paths.append(p)
                else:
                    ok_tensors.append(t)
                    ok_paths.append(p)
            batch = torch.stack(ok_tensors, 0) if ok_tensors else None
            return batch, ok_paths, failed_paths

        ds = _ImageDataset(image_paths)
        loader = DataLoader(
            ds,
            batch_size=max(1, int(batch_size)),
            shuffle=False,
            num_workers=max(0, int(num_workers)),
            pin_memory=bool(pin_memory),
            drop_last=False,
            collate_fn=_collate,
        )

        with torch.no_grad():
            for batch_cpu, ok_paths, failed_paths in tqdm(loader, desc=f"Blending Detector Inference (bs={batch_size})"):
                # Log failed reads
                for p in failed_paths:
                    print(f"\nWarning: Failed to load image {p}, skipping.")
                    all_results[p] = {"error": "Failed to load image"}

                if batch_cpu is None or not ok_paths:
                    continue

                batch = batch_cpu.to(self.device, non_blocking=True)
                outputs = self.model(batch)
                predictions = torch.nn.functional.softmax(outputs, dim=-1)
                fake_scores = predictions[:, 1].detach().cpu().tolist()
                for p, s in zip(ok_paths, fake_scores):
                    all_results[p] = {"score": float(s)}

        return all_results
