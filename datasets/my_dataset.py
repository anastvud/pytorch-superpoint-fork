import numpy as np
import torch
from pathlib import Path
import torch.utils.data as data
from utils.tools import dict_update
import cv2
import logging

# Inherit from Coco to get all the helper methods (gaussian blur, homography, etc.)
from datasets.Coco import Coco

class my_dataset(Coco):
    default_config = {
        "labels": None,
        "cache_in_memory": False,
        "validation_size": 100,
        "truncate": None,
        "preprocessing": {"resize": [360, 640]}, # [H, W]
        "num_parallel_calls": 10,
        "augmentation": {
            "photometric": {
                "enable": False,
                "primitives": "all",
                "params": {},
                "random_order": True,
            },
            "homographic": {"enable": False, "params": {}, "valid_border_margin": 0},
        },
        "warped_pair": {"enable": False, "params": {}, "valid_border_margin": 0},
        "homography_adaptation": {"enable": False},
    }

    def __init__(self, export=False, transform=None, task="train", **config):
        # 1. Update Config
        self.config = self.default_config
        self.config = dict_update(self.config, config)
        self.transforms = transform
        self.action = "train" if task == "train" else "val"

        # 2. Define Root Path
        # The config['root'] comes from your yaml file
        self.root = Path(self.config["root"]) 
        
        # If exporting, use the export_folder (e.g., 'train'), else use task
        if export:
             subfolder = self.config.get('export_folder', 'train')
             self.root = self.root / subfolder
        else:
             self.root = self.root / task

        # 3. Get Image Files
        # We search for .jpg and .png
        self.files = sorted(list(self.root.glob('*.jpg')) + list(self.root.glob('*.png')))
        
        if not self.files:
            logging.warning(f"No images found in {self.root}")

        # 4. Handle Labels (for training step later)
        if self.config["labels"]:
            self.labels = True
            self.labels_path = Path(self.config["labels"]) # Folder with .npz files
            logging.info(f"Loading labels from: {self.labels_path}")
        else:
            self.labels = False

        # 5. Build the samples list
        self.samples = self._make_dataset()
        
        # Initialize variables from parent class (Coco)
        self.init_var()

    def _make_dataset(self):
        """
        Creates a list of dicts, where each dict contains info for one sample.
        """
        samples = []
        for img_path in self.files:
            name = img_path.stem
            sample = {'image': str(img_path), 'name': name, 'scene_name': 'my_scene'}
            
            # If labels are enabled (Training step), add the path to the point file
            if self.labels:
                # Expects labels in: datasets/my_dataset_labels/train/image_name.npz
                p = self.labels_path / f"{name}.npz"
                if p.exists():
                    sample['points'] = str(p)
                    samples.append(sample)
                else:
                    # If training but label missing, skip or warn
                    # logging.debug(f"Label not found for {name}, skipping.")
                    pass
            else:
                # If no labels (Export step), just add the image
                samples.append(sample)
                
        return samples

    def get_img_from_sample(self, sample):
        # Helper required by Coco.__getitem__
        return sample['image']

    def format_sample(self, sample):
        # Helper required by Coco.__getitem__
        return sample