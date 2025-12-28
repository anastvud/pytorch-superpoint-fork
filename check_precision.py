import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging

# Repository imports
from utils.loader import dataLoader
from models.model_wrap import SuperPointFrontend_torch

def compute_metrics(pred_pts, gt_pts, dist_thresh=4):
    """
    Calculates Precision and Recall.
    pred_pts: (3, N) [x, y, prob]
    gt_pts:   (2, M) [x, y]
    """
    # Safety check for empty arrays
    if pred_pts.size == 0 or gt_pts.size == 0:
        return 0.0, 0.0

    # Transpose to (N, 2) and (M, 2)
    pred_coords = pred_pts[:2, :].T
    gt_coords = gt_pts[:2, :].T

    # Calculate distance matrix (N, M)
    diff = pred_coords[:, None, :] - gt_coords[None, :, :]
    dist = np.linalg.norm(diff, axis=2)

    # PRECISION: For each prediction, is there a GT point close enough?
    min_dist_pred = dist.min(axis=1)
    correct_preds = (min_dist_pred <= dist_thresh).sum()
    precision = correct_preds / max(pred_pts.shape[1], 1)

    # RECALL: For each GT point, is there a prediction close enough?
    min_dist_gt = dist.min(axis=0)
    found_gts = (min_dist_gt <= dist_thresh).sum()
    recall = found_gts / max(gt_pts.shape[1], 1)

    return precision, recall

def get_ground_truth_points(sample):
    """
    Extracts ground truth points from the dataset sample.
    """
    if 'labels_2D' in sample:
        heatmap = sample['labels_2D'].detach().cpu().numpy().squeeze() # [H, W]
        ys, xs = np.where(heatmap >= 1.0)
        return np.stack([xs, ys], axis=0)
    return np.zeros((2, 0))

def main(config_path, model_path):
    # 1. Load Configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running evaluation on device: {device}")

    # 2. Setup Config for Eval
    config['model']['batch_size'] = 1
    config['model']['eval_batch_size'] = 1
    if 'warped_pair' in config['data']:
        config['data']['warped_pair']['enable'] = False

    # 3. Load Data
    print(f"Loading validation set from: {config['data']['root']}/val")
    data = dataLoader(config, dataset='my_dataset', warp_input=False)
    val_loader = data['val_loader']
    
    if len(val_loader) == 0:
        print("ERROR: Validation loader is empty!")
        return

    # 4. Load Model
    print(f"Loading model weights: {model_path}")
    fe = SuperPointFrontend_torch(
        config=config,
        weights_path=model_path,
        nms_dist=config['model']['nms'],
        conf_thresh=config['model']['detection_threshold'],
        nn_thresh=0.7,
        cuda=True,
        device=device
    )

    # 5. Evaluation Loop
    precision_list = []
    recall_list = []
    
    print("Starting precision check...")
    
    for i, sample in tqdm(enumerate(val_loader), total=len(val_loader)):
        img = sample['image'].to(device)
        
        # A. Get Predictions
        prediction = fe.run(img)
        
        # Handle tuple return (pts, desc, heatmap) vs just pts
        if isinstance(prediction, tuple):
            pred_pts = prediction[0] 
        else:
            pred_pts = prediction
            
        # --- FIX: ROBUST CONVERSION START ---
        # 1. If it's a list, convert to numpy
        if isinstance(pred_pts, list):
            pred_pts = np.array(pred_pts)

        # 2. Check if None or Empty
        if pred_pts is None or pred_pts.size == 0:
            pred_pts = np.zeros((3, 0))
        
        # 3. Ensure correct shape (3, N)
        # If shape is (N, 3), transpose it
        if pred_pts.ndim == 2 and pred_pts.shape[0] != 3 and pred_pts.shape[1] == 3:
            pred_pts = pred_pts.T
            
        # 4. Final safety check for dimensions
        if pred_pts.ndim != 2:
             pred_pts = np.zeros((3, 0))
        # --- FIX END ---

        # B. Get Ground Truth
        gt_pts = get_ground_truth_points(sample)
        
        # Skip if no GT points exist for this image
        if gt_pts.shape[1] == 0:
            continue

        # C. Calculate Metrics
        p, r = compute_metrics(pred_pts, gt_pts, dist_thresh=4)
        
        # D. Append to lists
        precision_list.append(p)
        recall_list.append(r)

    # 6. Final Report
    if len(precision_list) > 0:
        mean_p = np.mean(precision_list)
        mean_r = np.mean(recall_list)
        
        print("\n" + "="*40)
        print(f"  EVALUATION RESULTS")
        print("="*40)
        print(f"  Model:      {Path(model_path).name}")
        print(f"  Dataset:    {config['data']['name']} (Val)")
        print(f"  Samples:    {len(precision_list)}")
        print(f"  Threshold:  {config['model']['detection_threshold']}")
        print("-" * 40)
        print(f"  Mean Precision: {mean_p:.4f}")
        print(f"  Mean Recall:    {mean_r:.4f}")
        print("="*40 + "\n")
    else:
        print("\nNo valid samples with ground truth found.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to your YAML config file')
    parser.add_argument('model', type=str, help='Path to the .pth.tar model checkpoint')
    
    args = parser.parse_args()
    main(args.config, args.model)