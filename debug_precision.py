import argparse
import yaml
import torch
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

# Repository imports
from utils.loader import dataLoader
from models.model_wrap import SuperPointFrontend_torch

def draw_debug_image(img_tensor, pred_pts, gt_pts, save_path="debug_eval.png"):
    """
    Saves an image showing Ground Truth (Green) and Predictions (Red).
    """
    # 1. Image Conversion
    img = img_tensor.detach().cpu().numpy().squeeze()
    img = (img * 255).astype(np.uint8)
    if len(img.shape) == 2:
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img_color = img.copy()

    # 2. Draw GT (Green Circles)
    # Expects gt_pts to be (2, M) -> [[x1, x2...], [y1, y2...]]
    if gt_pts is not None and gt_pts.shape[1] > 0:
        for i in range(gt_pts.shape[1]):
            # Cast strictly to python int
            x = int(float(gt_pts[0, i]))
            y = int(float(gt_pts[1, i]))
            cv2.circle(img_color, (x, y), 3, (0, 255, 0), -1)

    # 3. Draw Predictions (Red X)
    # Expects pred_pts to be (3, N) -> [[x1...], [y1...], [prob...]]
    if pred_pts is not None and pred_pts.shape[1] > 0:
        for i in range(pred_pts.shape[1]):
            x = int(float(pred_pts[0, i]))
            y = int(float(pred_pts[1, i]))
            cv2.drawMarker(img_color, (x, y), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=5, thickness=2)

    cv2.imwrite(save_path, img_color)
    print(f"Saved debug visualization to: {save_path}")
    print(f" - Ground Truth points: {gt_pts.shape[1] if gt_pts is not None else 0}")
    print(f" - Predicted points:    {pred_pts.shape[1] if pred_pts is not None else 0}")

def main(config_path, model_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    config['model']['batch_size'] = 1
    config['model']['eval_batch_size'] = 1
    if 'warped_pair' in config['data']:
        config['data']['warped_pair']['enable'] = False
        
    config['model']['detection_threshold'] = 0.005 # Low threshold for debug

    print(f"Loading data from: {config['data']['root']}")
    data = dataLoader(config, dataset='my_dataset', warp_input=False)
    val_loader = data['val_loader']

    print(f"Loading model: {model_path}")
    fe = SuperPointFrontend_torch(
        config=config,
        weights_path=model_path,
        nms_dist=config['model']['nms'],
        conf_thresh=config['model']['detection_threshold'],
        nn_thresh=0.7,
        cuda=True,
        device=device
    )

    print("Running check on first batch...")
    for i, sample in enumerate(val_loader):
        img = sample['image'].to(device)
        
        # 1. Run Model
        prediction = fe.run(img)
        
        # Unwrap tuple/list
        if isinstance(prediction, tuple): prediction = prediction[0]
        if isinstance(prediction, list):  prediction = np.array(prediction)
        
        pred_pts = prediction
        
        # 2. Fix Shapes (CRITICAL FIX)
        # Ensure pred_pts is (3, N). If it came out as (N, 3), transpose it.
        if pred_pts is not None and pred_pts.size > 0:
            if pred_pts.shape[0] != 3 and pred_pts.shape[1] == 3:
                pred_pts = pred_pts.T
        else:
             pred_pts = np.zeros((3, 0))

        # 3. Get Ground Truth
        if 'labels_2D' in sample:
            heatmap = sample['labels_2D'].detach().cpu().numpy().squeeze()
            ys, xs = np.where(heatmap >= 1.0)
            gt_pts = np.stack([xs, ys], axis=0) # Returns (2, M)
        else:
            gt_pts = np.zeros((2, 0))
        
        # 4. Draw & Exit
        draw_debug_image(img, pred_pts, gt_pts)
        break 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('model', type=str)
    args = parser.parse_args()
    main(args.config, args.model)