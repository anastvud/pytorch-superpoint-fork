import torch
import os
import argparse
import sys

def get_scalar_value(v):
    """Helper to convert 1-element tensors to python scalars"""
    if isinstance(v, torch.Tensor):
        if v.numel() == 1:
            return v.item()
        return "Tensor(Multi-dim)"
    return v

def scan_checkpoints(directory):
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        return

    # Filter for .pth.tar files
    files = [f for f in os.listdir(directory) if f.endswith('.pth.tar')]
    
    if not files:
        print(f"No .pth.tar files found in {directory}")
        return

    # Sort files to keep output organized (e.g., by iteration number if in filename)
    files.sort(key=lambda x: [int(c) if c.isdigit() else c for c in __import__('re').split(r'(\d+)', x)])

    print(f"\nScanning {len(files)} files in: {directory}\n")
    
    # Define Column Headers
    header = f"{'Filename':<50} | {'Loss':<20} | {'Epoch':<10} | {'Precision':<10}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    for filename in files:
        filepath = os.path.join(directory, filename)
        
        try:
            # Load only the dictionary, map to CPU to save GPU memory
            checkpoint = torch.load(filepath, map_location='cpu')
            
            if not isinstance(checkpoint, dict):
                print(f"{filename:<50} | {'Not a dict/checkpoint':<20} | {'-':<10} | {'-':<10}")
                continue

            # --- Extract Loss ---
            # Try common keys for loss
            loss_val = "N/A"
            for key in ['loss', 'total_loss', 'train_loss', 'valid_loss']:
                if key in checkpoint:
                    loss_val = get_scalar_value(checkpoint[key])
                    # formatting float if possible
                    if isinstance(loss_val, float):
                        loss_val = f"{loss_val:.6f}"
                    break
            
            # --- Extract Epoch/Iter ---
            epoch_val = "N/A"
            for key in ['epoch', 'n_iter', 'iter', 'iteration']:
                if key in checkpoint:
                    epoch_val = get_scalar_value(checkpoint[key])
                    break

            # --- Extract Precision (Optional, but useful) ---
            prec_val = "-"
            for key in ['precision', 'valid_precision', 'acc', 'accuracy']:
                if key in checkpoint:
                    raw_p = get_scalar_value(checkpoint[key])
                    if isinstance(raw_p, float):
                        prec_val = f"{raw_p:.4f}"
                    else:
                        prec_val = str(raw_p)
                    break

            print(f"{filename:<50} | {str(loss_val):<20} | {str(epoch_val):<10} | {str(prec_val):<10}")

        except Exception as e:
            # Handle corrupted files or load errors
            print(f"{filename:<50} | {'[Error loading file]':<20} | {'-':<10} | {'-':<10}")

    print("-" * len(header))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scan directory for .pth.tar models and print loss.")
    parser.add_argument('--dir', type=str, required=True, help="Path to the directory containing checkpoints")
    
    args = parser.parse_args()
    
    scan_checkpoints(args.dir)