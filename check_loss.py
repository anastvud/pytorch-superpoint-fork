import torch
import os
import argparse
import sys
import matplotlib.pyplot as plt

def get_scalar_value(v):
    """Helper to convert 1-element tensors to python scalars"""
    if isinstance(v, torch.Tensor):
        if v.numel() == 1:
            return v.item()
        return None # Return None for multi-dim tensors to avoid plotting errors
    return v

def scan_checkpoints(directory, save_path=None):
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        return

    # Filter for .pth.tar files
    files = [f for f in os.listdir(directory) if f.endswith('.pth.tar')]
    
    if not files:
        print(f"No .pth.tar files found in {directory}")
        return

    # Sort files naturally
    files.sort(key=lambda x: [int(c) if c.isdigit() else c for c in __import__('re').split(r'(\d+)', x)])

    print(f"\nScanning {len(files)} files in: {directory}\n")
    
    # Define Column Headers
    header = f"{'Filename':<50} | {'Loss':<20} | {'Epoch':<10} | {'Precision':<10}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    # List to store data for plotting: [(epoch, loss), ...]
    plot_data = []

    for filename in files:
        filepath = os.path.join(directory, filename)
        
        try:
            # Load only the dictionary, map to CPU
            checkpoint = torch.load(filepath, map_location='cpu')
            
            if not isinstance(checkpoint, dict):
                print(f"{filename:<50} | {'Not a dict':<20} | {'-':<10} | {'-':<10}")
                continue

            # --- Extract Loss ---
            raw_loss = None
            for key in ['loss', 'total_loss', 'train_loss', 'valid_loss']:
                if key in checkpoint:
                    raw_loss = get_scalar_value(checkpoint[key])
                    break
            
            # Formatting for print
            loss_str = f"{raw_loss:.6f}" if isinstance(raw_loss, (int, float)) else "N/A"

            # --- Extract Epoch/Iter ---
            raw_epoch = None
            for key in ['epoch', 'n_iter', 'iter', 'iteration']:
                if key in checkpoint:
                    raw_epoch = get_scalar_value(checkpoint[key])
                    break
            
            # Formatting for print
            epoch_str = str(raw_epoch) if raw_epoch is not None else "N/A"

            # --- Extract Precision ---
            prec_str = "-"
            for key in ['precision', 'valid_precision', 'acc', 'accuracy']:
                if key in checkpoint:
                    raw_p = get_scalar_value(checkpoint[key])
                    if isinstance(raw_p, (int, float)):
                        prec_str = f"{raw_p:.4f}"
                    break

            print(f"{filename:<50} | {loss_str:<20} | {epoch_str:<10} | {prec_str:<10}")

            # --- Collect Data for Plotting ---
            # Only add if we have valid numbers for both Epoch and Loss
            if isinstance(raw_loss, (int, float)) and isinstance(raw_epoch, (int, float)):
                plot_data.append((raw_epoch, raw_loss))

        except Exception as e:
            print(f"{filename:<50} | {'[Error loading]':<20} | {'-':<10} | {'-':<10}")

    print("-" * len(header))

    # --- Plotting Logic ---
    if not plot_data:
        print("\nNo valid numeric data found to plot.")
        return

    # Sort data by epoch (x-axis) so the line draws correctly
    plot_data.sort(key=lambda x: x[0])
    
    # Unzip into two lists
    epochs, losses = zip(*plot_data)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses, marker='o', linestyle='-', color='b', label='Loss')
    
    plt.title(f'Training Loss over Epochs\n({directory})')
    plt.xlabel('Epoch / Iteration')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
        print(f"\nPlot saved to: {save_path}")
    else:
        print("\nDisplaying plot...")
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scan directory for .pth.tar models, print table, and plot loss.")
    parser.add_argument('--dir', type=str, required=True, help="Path to the directory containing checkpoints")
    parser.add_argument('--save', type=str, default=None, help="Optional: Path to save the plot image (e.g., loss_plot.png)")
    
    args = parser.parse_args()
    
    scan_checkpoints(args.dir, args.save)