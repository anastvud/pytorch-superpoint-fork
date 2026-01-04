import torch
import os
from collections import OrderedDict


# Paths
input_path = 'logs/barka_dataset_labels/checkpoints/batchsize1/superPointNet_196000_checkpoint.pth.tar'
output_path = 'logs/barka_dataset_labels/checkpoints/batchsize1/PTH_superPointNet_196000_checkpoint.pth'

#for display with keypoints_on_image_compare_models.py, not suitable for pyslam now
def convert_tar_to_pth():
    print(f"Loading {input_path}...")
    try:
        checkpoint = torch.load(input_path, map_location=torch.device('cpu'))
    except FileNotFoundError:
        print(f"Error: {input_path} not found. Please check the path.")
        return

    # 1. Extract the model weights
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        print("Warning: 'model_state_dict' key not found. Assuming file is already a state dict.")
        state_dict = checkpoint

    # 2. Fix 'module.' prefix (common if trained with DataParallel)
    # If the target code expects keys like 'conv1a.weight' but you have 'module.conv1a.weight'
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '') # remove 'module.'
        new_state_dict[name] = v

    print(f"Saving raw weights to {output_path}...")
    torch.save(new_state_dict, output_path)
    print("Conversion complete! You can now use this file with standard .pth loaders.")


if __name__ == '__main__':
    convert_tar_to_pth()