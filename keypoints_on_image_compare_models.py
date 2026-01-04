import torch
import torch.nn as nn
import numpy as np
import cv2
import os
import argparse


class DoubleConv(nn.Module):
    """(Conv => BN => ReLU) * 2"""
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class InConv(nn.Module):
    """Wrapper for the first layer to match 'inc.conv.conv' key structure"""
    def __init__(self, in_ch, out_ch):
        super(InConv, self).__init__()
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(x)

class Down(nn.Module):
    """Wrapper for Downsampling layers to match 'downX.mpconv' key structure"""
    def __init__(self, in_ch, out_ch):
        super(Down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        return self.mpconv(x)

class SuperPointBN(nn.Module):
    """
    Custom SuperPoint Architecture with Batch Norm.
    Matches 'logs/.../checkpoints/*.pth.tar'.
    """
    def __init__(self):
        super(SuperPointBN, self).__init__()
        
        # Encoder (U-Net style wrappers)
        self.inc = InConv(1, 64)
        self.down1 = Down(64, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 128)

        # Detector Head (Explicit BN layers based on 'bnPa' keys)
        self.convPa = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bnPa = nn.BatchNorm2d(256)
        self.convPb = nn.Conv2d(256, 65, kernel_size=1, stride=1, padding=0)
        self.bnPb = nn.BatchNorm2d(65)
        
        # Descriptor Head
        self.convDa = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bnDa = nn.BatchNorm2d(256)
        self.convDb = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.bnDb = nn.BatchNorm2d(256)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        # Detector Head
        cPa = self.relu(self.bnPa(self.convPa(x4)))
        semi = self.bnPb(self.convPb(cPa)) # BN applied to output scores
        
        # Descriptor Head
        cDa = self.relu(self.bnDa(self.convDa(x4)))
        desc = self.bnDb(self.convDb(cDa))
        
        dn = torch.norm(desc, p=2, dim=1)
        desc = desc.div(torch.unsqueeze(dn, 1))
        return semi, desc
    

class SuperPointStandard(nn.Module):
    """
    Standard SuperPoint Architecture (VGG-style).
    Matches 'pretrained/superpoint_v1.pth'.
    """
    def __init__(self):
        super(SuperPointStandard, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Shared Encoder
        self.conv1a = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        
        self.conv2a = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        
        self.conv3a = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        
        self.conv4a = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        
        # Detector Head
        self.convPa = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.convPb = nn.Conv2d(256, 65, kernel_size=1, stride=1, padding=0)
        
        # Descriptor Head
        self.convDa = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # Encoder
        x = self.relu(self.conv1a(x))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        
        # Detector
        cPa = self.relu(self.convPa(x))
        semi = self.convPb(cPa)
        
        # Descriptor
        cDa = self.relu(self.convDa(x))
        desc = self.convDb(cDa)
        
        dn = torch.norm(desc, p=2, dim=1)
        desc = desc.div(torch.unsqueeze(dn, 1))
        return semi, desc
    




def load_weights_and_inspect(model, weight_path, device='cpu'):
    """
    Loads weights and prints training metadata (loss, epoch, etc.) if available.
    """
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Weight file not found: {weight_path}")

    print(f"\n--- Loading: {os.path.basename(weight_path)} ---")
    checkpoint = torch.load(weight_path, map_location=device)
    
    state_dict = None
    
    # Check if it is a checkpoint dictionary or just weights
    if isinstance(checkpoint, dict):
        # 1. Inspect and Print Metadata
        print(">> Checkpoint Parameters found:")
        found_meta = False
        for key, value in checkpoint.items():
            # Skip the huge tensor dictionaries
            if key in ['state_dict', 'model_state_dict', 'optimizer', 'optimizer_state_dict']:
                continue
            
            # Print simple scalar values (metrics, epoch, config)
            if isinstance(value, (int, float, str, bool)):
                print(f"   - {key}: {value}")
                found_meta = True
            elif isinstance(value, torch.Tensor) and value.numel() == 1:
                print(f"   - {key}: {value.item():.5f}")
                found_meta = True
        
        if not found_meta:
            print("   (No scalar metrics found in this checkpoint)")

        # 2. Extract State Dict
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            # Fallback: maybe the dict itself is the state_dict
            state_dict = checkpoint
    else:
        # It's likely just the state_dict directly
        print(">> No metadata dictionary found (likely just a pure weight file).")
        state_dict = checkpoint

    # 3. Clean Keys and Load
    clean_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("module.", "") # Remove DataParallel prefix
        clean_state_dict[name] = v
        
    try:
        model.load_state_dict(clean_state_dict)
        print(">> Weights loaded successfully.")
    except RuntimeError as e:
        print(f"\n[ERROR] Architecture Mismatch for {weight_path}!")
        print("Please ensure you are loading the correct model class (Standard vs Custom).")
        raise e
        
    model.to(device)
    model.eval()
    return model

def process_image(img_path, device='cpu'):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    inp = gray.astype(np.float32) / 255.0
    inp_tensor = torch.from_numpy(inp).unsqueeze(0).unsqueeze(0).to(device)
    return img, inp_tensor, H, W

def run_inference(model, inp_tensor, conf_thresh=0.015, nms_dist=4):
    with torch.no_grad():
        semi, _ = model(inp_tensor)
        
    dense = torch.nn.functional.softmax(semi, dim=1)
    nodust = dense[0, :-1, :, :]
    
    heatmap = nodust.permute(1, 2, 0) # [H/8, W/8, 64]
    H_new = heatmap.shape[0] * 8
    W_new = heatmap.shape[1] * 8
    heatmap = heatmap.reshape(heatmap.shape[0], heatmap.shape[1], 8, 8)
    heatmap = heatmap.permute(0, 2, 1, 3).reshape(H_new, W_new)
    heatmap = heatmap.cpu().numpy()
    
    pts = np.where(heatmap >= conf_thresh)
    ys, xs = pts[0], pts[1]
    confidences = heatmap[ys, xs]
    
    if len(xs) == 0: return []

    pts_raw = np.zeros((3, len(xs)))
    pts_raw[0, :] = xs
    pts_raw[1, :] = ys
    pts_raw[2, :] = confidences
    
    # Simple NMS
    grid = np.zeros((H_new, W_new)).astype(int)
    grid_pad = np.pad(grid, (nms_dist, nms_dist), mode='constant')
    inds = np.argsort(pts_raw[2,:])[::-1]
    pts_nms = []
    for i in inds:
        x, y, conf = int(pts_raw[0, i]), int(pts_raw[1, i]), pts_raw[2, i]
        if grid_pad[y + nms_dist, x + nms_dist] == 0:
            grid_pad[y:y+2*nms_dist+1, x:x+2*nms_dist+1] = 1
            pts_nms.append([x, y, conf])
    return pts_nms

def paint_keypoints(img, keypoints, color, label):
    """
    Draws keypoints and handles multi-line labels.
    Expected label format: "Title\nSubtitle" or just "Title"
    """
    out = img.copy()
    for kp in keypoints:
        cv2.circle(out, (int(kp[0]), int(kp[1])), 3, color, -1, lineType=cv2.LINE_AA)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6  
    thickness = 2
    
    # Split by newline to support 2 lines of text
    lines = label.split('\n')
    y_pos = 30
    
    # Line 1: The main label + count
    text_line1 = f"{lines[0]}: {len(keypoints)} kps"
    cv2.putText(out, text_line1, (20, y_pos), font, font_scale, (0,0,0), thickness)
    
    # Line 2: The model name (if provided)
    if len(lines) > 1:
        y_pos += 25
        text_line2 = lines[1]
        cv2.putText(out, text_line2, (20, y_pos), font, font_scale, color, thickness)
        
    return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', type=str, required=True)
    parser.add_argument('--model_pth', type=str, required=True, help="Path to standard .pth")
    parser.add_argument('--model_tar', type=str, required=True, help="Path to custom .pth.tar")
    parser.add_argument('--out', type=str, default='comparison.png')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Instantiate both architectures
    net_pth = SuperPointStandard()
    net_tar = SuperPointBN()

    # 2. Load and Inspect
    # This will now print the metadata (epoch, loss, etc.) to the console
    net_pth = load_weights_and_inspect(net_pth, args.model_pth, device)
    net_tar = load_weights_and_inspect(net_tar, args.model_tar, device)

    # 3. Process
    original_img, inp_tensor, H, W = process_image(args.img, device)
    
    # Run Inference
    kps_pth = run_inference(net_pth, inp_tensor)
    kps_tar = run_inference(net_tar, inp_tensor, conf_thresh=0.015, nms_dist=4)
    
    # 4. Visualize
    model_tar_name = os.path.basename(args.model_tar)
    
    img_pth_viz = paint_keypoints(original_img, kps_pth, (0, 255, 0), "superpoint_v1")
    img_tar_viz = paint_keypoints(original_img, kps_tar, (0, 0, 255), f"{model_tar_name}")
    
    combined = np.hstack((img_pth_viz, img_tar_viz))
    cv2.imwrite(args.out, combined)
    print(f"\nSaved comparison to {args.out}")