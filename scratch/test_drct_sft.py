import torch
import sys
import os

# Set up paths to import from DRCT
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
drct_path = os.path.join(proj_root, 'DRCT')
sys.path.insert(0, drct_path)

# DRCT imports basicSR. Make sure BasicSR or sys paths are resolved
sys.path.insert(0, proj_root)
# We remove Real-ESRGAN from sys.path to avoid importing local outdated basicsr package
# sys.path.insert(0, os.path.join(proj_root, 'Real-ESRGAN')) 

# Avoid duplicate registration collision in basicsr ARCH_REGISTRY
from basicsr.utils.registry import ARCH_REGISTRY
for key in ['UNetDiscriminatorSN', 'SRVGGNetCompact', 'DRCT']:
    if key in ARCH_REGISTRY._obj_map:
        ARCH_REGISTRY._obj_map.pop(key)

from drct.archs.DRCT_arch import DRCT

def test_drct_sft():
    print("Testing DRCT with SFT injection...")
    
    # Try to find a local checkpoint for testing, or use dummy
    ckpt_path = os.path.join(proj_root, 'PD_MAE_SR/checkpoints/stage1_smoke_500_v2/pd_mae_s1_iter500.pth')
    if not os.path.exists(ckpt_path):
        ckpt_path = None
        print("No local checkpoint found. Running with uninitialized weights.")
    else:
        print(f"Using checkpoint: {ckpt_path}")
    
    # Initialize model using Real-DRCT settings
    model = DRCT(
        upscale=4,
        in_chans=3,
        img_size=64,
        window_size=16,
        compress_ratio=3,
        squeeze_factor=30,
        conv_scale=0.01,
        overlap_ratio=0.5,
        img_range=1.0,
        depths=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler='pixelshuffle',
        resi_connection='1conv',
        use_sft=True,
        mae_checkpoint=ckpt_path
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Dummy input (Batch=1, Channels=3, H=64, W=64)
    # Output should be (Batch=1, Channels=3, H=256, W=256) since scale=4
    dummy_input = torch.randn(1, 3, 64, 64).to(device)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Device: {device}")
    
    with torch.no_grad():
        output = model(dummy_input)
        print(f"Output shape: {output.shape}")
        
        # Check for correct shape
        assert output.shape == (1, 3, 256, 256), f"Expected shape (1, 3, 256, 256), got {output.shape}"
        
        print("DRCT SFT Smoke test passed successfully!")

if __name__ == "__main__":
    test_drct_sft()
