import torch
import sys
import os

# Ensure the correct paths are set up
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../Real-ESRGAN')))

from basicsr.archs.rrdbnet_arch import RRDBNet

def test_rrdbnet_sft():
    print("Testing RRDBNet with SFT injection...")
    
    # Checkpoint path
    ckpt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../PD_MAE_SR/checkpoints/stage1_smoke_500_v2/pd_mae_s1_iter500.pth'))
    
    print(f"Using checkpoint: {ckpt_path}")
    
    # Initialize model
    model = RRDBNet(num_in_ch=3, num_out_ch=3, scale=4, num_feat=64, num_block=23, num_grow_ch=32, use_sft=True, mae_checkpoint=ckpt_path)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Dummy input (Batch=1, Channels=3, H=64, W=64)
    # Output should be (Batch=1, Channels=3, H=256, W=256) since scale=4
    dummy_input = torch.randn(1, 3, 64, 64).to(device)
    
    print(f"Input shape: {dummy_input.shape}")
    
    with torch.no_grad():
        output = model(dummy_input)
        print(f"Output shape: {output.shape}")
        
        # Check for correct shape
        assert output.shape == (1, 3, 256, 256), f"Expected shape (1, 3, 256, 256), got {output.shape}"
        
        print("Smoke test passed successfully!")

if __name__ == "__main__":
    test_rrdbnet_sft()
