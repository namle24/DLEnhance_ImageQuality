import torch

path = r'd:\Projects\DLEnhance_ImageQuality\BasicSR\experiments\pretrained_models\net_g_550000.pth'
ckpt = torch.load(path, map_location='cpu')

print(f"Type: {type(ckpt)}")
if isinstance(ckpt, dict):
    print(f"Keys: {ckpt.keys()}")
    for k in ckpt.keys():
        if isinstance(ckpt[k], dict):
            print(f"Sub-keys for {k}: {list(ckpt[k].keys())[:5]}... (total {len(ckpt[k])})")
else:
    print("Not a dictionary.")
