import os
import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.archs.swinir_arch import SwinIR
from basicsr.metrics import calculate_psnr, calculate_ssim, calculate_niqe
from realesrgan import RealESRGANer

def calculate_metrics(img_path, gt_path):
    img = cv2.imread(img_path)
    gt = cv2.imread(gt_path)
    
    # Ensure same size for PSNR/SSIM
    if img.shape != gt.shape:
        img = cv2.resize(img, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LANCZOS4)
        
    psnr = calculate_psnr(img, gt, crop_border=4)
    ssim = calculate_ssim(img, gt, crop_border=4)
    
    # NIQE calculation (often requires grayscale or specific implementation)
    # Using basicsr's NIQE which is standard
    niqe_score = calculate_niqe(img, crop_border=4)
    
    return psnr, ssim, niqe_score

def rename_keys(state_dict):
    new_dict = {}
    for k, v in state_dict.items():
        # Map old RRDB keys to new BasicSR keys
        new_k = k.replace('RRDB_trunk', 'body')
        new_k = new_k.replace('.RDB', '.rdb')
        new_k = new_k.replace('trunk_conv', 'conv_body')
        new_k = new_k.replace('upconv1', 'conv_up1')
        new_k = new_k.replace('upconv2', 'conv_up2')
        new_k = new_k.replace('HRconv', 'conv_hr')
        new_dict[new_k] = v
    return new_dict

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    base_data_path = 'BasicSR/datasets/benchmark_datasets'
    
    def fix_path(p):
        # If the user manually included the base folders, strip them
        marker = 'benchmark_datasets'
        if marker in p:
            p = p.split(marker)[-1].lstrip('\\').lstrip('/')
        return os.path.join(base_data_path, p)

    datasets = {
        'Set5': fix_path('Set5/Set5/GTmod12'),
        'Set14': fix_path('Set14/Set14/GTmod12'),
        'BSDS100': fix_path('BSDS100/BSDS100'),
        'Urban100': fix_path('urban100/urban100'),
        'RealSR_V3_Canon': fix_path('RealSR(V3)/Canon/HR'),
        'RealSR_V3_Nikon': fix_path('RealSR(V3)/Nikon/HR')
    }
    
    # Mapping LQ for PSNR calculation
    lq_mapping = {
        'Set5': fix_path('Set5/Set5/LRbicx4'),
        'Set14': fix_path('Set14/Set14/LRbicx4'),
        'BSDS100': fix_path('BSDS100/BSD100_LQ'),
        'Urban100': fix_path('urban100/urban100_LQ'),
        'RealSR_V3_Canon': fix_path('RealSR(V3)/Canon/LR'),
        'RealSR_V3_Nikon': fix_path('RealSR(V3)/Nikon/LR')
    }

    models_info = [
        {
            'name': 'RealSR_JPEG',
            'type': 'rrdb',
            'path': 'BasicSR/experiments/pretrained_models/RealSR_JPEG.pth'
        },
        {
            'name': 'BSRGAN',
            'type': 'rrdb',
            'path': 'BasicSR/experiments/pretrained_models/BSRGAN.pth'
        },
        {
            'name': 'SwinIR',
            'type': 'swin',
            'path': 'BasicSR/experiments/pretrained_models/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth'
        },
        {
            'name': 'Siamese_Phase2 (Ours)',
            'type': 'rrdb',
            'path': 'BasicSR/experiments/pretrained_models/net_g_550000.pth'
        },
        {
            'name': 'Siamese_Phase3 (Ours)',
            'type': 'rrdb',
            'path': 'BasicSR/experiments/pretrained_models/net_g_640000_phase3.pth'
        }
    ]

    results = []
    output_base = 'results/sota_benchmark'
    os.makedirs(output_base, exist_ok=True)

    for m_info in models_info:
        print(f"\nEvaluating Model: {m_info['name']}")
        
        # Load Model
        if m_info['type'] == 'rrdb':
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4).to(device)
            load_net = torch.load(m_info['path'])
            if 'params' in load_net:
                load_net = load_net['params']
            elif 'params_ema' in load_net:
                load_net = load_net['params_ema']
            
            # Rename keys for compatibility
            load_net = rename_keys(load_net)
            model.load_state_dict(load_net, strict=True)
            model.eval()
        elif m_info['type'] == 'swin':
            model = SwinIR(upscale=4, in_chans=3, img_size=64, window_size=8, img_range=1., depths=[6, 6, 6, 6, 6, 6],
                           embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv').to(device)
            load_net = torch.load(m_info['path'])
            if 'params' in load_net:
                load_net = load_net['params']
            elif 'params_ema' in load_net:
                load_net = load_net['params_ema']
            
            # Rename keys for compatibility
            load_net = rename_keys(load_net)
            model.load_state_dict(load_net, strict=True)
            model.eval()
        elif m_info['type'] == 'realesrgan':
            upsampler = RealESRGANer(scale=4, model_path=None, model=None, tile=0, tile_pad=10, pre_pad=0, half=True)
            # This downloads weights automatically if model_path is None

        for ds_name, gt_folder in datasets.items():
            print(f" Processing Dataset: {ds_name}")
            lq_folder = lq_mapping[ds_name]
            ds_output = os.path.join(output_base, m_info['name'], ds_name)
            os.makedirs(ds_output, exist_ok=True)
            
            lq_images = sorted(os.listdir(lq_folder))
            ds_metrics = []

            for img_file in tqdm(lq_images):
                lq_path = os.path.join(lq_folder, img_file)
                gt_path = os.path.join(gt_folder, img_file.replace('_LRbicx4', '').replace('x4', '')) # Simple heuristic
                
                # Try to find matching GT if filename differs
                if not os.path.exists(gt_path):
                    # Try original name
                    potential_gt = os.path.join(gt_folder, img_file.split('.')[0].replace('x4', '') + '.png')
                    if os.path.exists(potential_gt): gt_path = potential_gt

                if not os.path.exists(gt_path): continue

                # Inference
                img_lq = cv2.imread(lq_path)
                
                # Timing setup
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                
                if m_info['type'] == 'realesrgan':
                    start_event.record()
                    output, _ = upsampler.enhance(img_lq, outscale=4)
                    end_event.record()
                else:
                    img_input = torch.from_numpy(np.transpose(img_lq[:, :, [2, 1, 0]], (2, 0, 1))).float().divide(255.).unsqueeze(0).to(device)
                    with torch.no_grad():
                        start_event.record()
                        # SwinIR padding
                        if m_info['type'] == 'swin':
                            window_size = 8
                            _, _, h, w = img_input.size()
                            mod_pad_h = (window_size - h % window_size) % window_size
                            mod_pad_w = (window_size - w % window_size) % window_size
                            img_input = torch.nn.functional.pad(img_input, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
                            output_tensor = model(img_input)
                            output_tensor = output_tensor[:, :, 0:h*4, 0:w*4]
                        else:
                            output_tensor = model(img_input)
                        end_event.record()
                        
                        torch.cuda.synchronize()
                        inf_time = start_event.elapsed_time(end_event) / 1000.0 # to seconds
                        
                        output = output_tensor.data.squeeze().float().cpu().clamp_(0, 1).numpy()
                        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
                        output = (output * 255.0).round().astype(np.uint8)
                
                if m_info['type'] == 'realesrgan':
                    torch.cuda.synchronize()
                    inf_time = start_event.elapsed_time(end_event) / 1000.0

                out_path = os.path.join(ds_output, img_file)
                cv2.imwrite(out_path, output)
                
                # Metrics
                psnr, ssim, niqe_val = calculate_metrics(out_path, gt_path)
                ds_metrics.append({
                    'psnr': psnr, 
                    'ssim': ssim, 
                    'niqe': niqe_val,
                    'time': inf_time
                })

            # Averaging
            if ds_metrics:
                avg_psnr = np.mean([x['psnr'] for x in ds_metrics])
                avg_ssim = np.mean([x['ssim'] for x in ds_metrics])
                avg_niqe = np.mean([x['niqe'] for x in ds_metrics])
                avg_time = np.mean([x['time'] for x in ds_metrics])
                results.append({
                    'Model': m_info['name'],
                    'Dataset': ds_name,
                    'PSNR': avg_psnr,
                    'SSIM': avg_ssim,
                    'NIQE': avg_niqe,
                    'Time (s)': avg_time
                })

    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv('results/sota_final_benchmark_results.csv', index=False)
    print("\nBenchmark Finished! Results saved to results/sota_final_benchmark_results.csv")

if __name__ == '__main__':
    main()
