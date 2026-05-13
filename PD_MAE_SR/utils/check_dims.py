import cv2
import os
import argparse
from tqdm import tqdm
from collections import Counter

def check_dimensions(input_path):
    if not os.path.exists(input_path):
        print(f"Error: Path {input_path} does not exist.")
        return

    image_files = [f for f in os.listdir(input_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    
    if not image_files:
        print(f"No images found in {input_path}")
        return

    print(f"Checking {len(image_files)} images in {input_path}...")
    
    dims = []
    # Check first 100 images for speed, or all if you prefer
    sample_size = min(100, len(image_files))
    
    for i in tqdm(range(sample_size)):
        img_path = os.path.join(input_path, image_files[i])
        img = cv2.imread(img_path)
        if img is not None:
            h, w = img.shape[:2]
            dims.append(f"{w}x{h}")
        else:
            print(f"Could not read: {img_path}")

    counter = Counter(dims)
    print("\n--- Resolution Summary (Sample of 100) ---")
    for res, count in counter.items():
        print(f"Resolution {res}: {count} images")
    
    if len(counter) == 1:
        print(f"\nConclusion: Your patches are consistently {list(counter.keys())[0]}")
    else:
        print("\nWarning: Mixed resolutions found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to images folder")
    args = parser.parse_args()
    
    check_dimensions(args.input)
