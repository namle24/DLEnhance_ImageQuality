import os

# Đường dẫn gốc
root_gt = '/datasets/RealESRGAN_data/dataset/train'
root_lq_a = '/datasets/RealESRGAN_data/dataset/train'
root_lq_b = '/datasets/RealESRGAN_data/dataset/train'

meta_info_path = '/datasets/RealESRGAN_data/meta_info_triplet_nosub.txt'

missing_triplets = []

with open(meta_info_path, 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        gt_rel, lq_a_rel, lq_b_rel = [s.strip() for s in line.split(',')]

        gt_path = os.path.join(root_gt, gt_rel)
        lq_a_path = os.path.join(root_lq_a, lq_a_rel)
        lq_b_path = os.path.join(root_lq_b, lq_b_rel)

        missing = []
        if not os.path.isfile(gt_path):
            missing.append('GT')
        if not os.path.isfile(lq_a_path):
            missing.append('LQ_A')
        if not os.path.isfile(lq_b_path):
            missing.append('LQ_B')

        if missing:
            missing_triplets.append((line, missing))

print(f'Tổng số triplet thiếu: {len(missing_triplets)}\n')
for triplet, missing_parts in missing_triplets:
    print(f'Missing {missing_parts} in line: {triplet}')
