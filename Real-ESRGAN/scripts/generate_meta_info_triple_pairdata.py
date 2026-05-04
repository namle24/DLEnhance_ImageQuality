import os
import argparse
import glob

def main(args):
    txt_file = open(args.meta_info, 'w')

    img_paths_gt = sorted(glob.glob(os.path.join(args.input[0], '*')))
    img_paths_lq_a = sorted(glob.glob(os.path.join(args.input[1], '*')))
    img_paths_lq_b = sorted(glob.glob(os.path.join(args.input[2], '*')))

    assert len(img_paths_gt) == len(img_paths_lq_a) == len(img_paths_lq_b), \
        f"Folders must have same number of images. Got: {len(img_paths_gt)}, {len(img_paths_lq_a)}, {len(img_paths_lq_b)}"

    for p_gt, p_lq_a, p_lq_b in zip(img_paths_gt, img_paths_lq_a, img_paths_lq_b):
        name_gt = os.path.relpath(p_gt, args.root[0])
        name_lq_a = os.path.relpath(p_lq_a, args.root[1])
        name_lq_b = os.path.relpath(p_lq_b, args.root[2])
        txt_file.write(f'{name_gt}, {name_lq_a}, {name_lq_b}\n')

    print(f"Meta info file written to: {args.meta_info}")
    txt_file.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input',
        nargs='+',
        default=[
            '/datasets/RealESRGAN_data/dataset/train/HR_sub',
            '/datasets/RealESRGAN_data/dataset/train/LR_light_sub',
            '/datasets/RealESRGAN_data/dataset/train/LR_moderate_sub'
        ],
        help='Input folders: [HR, LQ_A, LQ_B]'
    )
    parser.add_argument(
        '--root',
        nargs='+',
        default=[None, None, None],
        help='Root folders corresponding to [HR, LQ_A, LQ_B]'
    )
    parser.add_argument(
        '--meta_info',
        type=str,
        default='/datasets/RealESRGAN_data/meta_info/meta_info_triplet_sub.txt',
        help='Output path for meta info file'
    )
    args = parser.parse_args()

    assert len(args.input) == 3, 'Input must have 3 folders: HR, LQ_A, LQ_B'
    assert len(args.root) == 3, 'Root must have 3 paths'

    os.makedirs(os.path.dirname(args.meta_info), exist_ok=True)

    for i in range(3):
        if args.input[i].endswith('/'):
            args.input[i] = args.input[i][:-1]
        if args.root[i] is None:
            args.root[i] = os.path.dirname(args.input[i])

    main(args)
