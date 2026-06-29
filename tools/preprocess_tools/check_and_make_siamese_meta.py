import argparse
from pathlib import Path


IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}


def list_images(folder):
    folder = Path(folder)
    return {p.name: p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS}


def rel_under(path, root):
    return Path(path).relative_to(root).as_posix()


def main():
    parser = argparse.ArgumentParser(
        description='Check Siamese HR/LQ level folders and write gt,lq_a,lq_b meta info.'
    )
    parser.add_argument('--root', default='/home/namlh/data/dataset/train')
    parser.add_argument('--gt-dir', default='HR_sub')
    parser.add_argument('--teacher-dir', default='LR_70ugly_sub')
    parser.add_argument('--student-dir', default='LR_60ugly_sub')
    parser.add_argument(
        '--output',
        default='/home/namlh/data/dataset/meta_info/meta_info_triplet_direct_70_60.txt',
    )
    parser.add_argument('--max-missing', type=int, default=20)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    gt_dir = root / args.gt_dir
    teacher_dir = root / args.teacher_dir
    student_dir = root / args.student_dir
    output = Path(args.output)

    for folder in [gt_dir, teacher_dir, student_dir]:
        if not folder.is_dir():
            raise FileNotFoundError(f'Missing folder: {folder}')

    gt = list_images(gt_dir)
    teacher = list_images(teacher_dir)
    student = list_images(student_dir)

    common = sorted(set(gt) & set(teacher) & set(student))
    missing_teacher = sorted(set(gt) - set(teacher))
    missing_student = sorted(set(gt) - set(student))
    extra_teacher = sorted(set(teacher) - set(gt))
    extra_student = sorted(set(student) - set(gt))

    print(f'Root: {root}')
    print(f'GT images: {len(gt)} ({args.gt_dir})')
    print(f'Teacher images: {len(teacher)} ({args.teacher_dir})')
    print(f'Student images: {len(student)} ({args.student_dir})')
    print(f'Valid triplets: {len(common)}')
    print(f'Missing teacher for GT: {len(missing_teacher)}')
    print(f'Missing student for GT: {len(missing_student)}')
    print(f'Extra teacher images: {len(extra_teacher)}')
    print(f'Extra student images: {len(extra_student)}')

    for label, items in [
        ('missing_teacher', missing_teacher),
        ('missing_student', missing_student),
        ('extra_teacher', extra_teacher),
        ('extra_student', extra_student),
    ]:
        if items:
            shown = ', '.join(items[: args.max_missing])
            suffix = ' ...' if len(items) > args.max_missing else ''
            print(f'{label}: {shown}{suffix}')

    if not common:
        raise RuntimeError('No valid triplets found; meta file was not written.')

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w') as f:
        for name in common:
            f.write(
                f'{rel_under(gt[name], root)}, '
                f'{rel_under(teacher[name], root)}, '
                f'{rel_under(student[name], root)}\n'
            )

    print(f'Meta info written to: {output}')


if __name__ == '__main__':
    main()
