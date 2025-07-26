import os
import re
import shutil


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def process_directory(base_dir, subdir):
    input_path = os.path.join(base_dir, subdir)
    if not os.path.isdir(input_path):
        print(f"❌ Directory not found: {input_path}")
        return

    print(f"\n📂 Processing folder: {input_path}")

    # Tạo thư mục con HR và LRx
    hr_path = os.path.join(input_path, 'HR')
    lr_path = os.path.join(input_path, f'LRx{subdir}')

    ensure_dir(hr_path)
    ensure_dir(lr_path)

    for filename in os.listdir(input_path):
        file_path = os.path.join(input_path, filename)
        if os.path.isfile(file_path):
            name, ext = os.path.splitext(filename)

            if name.endswith('_HR'):
                new_name = name.replace('_HR', '') + ext
                target_path = os.path.join(hr_path, new_name)
                shutil.move(file_path, target_path)
                print(f"✅ Moved HR: {filename} → HR/{new_name}")

            elif re.search(rf'_LR{subdir}$', name):
                new_name = re.sub(rf'_LR{subdir}$', '', name) + ext
                target_path = os.path.join(lr_path, new_name)
                shutil.move(file_path, target_path)
                print(f"✅ Moved LR{subdir}: {filename} → LRx{subdir}/{new_name}")

            else:
                print(f"⚠️ Skipped (not HR or LR{subdir}): {filename}")


def main():
    base_dir = r'D:\EnhanceVideo_ImageDLM\data\RealSR(V3)\Nikon\Train'
    subdirs = ['2', '3', '4']  # xử lý LR2, LR3, LR4

    for subdir in subdirs:
        process_directory(base_dir, subdir)


if __name__ == '__main__':
    main()
