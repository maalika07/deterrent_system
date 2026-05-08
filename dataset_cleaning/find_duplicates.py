import os
import shutil
from PIL import Image
import imagehash
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict

DATASET_PATH = r"C:\Users\Mithul\Academics\Semester - 4\IoT\Animals_dataset"
REPORT_PATH = "duplicate_reports"
DELETED_PATH = "trash_bin"
THRESHOLD = 2


def generate_visual_report():
    if not os.path.exists(REPORT_PATH): os.makedirs(REPORT_PATH)

    hash_groups = defaultdict(list)

    print("Step 1: Hashing images...")
    for root, dirs, files in os.walk(DATASET_PATH):
        animal_class = os.path.basename(root)
        for filename in tqdm(files, desc=f"Hashing {animal_class}"):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')): continue

            file_path = os.path.join(root, filename)
            try:
                with Image.open(file_path) as img:
                    h = imagehash.phash(img)

                    matched = False
                    for existing_hash in hash_groups.keys():
                        if h - existing_hash <= THRESHOLD:
                            hash_groups[existing_hash].append(file_path)
                            matched = True
                            break
                    if not matched:
                        hash_groups[h].append(file_path)
            except Exception as e:
                print(f"Error: {e}")

    print("\nStep 2: Generating visual reports...")
    report_idx = 0
    for h, paths in hash_groups.items():
        if len(paths) > 1:
            report_idx += 1
            original = paths[0]
            dupes = paths[1:]

            num_plots = min(len(paths), 4)
            fig, axes = plt.subplots(1, num_plots, figsize=(15, 5))
            if num_plots == 1: axes = [axes]  # Handle single plot case

            axes[0].imshow(Image.open(original))
            axes[0].set_title(f"ORIGINAL\n{os.path.basename(original)}", color='green', fontsize=8)
            axes[0].axis('off')

            for i in range(1, num_plots):
                axes[i].imshow(Image.open(paths[i]))
                axes[i].set_title(f"DUPLICATE\n{os.path.basename(paths[i])}", color='red', fontsize=8)
                axes[i].axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(REPORT_PATH, f"group_{report_idx}.jpg"))
            plt.close()

    print(f"\nDone! Check the '{REPORT_PATH}' folder to see the side-by-side comparisons.")


if __name__ == "__main__":
    generate_visual_report()