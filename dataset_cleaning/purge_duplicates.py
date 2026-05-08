import os
from PIL import Image
import imagehash
from tqdm import tqdm


DATASET_PATH = r"C:\Users\Mithul\Academics\Semester - 4\IoT\Animals_dataset"
THRESHOLD = 2


def purge_duplicates():
    seen_hashes = {}
    deleted_count = 0

    print("Starting permanent deletion of duplicates...")

    for root, dirs, files in os.walk(DATASET_PATH):
        animal_class = os.path.basename(root)

        for filename in tqdm(files, desc=f"Cleaning {animal_class}"):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            file_path = os.path.join(root, filename)

            try:
                with Image.open(file_path) as img:
                    h = imagehash.phash(img)

                    is_duplicate = False
                    for existing_hash in seen_hashes.keys():
                        if h - existing_hash <= THRESHOLD:
                            is_duplicate = True
                            break

                    if is_duplicate:
                        os.remove(file_path)
                        deleted_count += 1
                    else:
                        seen_hashes[h] = file_path

            except Exception as e:
                print(f"\nError processing {filename}: {e}")

    print(f"\nCleanup Complete!")
    print(f"Total images permanently deleted: {deleted_count}")
    print(f"Remaining unique images: {len(seen_hashes)}")


if __name__ == "__main__":
    purge_duplicates()