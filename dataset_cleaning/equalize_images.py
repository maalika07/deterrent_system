import os
import random
from tqdm import tqdm


DATASET_PATH = r"C:\Users\Mithul\Academics\Semester - 4\IoT\Animals_dataset"
TARGET_COUNT = 351


def equalize_dataset():
    print(f"Equalizing all classes to {TARGET_COUNT} images...")


    classes = [d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))]

    for animal_class in classes:
        class_path = os.path.join(DATASET_PATH, animal_class)

        files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        current_count = len(files)

        if current_count > TARGET_COUNT:
            num_to_delete = current_count - TARGET_COUNT
            print(f"Class '{animal_class}': Found {current_count}. Removing {num_to_delete} random images...")

            to_remove = random.sample(files, num_to_delete)

            for filename in tqdm(to_remove, desc=f"Pruning {animal_class}"):
                file_path = os.path.join(class_path, filename)
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"Error deleting {filename}: {e}")

        elif current_count < TARGET_COUNT:
            print(f"Warning: Class '{animal_class}' only has {current_count} images (under target).")
        else:
            print(f"Class '{animal_class}' already at {TARGET_COUNT}. Skipping.")

    print("\nDataset equalization complete.")


if __name__ == "__main__":
    if os.path.exists(DATASET_PATH):
        equalize_dataset()
    else:
        print("Error: DATASET_PATH not found. Please check the path in the script.")