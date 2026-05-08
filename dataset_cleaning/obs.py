import os
import shutil
import random
from tqdm import tqdm

# --- CONFIGURATION ---
SOURCE_PATH = r"C:\Users\Mithul\Academics\Semester - 4\IoT\Animals_test_split"
DEST_PATH = r"C:\Users\Mithul\Academics\Semester - 4\IoT\OBS_Slideshow"
IMAGES_PER_CLASS = 2


def create_obs_slideshow():
    # Create the destination folder if it doesn't exist
    if not os.path.exists(DEST_PATH):
        os.makedirs(DEST_PATH)
        print(f"Created directory: {DEST_PATH}")
    else:
        # Clear existing images in the slideshow folder to start fresh
        for f in os.listdir(DEST_PATH):
            os.remove(os.path.join(DEST_PATH, f))
        print("Cleared existing slideshow images.")

    # Get all animal classes
    classes = [d for d in os.listdir(SOURCE_PATH) if os.path.isdir(os.path.join(SOURCE_PATH, d))]

    for animal_class in tqdm(classes, desc="Selecting images"):
        class_folder = os.path.join(SOURCE_PATH, animal_class)
        images = [f for f in os.listdir(class_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if len(images) >= IMAGES_PER_CLASS:
            selected = random.sample(images, IMAGES_PER_CLASS)
            for idx, img_name in enumerate(selected):
                src = os.path.join(class_folder, img_name)
                # Rename to include class name so you can track it in the slideshow
                dest_name = f"{animal_class}_{idx}_{img_name}"
                dest = os.path.join(DEST_PATH, dest_name)
                shutil.copy(src, dest)
        else:
            print(f"Warning: {animal_class} only has {len(images)} images.")

    print(f"\nDone! {len(os.listdir(DEST_PATH))} images are ready in {DEST_PATH}")


if __name__ == "__main__":
    create_obs_slideshow()