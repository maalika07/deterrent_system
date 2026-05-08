import os
import shutil
import random

DATASET_PATH = r"C:\Users\Mithul\Academics\Semester - 4\IoT\Animals_dataset"
TEST_PATH = r"C:\Users\Mithul\Academics\Semester - 4\IoT\Animals_test_split"
TEST_RATIO = 0.1  # 10% for the final unseen test

if not os.path.exists(TEST_PATH): os.makedirs(TEST_PATH)

for cls in os.listdir(DATASET_PATH):
    cls_src = os.path.join(DATASET_PATH, cls)
    if not os.path.isdir(cls_src): continue

    cls_dest = os.path.join(TEST_PATH, cls)
    os.makedirs(cls_dest, exist_ok=True)

    files = [f for f in os.listdir(cls_src) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    test_files = random.sample(files, int(len(files) * TEST_RATIO))

    for f in test_files:
        shutil.move(os.path.join(cls_src, f), os.path.join(cls_dest, f))

print(f"Moved images to {TEST_PATH}. Now re-run your metrics script pointing to THIS folder.")