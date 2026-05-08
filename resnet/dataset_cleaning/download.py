import os
import shutil
from icrawler.builtin import BingImageCrawler

# --- CONFIGURATION ---
# Set this to your local path
TARGET_DIR = r"C:\Users\Mithul\Academics\Semester - 4\IoT\Animals_dataset\background_2"
TEMP_DIR = "temp_crawl"

# Diverse queries to ensure we hit 351 without exhaustion
# We split 351 across 12 queries (~29-30 images each)
search_queries = [
    "Indian paddy field empty",
    "rural dirt road india countryside",
    "Indian village farm land landscape",
    "dry agricultural field India",
    "lush green rice fields India",
    "Indian countryside gravel path",
    "traditional Indian farm background",
    "empty farmland India",
    "village road trees India",
    "Indian agricultural landscape 4k",
    "unpaved village road India",
    "open farm field India"
]

TOTAL_TARGET = 351
images_per_query = TOTAL_TARGET // len(search_queries)
remainder = TOTAL_TARGET % len(search_queries)

if not os.path.exists(TARGET_DIR):
    os.makedirs(TARGET_DIR)

print(f"Starting crawl for {TOTAL_TARGET} images...")

for i, query in enumerate(search_queries):
    # Add the remainder to the first query
    count = images_per_query + (remainder if i == 0 else 0)

    print(f"\n[{i + 1}/{len(search_queries)}] Crawling {count} images for: {query}")

    # Use a subfolder for each query to avoid naming conflicts during download
    query_temp_dir = os.path.join(TEMP_DIR, f"query_{i}")

    crawler = BingImageCrawler(
        downloader_threads=4,
        storage={'root_dir': query_temp_dir}
    )

    crawler.crawl(keyword=query, max_num=count)

    # Move and rename images to the final directory
    if os.path.exists(query_temp_dir):
        for filename in os.listdir(query_temp_dir):
            source = os.path.join(query_temp_dir, filename)
            # Create a unique filename based on the query index
            new_name = f"rural_bg_{i}_{filename}"
            dest = os.path.join(TARGET_DIR, new_name)

            try:
                shutil.move(source, dest)
            except Exception as e:
                pass

# Cleanup temp folder
if os.path.exists(TEMP_DIR):
    shutil.rmtree(TEMP_DIR)

final_count = len(os.listdir(TARGET_DIR))
print(f"\nProcess Complete!")
print(f"Total images in {TARGET_DIR}: {final_count}")