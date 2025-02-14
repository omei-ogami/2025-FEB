import os
import shutil
import cv2

'''
Run this file if the data hasn't been organized yet
You can delete the raw_images and raw_targets folders after it
'''

def organize_data(root_dir):
    """
    Move pre/post images into their respective folders.
    """
    raw_image_dir = os.path.join(root_dir, "raw_images")
    raw_target_dir = os.path.join(root_dir, "raw_targets")
    
    pre_dir = os.path.join(root_dir, "images", "pre")
    post_dir = os.path.join(root_dir, "images", "post")
    pre_target_dir = os.path.join(root_dir, "targets", "pre")
    post_target_dir = os.path.join(root_dir, "targets", "post")

    # Create target directories if they don’t exist
    os.makedirs(pre_dir, exist_ok=True)
    os.makedirs(post_dir, exist_ok=True)

    # List all pre-disaster images
    pre_files = {f.replace("_pre_disaster.png", ""): f for f in os.listdir(raw_image_dir) if f.endswith("_pre_disaster.png")}

    for prefix, pre_filename in pre_files.items():
        post_filename = f"{prefix}_post_disaster.png"
        pre_target_filename = f"{prefix}_pre_disaster_target.png"
        post_target_filename = f"{prefix}_post_disaster_target.png"

        pre_path = os.path.join(raw_image_dir, pre_filename)
        post_path = os.path.join(raw_image_dir, post_filename)
        pre_target_path = os.path.join(raw_target_dir, pre_target_filename)
        post_target_path = os.path.join(raw_target_dir, post_target_filename)

        # Ensure both post-disaster and target images exist
        if os.path.exists(post_path) and os.path.exists(pre_target_path) and os.path.exists(post_target_path):
            # Move files to their respective directories
            shutil.move(pre_path, os.path.join(pre_dir, pre_filename))
            shutil.move(post_path, os.path.join(post_dir, post_filename))
            shutil.move(pre_target_path, os.path.join(pre_target_dir, pre_target_filename))
            shutil.move(post_target_path, os.path.join(post_target_dir, post_target_filename))
    
    print("Data has been organized into images/pre, images/post, targets/pre, targets/post")

# orgainize data to pre and post folders
organize_data(root_dir = 'data/train')

# get the prefixes and orgainze the target
'''
prefixes = []
image_dir = "data/train/images/pre"
for filename in os.listdir(image_dir):
    if "_pre_disaster.png" in filename:  # Only check pre-disaster images
        prefix = filename.replace("_pre_disaster.png", "")
        prefixes.append(prefix)
prefixes = sorted(list(prefixes))  # Convert set to sorted list
process_all_targets(prefixes, 'data/train/raw_targets', 'data/train/targets')
'''