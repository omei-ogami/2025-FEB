import os
import shutil
import cv2

'''
Run this file if the data hasn't been organized yet
You can delete the raw_images and raw_targets folders after it
'''

def extract_change_mask(pre_path, post_path):
    # Load the target image (assume it's a grayscale mask)
    pre_image = cv2.imread(pre_path, cv2.IMREAD_UNCHANGED)  # Keep original format
    post_image = cv2.imread(post_path, cv2.IMREAD_UNCHANGED)

    # Compute the change mask (absolute difference)
    change_mask = cv2.absdiff(pre_image, post_image)

    return change_mask

def process_all_targets(prefix_lst, target_dir, output_dir):

    for prefix in prefix_lst:
        pre_filename = f"{prefix}_pre_disaster_target.png"
        post_filename = f"{prefix}_post_disaster_target.png"
        pre_path = os.path.join(target_dir, pre_filename)
        post_path = os.path.join(target_dir, post_filename) 

        change_mask = extract_change_mask(pre_path, post_path)

        # Save the change mask to the output directory
        output_filename = f"{prefix}_target.png"
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, change_mask)
    
    print("Data has been organized into /targets")

def organize_data(root_dir):
    """
    Move pre/post images into their respective folders.
    """
    raw_image_dir = os.path.join(root_dir, "raw_images")
    
    pre_dir = os.path.join(root_dir, "images", "pre")
    post_dir = os.path.join(root_dir, "images", "post")

    # Create target directories if they don’t exist
    os.makedirs(pre_dir, exist_ok=True)
    os.makedirs(post_dir, exist_ok=True)

    # List all pre-disaster images
    pre_files = {f.replace("_pre_disaster.png", ""): f for f in os.listdir(raw_image_dir) if f.endswith("_pre_disaster.png")}

    for prefix, pre_filename in pre_files.items():
        post_filename = f"{prefix}_post_disaster.png"

        pre_path = os.path.join(raw_image_dir, pre_filename)
        post_path = os.path.join(raw_image_dir, post_filename)

        # Ensure both post-disaster and target images exist
        if os.path.exists(post_path):
            # Move files to their respective directories
            shutil.move(pre_path, os.path.join(pre_dir, pre_filename))
            shutil.move(post_path, os.path.join(post_dir, post_filename))
    
    print("Data has been organized into images/pre, images/post")

# orgainize data to pre and post folders
organize_data(root_dir = 'data/train')

# get the prefixes and orgainze the target
prefixes = []
image_dir = "data/train/images/pre"
for filename in os.listdir(image_dir):
    if "_pre_disaster.png" in filename:  # Only check pre-disaster images
        prefix = filename.replace("_pre_disaster.png", "")
        prefixes.append(prefix)
prefixes = sorted(list(prefixes))  # Convert set to sorted list
process_all_targets(prefixes, 'data/train/raw_targets', 'data/train/targets')
