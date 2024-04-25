# Take in the path of the data
import argparse
from nuimages import NuImages
import os
import subprocess
import time

parser = argparse.ArgumentParser(description='Prepare data for YOLO')

parser.add_argument("--input_dir", nargs = 1, metavar = "input_dir", default=None, type = str, help = "Path to the directory of NuImages")
 
parser.add_argument("--version", nargs = 1, metavar = "version", type = str, choices=["v1.0-mini", "v1.0-train", "v1.0-val", "v1.0-test"], help = "Version of the dataset")
 
parser.add_argument("--output_dir", nargs = 1, metavar = "output_dir", type = str, help = "Path to directory to copy images and labels into")

args = parser.parse_args()

# Define constants
IMG_WIDTH = 1600
IMG_HEIGHT = 900
INPUT_DIR = args.input_dir[0]
OUTPUT_DIR = args.output_dir[0]

# Args will be in a list in default
# deactivate lazy loading, will load everything first.
nuim = NuImages(dataroot=args.input_dir[0], version=args.version[0], verbose=True, lazy=False)

# Create labels and images folder
try:
    os.mkdir(f'{OUTPUT_DIR}/images')
    os.mkdir(f'{OUTPUT_DIR}/labels')
except OSError as e:
    print("Folders already existed!")

# Map class to a number (to be used with YOLO)
class_mapper = {}
class_counter = 0

# Use to give copied image coresponding names
img_counter = 0

def normalize_bb(bb, img_width, img_height):
    """
    Normalize a bounding box into YOLO format.

    Args:
        bb: bouding boxes in pixel format.
        img_width: width of original image (not bb).
        img_height: height of original image (not bb).
    
    Returns:
        bb: normalized bounding boxes (0 to 1)
    """
    bb[0], bb[2] = bb[0] / img_width, bb[2] / img_width
    bb[1], bb[3] = bb[1] / img_height, bb[3] / img_height

    return bb

def annotate_img(sample):
    """
    Normalize an image's bounding boxes and move it along with its labels to OUTPUT_DIR.

    Args:
        sample: A sample instance from NuImages        
    """
    global class_counter
    global img_counter
    # first retrieve its details
    detailed_sample = nuim.get('sample_data', sample['key_camera_token'])

    # then load the annotation
    object_tokens, _ = nuim.list_anns(sample['token'], verbose=False)

    subprocess.call(['cp', 
                     f"{INPUT_DIR}/{detailed_sample['filename']}",
                     f'{OUTPUT_DIR}/images/img_{img_counter}'])
    
    with open(f'{OUTPUT_DIR}/labels/img_{img_counter}.txt', 'w') as f:
        for token in object_tokens:
            annotation = nuim.get('object_ann', token)
            bbox = annotation['bbox']
            cat_token = annotation['category_token']
            class_name = nuim.get('category', cat_token)['name']
    
            if class_name not in class_mapper:
                class_mapper[class_name] = class_counter
                class_counter += 1
                    
            # normal bbox has type [class, xcenter, ycenter, width, height]
            bbox_width = bbox[2] - bbox[0]
            bbox_height = bbox[3] - bbox[1]
            center_x, center_y = (bbox[0] + bbox[2]) / 2, (bbox[3] + bbox[1]) /2
            
            # modify the bounding box to match YoloV5 format
            norm_bb = normalize_bb([center_x, center_y, bbox_width, bbox_height], IMG_WIDTH, IMG_HEIGHT)

            f.write(str(class_mapper[class_name]) + ' '  + ' '.join(str(x) for x in norm_bb))
            f.write('\n')
    img_counter += 1

t0 = time.time()
for sample in nuim.sample:
    annotate_img(sample)


with open(f"{OUTPUT_DIR}/classes.txt", 'w') as out:
    for key, value in class_mapper.items():
        out.write(f'{value}: {key}\n')

t1 = time.time()
print(f"Finished. The code took {t1-t0}s")
