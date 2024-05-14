from nuimages import NuImages
import shutil

# Annotate an image
limit = 10
IMG_WIDTH = 1600
IMG_HEIGHT = 900

class_mapper = {}

class_counter = 0

img_counter = 0

nuim = NuImages(dataroot='/data/cmpe258-sp24/nuimages', version='v1.0-train', verbose=True, lazy=True)

def normalize_bb(bb, img_width, img_height):
    bb[0], bb[2] = bb[0] / img_width, bb[2] / img_width
    bb[1], bb[3] = bb[1] / img_height, bb[3] / img_height

    return bb

def annotate_img(sample):
    global class_counter
    global img_counter
    # first retrieve its details
    detailed_sample = nuim.get('sample_data', sample['key_camera_token'])

    # then load the annotation
    object_tokens, _ = nuim.list_anns(sample['token'], verbose=False)
        
    # all the filename except the extension
    fname = detailed_sample['filename']
    shutil.copy(f'/data/cmpe258-sp24/nuimages/{fname}',
            f'/data/cmpe258-sp24/nuimages/images/img_{img_counter}.jpg')
    
    with open(f'/data/cmpe258-sp24/nuimages/labels/img_{img_counter}.txt', 'w') as f:
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



for sample in nuim.sample:
	annotate_img(sample)

with open('/data/cmpe258-sp24/nuimages/classes.txt', 'w') as outfile:
	for key,value in class_mapper.items():
		outfile.write(f'{value}: {key}\n')


