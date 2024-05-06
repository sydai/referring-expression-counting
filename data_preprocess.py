import json
import os
import shutil
from tqdm import tqdm

"""
Assuming the following directory structure (refer to README.md for more details):
- <DATA_ROOT>/
    - Crowd Surveillance/images_train/images/train
    - mall_dataset/frames
    - DETRAC/DETRAC-train-data/Insight-MVT_Annotation_Train
    - FSC147_384_V2/images_384_VarV2
    - CARPK_devkit/data/Images
    - NWPU/images
    - internet
    - VisDrone2020-CC/sequences
    - jhu_crowd_v2.0/test/images
"""

DATA_ROOT = "/mnt/workstation/images" # Update this to your data root

# Create "rec-8k" images folder in the DATA_ROOT
os.makedirs(os.path.join(DATA_ROOT, "rec-8k"), exist_ok=True)

prefix_dict = {
    'fsc147': 'FSC147_384_V2/images_384_VarV2',
    'nwpu': 'NWPU/images',
    'internet': 'internet',
    'jhuv2': 'jhu_crowd_v2.0/test/images',
    "cs": 'Crowd_Surveillance/images_train/images/train',
    'detrac': 'DETRAC/DETRAC-train-data/Insight-MVT_Annotation_Train',
    'mall': 'mall_dataset/frames',
    'carpk': 'CARPK_devkit/data/Images',
    'visdrone': 'VisDrone2020-CC/sequences',
}

anno_file = "anno/annotations.json"

with open(anno_file, 'r') as f:
    anno = json.load(f)

err = False
for img in tqdm(anno.keys()):
    
    new_prefix = img.split("-")[1]
    org_prefix = prefix_dict[new_prefix]


    org_img = "/".join(img.split("-")[2:])
    
    org_img_file = os.path.join(DATA_ROOT, org_prefix, org_img) 
    new_img_file = os.path.join(DATA_ROOT, "rec-8k", img)
    
    # copy to new name
    try:
        # print(f"cp {org_img_file} {new_img_file}")
        shutil.copy(org_img_file, new_img_file)
    except Exception as e:
        err = True
        print(e)

if err:
    print("\nSome images are not copied. Please check the error messages.")
else:
    print("\nDone! Please check the 'rec-8k' folder for a total of 8011 images.")