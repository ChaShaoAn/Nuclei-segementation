# %%

from os import listdir
from os.path import isfile, isdir, join
import shutil
import cv2
import numpy as np
import pycocotools.mask as mask
from tqdm import tqdm
from pycocotools.coco import COCO
import base64
from detectron2.structures import BoxMode
import os
from PIL import Image  # (pip install Pillow)
import numpy as np                                 # (pip install numpy)
from skimage import measure                        # (pip install scikit-image)
from shapely.geometry import Polygon, MultiPolygon  # (pip install Shapely)
import numpy
import json


def decode_base64_rles(coco):
    for ann in coco.dataset['annotations']:
        segm = ann['segmentation']
        if type(segm) != list and type(segm['counts']) != list:
            segm['counts'] = base64.b64decode(segm['counts'])


inpath = "./dataset/train/"  # the train folder download from kaggle
outpath = "./train/"  # the folder putting all nuclei image

images_name = listdir(inpath)
cocoformat = {"licenses": [], "info": [], "images": [], "annotations": [],
              "categories": []}


# %%

cat = {"id": 1,
       "name": "nucleus",
       "supercategory": "nucleus",
       }
cocoformat["categories"].append(cat)
# %%

mask_id = 1
is_crowd = 0
category_id = 1
for i, im_name in enumerate(images_name):
    t_image = cv2.imread(inpath + im_name + "/images/" + im_name + ".png")
    mask_folder = listdir(inpath + im_name + "/masks/")
    im = {"id": int(i+1),
          "width": int(t_image.shape[1]),
          "height": int(t_image.shape[0]),
          "file_name": im_name + ".png",
          }
    cocoformat["images"].append(im)
    for mask_img in tqdm(mask_folder):
        t_image = Image.open(inpath + im_name + "/masks/" + mask_img)
        ground_truth_binary_mask = numpy.array(t_image)
        fortran_ground_truth_binary_mask = np.asfortranarray(
                                            ground_truth_binary_mask)
        encoded_ground_truth = mask.encode(fortran_ground_truth_binary_mask)
        ground_truth_area = mask.area(encoded_ground_truth)
        ground_truth_bounding_box = mask.toBbox(encoded_ground_truth)
        contours = measure.find_contours(ground_truth_binary_mask, 0.5)

        annotation = {
                "segmentation": [],
                "area": ground_truth_area.tolist(),
                "iscrowd": 0,
                "image_id": int(i+1),
                "bbox": ground_truth_bounding_box.tolist(),
                "bbox_mode": BoxMode.XYWH_ABS,
                "category_id": 1,
                "id": mask_id
        }

        for contour in contours:
            contour = np.flip(contour, axis=1)
            segmentation = contour.ravel().tolist()
            annotation["segmentation"].append(segmentation)
            # cocoformat["annotations"].append(segmentation)

        cocoformat["annotations"].append(annotation)

        mask_id += 1

# %%
with open("nucleus_cocoformat_poly2.json", "w") as f:
    json.dump(cocoformat, f)

json_obj = json.dumps(cocoformat, indent=4)
with open("test_poly2.json", "w") as outfile:
    outfile.write(json_obj)

# %%
# copy image to another folder
os.makedirs(outpath, exist_ok=True)
for f in images_name:
    image = listdir(inpath + f + "/images/")
    shutil.copyfile(inpath + f + "/images/" + image[0], outpath + image[0])
