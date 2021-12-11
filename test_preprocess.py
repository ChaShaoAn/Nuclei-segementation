import torchvision.transforms as transforms
import numpy as np
import cv2
import os
# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.modeling import build_model
import json
from detectron2.utils.visualizer import ColorMode
import pycocotools.mask as mask_util
from os import listdir
from tqdm import tqdm
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from PIL import Image

f = open('dataset/test_img_ids.json')
data = json.load(f)
f.close()
tmp_annotation = []
output_path = 'dataset/test_seg/'
os.makedirs(output_path, exist_ok=True)
for dict in tqdm(data):
    for pos in ['_left.jpg', '_right.jpg']:
        print(dict['file_name'])
        im = cv2.imread('dataset/test/' + dict['file_name'])

        # mask
        mask = np.zeros([im.shape[0], im.shape[1]], dtype=np.uint8)
        if(pos == '_left.jpg'):
            mask[0:1000, 0:500] = 255
        else:
            mask[0:1000, 500:1000] = 255

        image = cv2.add(im, np.zeros(np.shape(im), dtype=np.uint8), mask=mask)
        cv2.imwrite(
            output_path+str(dict['id'])+pos, image)
