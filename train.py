import detectron2
from detectron2.utils.logger import setup_logger
# import some common libraries
import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
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
import torch
import json
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

# %%
setup_logger()

register_coco_instances("nuclei_dataset", {},
                        "nucleus_cocoformat_poly2.json", "./train")
register_coco_instances("nuclei_dataset_val", {},
                        "nucleus_cocoformat_poly2.json", "./val")
metadata = MetadataCatalog.get("nuclei_dataset")
dataset_dicts = DatasetCatalog.get("nuclei_dataset")

# %%
cfg = get_cfg()
# cfg.merge_from_file("./model/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
cfg.merge_from_file(model_zoo.get_config_file(
                    'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'))
# cfg.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml'))
# cfg.OUTPUT_DIR = "./output"
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
#                   'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml')
# cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
# if you have pre-trained weight.
cfg.DATASETS.TRAIN = ("nuclei_dataset",)
# cfg.DATASETS.TRAIN = ("balloon_train",)
cfg.DATASETS.TEST = ()   # no metrics implemented for this dataset
cfg.DATALOADER.NUM_WORKERS = 0
cfg.SOLVER.IMS_PER_BATCH = 2
# cfg.SOLVER.BASE_LR = 0.00025
# 4999 iterations seems good enough, but you can certainly train longer
cfg.SOLVER.MAX_ITER = 10000
# faster, and good enough for this toy dataset
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 8
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # 1 classes (data, fig, hazelnut)
# cfg.MODEL.DEVICE ='cuda'
# cfg.INPUT.MASK_FORMAT = 'bitmask'

# %%
print(cfg.OUTPUT_DIR)
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)  # build output folder
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
# %%
for d in random.sample(dataset_dicts, 1):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5)
    out = visualizer.draw_dataset_dict(d)
    # cv2_imshow(out.get_image()[:, :, ::-1])
    # cv2.imshow("out", out.get_image()[:, :, ::-1])
    cv2.imwrite('output_poly.jpg', out.get_image()[:, :, ::-1])

# %%
# Inference should use the config with parameters that are used in training
# cfg now already contains everything we've set previously.
# We changed it a little bit for inference:
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
# path to the model we just trained
# cfg.MODEL.WEIGHTS = os.path.join("output-23100", "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4   # set a custom testing threshold
predictor = DefaultPredictor(cfg)

# %%
evaluator = COCOEvaluator("nuclei_dataset_val", output_dir="./output")
val_loader = build_detection_test_loader(cfg, "nuclei_dataset_val")
print(inference_on_dataset(predictor.model, val_loader, evaluator))
