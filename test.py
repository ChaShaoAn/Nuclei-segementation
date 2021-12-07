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

cfg = get_cfg()
cfg.merge_from_file(
    model_zoo.get_config_file(
        'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'))
cfg.DATASETS.TRAIN = ("nuclei_dataset",)
# cfg.DATASETS.TRAIN = ("balloon_train",)
cfg.DATASETS.TEST = ()   # no metrics implemented for this dataset
cfg.DATALOADER.NUM_WORKERS = 0
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 1000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 8
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

# Inference should use the config with parameters that are used in training
# cfg now already contains everything we've set previously.
# We changed it a little bit for inference:
# path to the model we just trained
# cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.WEIGHTS = os.path.join(
    cfg.OUTPUT_DIR, "model_final-23100-exceed_baseline.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3   # set a custom testing threshold
predictor = DefaultPredictor(cfg)

output_path = 'submission/'

f = open('dataset/test_img_ids.json')
data = json.load(f)
f.close()
tmp_annotation = []
for dict in tqdm(data):

    print(dict['file_name'])
    im = cv2.imread('dataset/test/' + dict['file_name'])

    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=MetadataCatalog.get("nuclei_dataset"),
                   scale=0.5,
                   instance_mode=ColorMode.IMAGE_BW)

    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    cv2.imwrite(
        output_path+dict['file_name']+'.jpg', out.get_image()[:, :, ::-1])

    instances = outputs['instances'].to('cpu')
    instances.pred_masks_rle = [
        mask_util.encode(
            np.asfortranarray(mask)) for mask in instances.pred_masks]
    for rle in instances.pred_masks_rle:
        rle['counts'] = rle['counts'].decode('utf-8')

    pred_boxes = instances.pred_boxes
    boxes = pred_boxes.tensor.cpu().numpy()
    tmp_scores = instances.scores
    scores = tmp_scores.cpu().numpy()
    pred_masks_rle = instances.pred_masks_rle

    for i in tqdm(range(0, len(instances))):
        x0, y0, x1, y1 = boxes[i]
        x = x0
        y = y0
        w = x1 - x0
        h = y1 - y0
        ann = {
               "image_id": dict['id'],
               "bbox": [float(x), float(y), float(w), float(h)],
               "score": float(scores[i]),
               "category_id": int(1),
               "segmentation": pred_masks_rle[i],
            }
        # print(ann)

        tmp_annotation.append(ann)

# print(tmp_annotation)

json_obj = json.dumps(tmp_annotation, indent=2)
with open("submission/answer.json", "w") as outfile:
    outfile.write(json_obj)

'''
evaluator = COCOEvaluator("nuclei_dataset_val", output_dir="./output")
val_loader = build_detection_test_loader(cfg, "nuclei_dataset_val")
print(inference_on_dataset(predictor.model, val_loader, evaluator))
'''
