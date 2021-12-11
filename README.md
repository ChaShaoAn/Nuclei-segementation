# Nuclei-segementation
This is deep learning homework3, the proposed challenge is nuclei instance segmentation.
### Environment
- Python 3.8.11
- Pytorch 1.9.1
- CUDA 11.1
- detectron2

### Download Data
- download [dataset.zip](https://drive.google.com/file/d/1nEJ7NTtHcCHNQqUXaoPk55VH3Uwh4QGG/view)
- unzip `dataset.zip`, and will get `dataset` folder, which contains two folder and one json file.

### How to train
1. first, run `trans_to_coco_format_poly.py` to trans mask to coco format. It will create `nucleus_cocoformat_poly2.json`, and copy train images into `train` folder.
2. run `train.py` to start train.

### How to test and create submission json:
1. first, run `test_preprocess.py` to separate test images into 2 image(left and right) and put them into `dataset/test_seg/` folder.
2. run `test_seg.py` and you will get `answer.json` in `submission_seg` folder.

### Reference
- [detectron2](https://github.com/facebookresearch/detectron2)
- [Create COCO Annotations From Scratch](https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch)

