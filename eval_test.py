import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm
import json
import torch

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from define_dataset import my_dataset_8ch, my_dataset_64ch
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, inference_context
from detectron2.data import build_detection_test_loader
from detectron2.modeling import build_model
from detectron2.data import DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import load_coco_json
from predictor import instances_to_coco_json

if __name__ == "__main__":
    def setup_cfg(args):
        # load config from file and command-line arguments
        cfg = get_cfg()
        cfg.merge_from_file(args.config_file)
        cfg.merge_from_list(args.opts)
        # Set score_threshold for builtin models
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
        cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
        cfg.freeze()
        return cfg


    def get_parser():
        parser = argparse.ArgumentParser(description="Detectron2 test for evaluation of custom dataset")
        parser.add_argument(
            "--config-file",
            default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
            metavar="FILE",
            help="path to config file",
        )
        parser.add_argument(
            "--output",
            help="Output directory for evaluation files",
        )
        parser.add_argument(
            "--confidence-threshold",
            type=float,
            default=0.5,
            help="Minimum score for instance predictions to be shown",
        )
        parser.add_argument(
            "--opts",
            help="Modify config options using the command-line 'KEY VALUE' pairs",
            default=[],
            nargs=argparse.REMAINDER,
        )
        return parser

    args = get_parser().parse_args()
    cfg = setup_cfg(args)
    model = build_model(cfg)
    #print('model : ', model)

    #load custom dataset
    DatasetCatalog.register("my_dataset", my_dataset_8ch)
    MetadataCatalog.get("my_dataset").thing_classes=['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                                                     'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                                                     'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                                                     'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                                                     'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                                                     'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                                                     'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                                                     'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                                                     'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                                                     'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                                                     'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                                                     'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                                                     'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                                                     'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    #data = DatasetCatalog.get("my_dataset")
    #print('data : ', data)

    #define dataloader to turn std detectron format into model input format
    data_loader = build_detection_test_loader(cfg, "my_dataset")
    #print('dataloader : ', data_loader)

    #define and use evaluator
    evaluator = COCOEvaluator("my_dataset", ("bbox",), False, output_dir=args.output)
    #print('evaluator: ', evaluator)
    print('results: ', inference_on_dataset(model, data_loader, evaluator))

    # esto hace lo mismo que inference on dataset pero puedo guardar los resultados de la inferencia:
    #with inference_context(model), torch.no_grad():
    #    def get_all_inputs_outputs():
    #        image_id = 0
    #        for data in data_loader:
    #            out = model(data)
    #            with open('out_dict_8ch.txt', 'a', encoding='utf-8') as f2:
    #                f2.write("image_id: ")
    #                f2.write(str(image_id))
    #                f2.write(", ")
    #                f2.write(str(out))
    #            image_id = image_id + 1
    #            yield data, out

    #    evaluator.reset()
    #    for inputs, outputs in get_all_inputs_outputs():
    #        evaluator.process(inputs, outputs)
    #    eval_results = evaluator.evaluate()
    #    print(eval_results)
