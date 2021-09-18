# Perform evaluation

import argparse
import torch

from detectron2.config import get_cfg
from define_dataset import my_dataset
from detectron2.evaluation import COCOEvaluator
from detectron2.data import build_detection_test_loader
from detectron2.data import DatasetCatalog
from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.data.detection_utils import read_image

if __name__ == "__main__":
    def setup_cfg(args):
        # Load config from file and command-line arguments
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
    model = DefaultPredictor(cfg)

    # Load custom dataset
    DatasetCatalog.register("my_dataset", my_dataset)
    MetadataCatalog.get("my_dataset").thing_classes=['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    data_loader = build_detection_test_loader(cfg, "my_dataset")

    # Define evaluator
    evaluator = COCOEvaluator("my_dataset", ("bbox",), False, output_dir=args.output)

	# Evaluation
    with torch.no_grad():
        def get_all_inputs_outputs():
            image_id = 0
            for data in data_loader:
                img = read_image(data[0]["file_name"], format="BGR")
                pred = model(img)
                out = []
                out.append(pred)
                image_id = image_id + 1
                yield data, out
        evaluator.reset()
        for inputs, outputs in get_all_inputs_outputs():
            evaluator.process(inputs, outputs)
        eval_results = evaluator.evaluate() # Compare preditions to annotations
        print(eval_results)
        results_file = args.output + 'eval_results.txt'
        with open(results_file, 'a', encoding='utf-8') as f:
            f.write(str(eval_results))
