# Define a custom dataset to evaluate detection

import os
import cv2
import numpy
import json

from detectron2.structures import BoxMode

dir = '/home/lab101m5/event/e2vid/rpg_e2vid/outputs/datasets/'
filename = 'person_anno.json'
bagname = 'person'

def my_dataset():
    dataset_dir = dir + bagname + '/reconstruction/'
    dataset = []
    image_list = sorted(os.listdir(dataset_dir))
    id = 0
    for img in image_list:
        # Read image:
        img_path = dataset_dir + img
        img_tensor = cv2.imread(img_path)
        h, w, c = numpy.shape(img_tensor)

        # Dict fields:
        img_dict = {}
        img_dict["file_name"] = img_path
        img_dict["height"], img_dict["width"] = h, w
        img_dict["image_id"] = id

        # Add annotations from the groundtruth json file
        img_dict["annotations"] = []
        json_file = open(filename, "r")
        json_data = json.load(json_file)
        json_file.close()
        for ann in json_data:
            for inst in ann:
                if inst["image_id"] == id:
                    annot = {}
                    annot["bbox"] = inst["bbox"]
                    annot["bbox_mode"] = BoxMode.XYWH_ABS
                    annot["category_id"] = inst["category_id"]
                    img_dict["annotations"].append(annot)

        # Append image dictionary to the list of all images
        dataset.append(img_dict)
        id = id + 1

    return dataset
