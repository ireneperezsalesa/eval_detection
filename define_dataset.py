import os
import cv2
import numpy
import json

from detectron2.structures import BoxMode


def my_dataset_8ch():
    dataset_dir = '/home/lab101m5/event/indoors_nuevo/indoors_8ch/reconstruction/'
    dataset = []
    image_list = sorted(os.listdir(dataset_dir))
    id = 0
    for img in image_list:
        #read image:
        img_path = dataset_dir + img
        img_tensor = cv2.imread(img_path)
        h, w, c = numpy.shape(img_tensor)

        #dict fields:
        img_dict = {}
        img_dict["file_name"] = img_path
        img_dict["height"], img_dict["width"] = h, w
        img_dict["image_id"] = id

        #add anotations from the output json file of groundtruth detection
        img_dict["annotations"] = []
        json_file = open("data_gt_is.json", "r")
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

        # append image dictionary to the list of all images
        dataset.append(img_dict)
        id = id + 1

    return dataset


def my_dataset_64ch():
    dataset_dir = '/home/lab101m5/event/indoors_nuevo/indoors_64ch/reconstruction/'
    dataset = []
    image_list = sorted(os.listdir(dataset_dir))
    id = 0
    for img in image_list:
        #read image:
        img_path = dataset_dir + img
        img_tensor = cv2.imread(img_path)
        h, w, c = numpy.shape(img_tensor)

        #dict fields:
        img_dict = {}
        img_dict["file_name"] = img_path
        img_dict["height"], img_dict["width"] = h, w
        img_dict["image_id"] = id

        #add anotations from the output json file of groundtruth detection
        img_dict["annotations"] = []
        json_file = open("data_gt_is.json", "r")
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

        # append image dictionary to the list of all images
        dataset.append(img_dict)
        id = id + 1

    return dataset