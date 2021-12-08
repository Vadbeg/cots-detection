"""Module with utils for whole project"""

from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import yaml
from cv2 import cv2
from yaml import Loader

from cots_detection.modules.evaluation.yolo_execution import YoloExecutor


def load_config(config_path: Union[Path, str]) -> Dict:
    config_path = Path(config_path)

    with config_path.open(mode='r') as file:
        config = yaml.load(file.read(), Loader=Loader)

    return config


def load_jit_model(model_path: Union[str, Path]) -> torch.jit.TracedModule:
    model = torch.jit.load(f=model_path, map_location=torch.device('cpu'))

    return model


def load_yolo_executor(model_path: Union[str, Path]) -> YoloExecutor:
    model = load_jit_model(model_path=model_path)
    yolo_executor = YoloExecutor(model=model, image_size=(1280, 1280))

    return yolo_executor


def resize_bboxes(bboxes: List[List[float]], old_size: Tuple, new_size: Tuple) -> List:
    resized_bboxes_and_conf = []

    for curr_bbox_and_conf in bboxes:
        curr_bbox_and_conf[0] = (curr_bbox_and_conf[0] / old_size[0]) * new_size[0]
        curr_bbox_and_conf[1] = (curr_bbox_and_conf[1] / old_size[1]) * new_size[1]
        curr_bbox_and_conf[2] = (curr_bbox_and_conf[2] / old_size[0]) * new_size[0]
        curr_bbox_and_conf[3] = (curr_bbox_and_conf[3] / old_size[1]) * new_size[1]

        resized_bboxes_and_conf.append(curr_bbox_and_conf)

    return resized_bboxes_and_conf


def load_image(image_path: Union[str, Path]) -> np.ndarray:
    image = cv2.imread(filename=str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image
