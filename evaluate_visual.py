"""CLI for evaluation result on whole folder of images"""

from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import torch
from cv2 import cv2
from tqdm import tqdm

from cots_detection.modules.evaluation.yolo_execution import YoloExecutor


def load_jit_model(model_path: Union[str, Path]) -> torch.jit.TracedModule:
    model = torch.jit.load(f=model_path, map_location=torch.device('cpu'))

    return model


def load_yolo_executor(model_path: Union[str, Path]) -> YoloExecutor:
    model = load_jit_model(model_path=model_path)
    yolo_executor = YoloExecutor(model=model)

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


def draw_bboxes(
    image: np.ndarray, bboxes_and_conf: List[List[float]], random_color: bool = True
) -> np.ndarray:
    for curr_pred in bboxes_and_conf:
        x_min, y_min, x_max, y_max, conf, _ = curr_pred

        x_min = int(x_min)
        y_min = int(y_min)
        x_max = int(x_max)
        y_max = int(y_max)

        color = (0, 255, 0)
        if random_color:
            color = (
                np.random.randint(0, 255),
                np.random.randint(0, 255),
                np.random.randint(0, 255),
            )

        image = cv2.rectangle(
            image,
            pt1=(x_min, y_min),
            pt2=(x_max, y_max),
            color=color,
            thickness=2,
        )

    return image


def load_image(image_path: Union[str, Path]) -> np.ndarray:
    image = cv2.imread(filename=str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


if __name__ == '__main__':
    model_path = 'best.torchscript.pt'
    yolo_executor = load_yolo_executor(model_path=model_path)

    image_path = '/Users/vadim.tsitko/Data/tensorflow-great-barrier-reef/yolo_train/images/train/1_4517.jpg'

    images_folder_path = Path(
        '/Users/vadim.tsitko/Data/tensorflow-great-barrier-reef/yolo_train/images/train'
    )

    for curr_image_path in tqdm(list(images_folder_path.glob(pattern='*.jpg'))):
        image = load_image(image_path=image_path)
        bboxes = yolo_executor.execute(image)

        if len(bboxes) > 0:
            print(bboxes)
            # image = draw_bboxes(image)
