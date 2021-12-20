"""CLI for evaluation result on whole folder of images"""

import random
from pathlib import Path
from typing import List

import numpy as np
from cv2 import cv2
from tqdm import tqdm

from cots_detection.utils import load_image, load_yolo_executor, resize_bboxes


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


if __name__ == '__main__':
    model_path = 'weights/best_long_train.pt'
    yolo_executor = load_yolo_executor(model_path=model_path)

    image_path = '/Users/vadim.tsitko/Data/tensorflow-great-barrier-reef/yolo_train/images/train/1_4517.jpg'

    images_folder_path = Path(
        '/Users/vadim.tsitko/Data/tensorflow-great-barrier-reef/yolo_train/images/val'
    )

    images_paths = list(images_folder_path.glob(pattern='*.jpg'))
    random.shuffle(images_paths)

    for curr_image_path in tqdm(images_paths):
        image = load_image(image_path=curr_image_path)
        bboxes = yolo_executor.execute(image)
        bboxes_to_draw = resize_bboxes(
            bboxes=bboxes,
            old_size=(640, 640),
            new_size=(image.shape[1], image.shape[0]),
        )

        image = draw_bboxes(image=image, bboxes_and_conf=bboxes, random_color=False)
        cv2.imshow('Image', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        key = cv2.waitKey(0)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
