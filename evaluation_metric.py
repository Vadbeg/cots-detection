"""CLI for evaluation result on whole folder of images"""

import math
import random
from copy import deepcopy
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from cv2 import cv2
from tqdm import tqdm

from cots_detection.modules.metrics.fbeta_score import imagewise_f2_score
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


def _preprocess_dataframe(annotation_dataframe: pd.DataFrame) -> pd.DataFrame:
    ANNOTATIONS_COLUMN = 'annotations'
    VIDEO_ID_COLUMN = 'video_id'
    VIDEO_FRAME_COLUMN = 'video_frame'

    if isinstance(annotation_dataframe[ANNOTATIONS_COLUMN].loc[0], str):
        annotation_dataframe[ANNOTATIONS_COLUMN] = annotation_dataframe[
            ANNOTATIONS_COLUMN
        ].apply(eval)

    annotation_dataframe[VIDEO_ID_COLUMN] = annotation_dataframe[VIDEO_ID_COLUMN].apply(
        int
    )
    annotation_dataframe[VIDEO_FRAME_COLUMN] = annotation_dataframe[
        VIDEO_FRAME_COLUMN
    ].apply(int)

    return annotation_dataframe


def _convert_xyxy_to_xywh_coords(x: List[float]) -> List[float]:
    y = deepcopy(x)
    y[2] = math.fabs(x[2] - x[0])
    y[3] = math.fabs(x[3] - x[1])

    return y


def get_annotations_by_image_id(
    annotations: pd.DataFrame, image_id: str
) -> List[List[float]]:
    ANNOTATIONS_COLUMN = 'annotations'

    all_annotations = annotations.loc[annotations['image_id'] == image_id][
        ANNOTATIONS_COLUMN
    ].values[0]

    new_coords = []

    for curr_annotation in all_annotations:
        x = [
            curr_annotation['x'],
            curr_annotation['y'],
            curr_annotation['width'],
            curr_annotation['height'],
        ]
        new_coords.append(x)

    return new_coords


def reformat_predictions(preds: List[List[float]]) -> List[List[float]]:
    new_predictions = []

    for curr_pred in preds:
        x = _convert_xyxy_to_xywh_coords(x=curr_pred[:4])
        new_predictions.append([curr_pred[4]] + x)

    return new_predictions


if __name__ == '__main__':
    model_path = 'weights/best_long_train.torchscript.pt'
    yolo_executor = load_yolo_executor(model_path=model_path)

    images_folder_path = Path(
        '/Users/vadim.tsitko/Data/tensorflow-great-barrier-reef/yolo_train/images/train'
    )

    DATAFRAME_PATH = '/Users/vadim.tsitko/Data/tensorflow-great-barrier-reef/train.csv'
    dataframe = pd.read_csv(DATAFRAME_PATH)
    dataframe = _preprocess_dataframe(annotation_dataframe=dataframe)

    images_paths = list(images_folder_path.glob(pattern='*.jpg'))
    random.shuffle(images_paths)

    all_f2_scores = []

    for curr_image_path in tqdm(images_paths):
        curr_image_path = Path(curr_image_path)

        image_name = curr_image_path.name.split('.')[0]
        image_id = image_name.replace('_', '-')

        bboxes_true = get_annotations_by_image_id(
            annotations=dataframe, image_id=image_id
        )

        image = load_image(image_path=curr_image_path)
        bboxes = yolo_executor.execute(image)
        bboxes_to_draw = resize_bboxes(
            bboxes=bboxes,
            old_size=(1280, 1280),
            new_size=(image.shape[1], image.shape[0]),
        )

        bboxes_pred = reformat_predictions(preds=bboxes_to_draw)

        print('pred ', bboxes_pred)
        print('true ', bboxes_true)

        f2_score = imagewise_f2_score(
            gt_bboxes=np.array(bboxes_true), pred_bboxes=np.array(bboxes_pred)
        )
        print(f2_score)
        all_f2_scores.append(f2_score)

    print(np.mean(all_f2_scores))
    cv2.destroyAllWindows()
