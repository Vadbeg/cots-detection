"""Module with YOLO execution"""

from typing import List, Tuple

import numpy as np
import torch
from cv2 import cv2

from cots_detection.modules.evaluation.base_execution import BaseExecutor
from cots_detection.modules.help import to_tensor


class YoloExecutor(BaseExecutor):
    def __init__(
        self,
        model: torch.jit.TracedModule,
        image_size: Tuple[int, int] = (640, 640),
    ):
        self.model = model
        self.image_size = image_size

    def execute(self, image: np.ndarray) -> List[List[float]]:
        image = cv2.resize(image, self.image_size)
        image = image.transpose((2, 0, 1))

        model_result = self._get_model_result(image=image)
        processed_result = self._perform_postprocessing(y=model_result)

        return processed_result

    def _get_model_result(self, image: np.ndarray) -> np.ndarray:
        image_tensor = torch.from_numpy(image) / 255
        image_tensor = image_tensor.unsqueeze(0).float()

        result = self.model(image_tensor)[0].cpu().numpy()

        return result

    def _perform_postprocessing(
        self, y: np.ndarray, min_score: float = 0.1, min_iou: float = 0.45
    ) -> List[List[float]]:
        y_mask = y[..., 4] > min_score
        y_pred = y[y_mask]
        y_pred = self._convert_xywh_to_xyxy_coords(x=y_pred)

        pred_idxs = self._non_maximum_supression_cpu(
            boxes=y_pred, overlap_threshold=min_iou
        )
        y_pred = np.take(y_pred, indices=pred_idxs, axis=0).tolist()

        return y_pred

    @staticmethod
    def _convert_xywh_to_xyxy_coords(x: np.ndarray) -> np.ndarray:
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

    @staticmethod
    def _resize_bboxes(
        bboxes_and_conf: List[List[float]], old_size: Tuple, new_size: Tuple
    ) -> List:
        resized_bboxes_and_conf = []

        for curr_bbox_and_conf in bboxes_and_conf:
            curr_bbox_and_conf[0] = (curr_bbox_and_conf[0] / old_size[0]) * new_size[0]
            curr_bbox_and_conf[1] = (curr_bbox_and_conf[1] / old_size[1]) * new_size[1]
            curr_bbox_and_conf[2] = (curr_bbox_and_conf[2] / old_size[0]) * new_size[0]
            curr_bbox_and_conf[3] = (curr_bbox_and_conf[3] / old_size[1]) * new_size[1]

            resized_bboxes_and_conf.append(curr_bbox_and_conf)

        return resized_bboxes_and_conf

    @staticmethod
    def _non_maximum_supression_cpu(
        boxes: np.ndarray, overlap_threshold: float = 0.5, min_mode: bool = False
    ) -> List[int]:
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        scores = boxes[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            keep.append(order[0])
            xx1 = np.maximum(x1[order[0]], x1[order[1:]])
            yy1 = np.maximum(y1[order[0]], y1[order[1:]])
            xx2 = np.minimum(x2[order[0]], x2[order[1:]])
            yy2 = np.minimum(y2[order[0]], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h

            if min_mode:
                ovr = inter / np.minimum(areas[order[0]], areas[order[1:]])
            else:
                ovr = inter / (areas[order[0]] + areas[order[1:]] - inter)

            inds = np.where(ovr <= overlap_threshold)[0]
            order = order[inds + 1]

        return keep
