"""Module with YOLO execution"""

from typing import List, Tuple

import numpy as np
import torch
import torchvision
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
        # image = image.transpose((2, 0, 1))

        model_result = self._get_model_result(image=image)
        # processed_result = self._perform_postprocessing(y=model_result)
        processed_result = self._yolox_postporcess(
            prediction=model_result,
            num_classes=1,
            conf_thre=0.1,
        )
        processed_result = [
            curr_res.detach().cpu().numpy().tolist()
            for curr_res in processed_result
            if curr_res is not None
        ]

        return processed_result

    def _get_model_result(self, image: np.ndarray) -> np.ndarray:
        # image_tensor = torch.from_numpy(image) / 255
        # image_tensor = image_tensor.unsqueeze(0).float()

        image_tensor = self.preproc(img=image, input_size=(640, 640, 3))[0]
        image_tensor = torch.from_numpy(image_tensor).unsqueeze(0)
        image_tensor = image_tensor.float()

        # result = self.model(image_tensor)[0].detach().cpu().numpy()
        result = self.model(image_tensor).detach()

        return result

    @staticmethod
    def preproc(img, input_size, swap=(2, 0, 1)):
        if len(img.shape) == 3:
            padded_img = (
                np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
            )
        else:
            padded_img = np.ones(input_size, dtype=np.uint8) * 114

        r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

        padded_img = padded_img.transpose(swap)
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        return padded_img, r

    def _perform_postprocessing(
        self, y: np.ndarray, min_score: float = 0.01, min_iou: float = 0.45
    ) -> List[List[float]]:
        y_mask = y[..., 4] > min_score
        y_pred = y[y_mask]
        y_pred = self._convert_xywh_to_xyxy_coords(x=y_pred)

        pred_idxs = self._non_maximum_supression_cpu(
            boxes=y_pred, overlap_threshold=min_iou
        )
        y_pred = np.take(y_pred, indices=pred_idxs, axis=0).tolist()

        return y_pred

    def _yolox_postporcess(
        self,
        prediction,
        num_classes,
        conf_thre=0.7,
        nms_thre=0.45,
        class_agnostic=False,
    ):
        box_corner = prediction.new(prediction.shape)
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
        prediction[:, :, :4] = box_corner[:, :, :4]

        output = [None for _ in range(len(prediction))]
        for i, image_pred in enumerate(prediction):

            # If none are remaining => process next image
            if not image_pred.size(0):
                continue
            # Get score and class with highest confidence
            class_conf, class_pred = torch.max(
                image_pred[:, 5 : 5 + num_classes], 1, keepdim=True
            )

            conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
            # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
            detections = torch.cat(
                (image_pred[:, :5], class_conf, class_pred.float()), 1
            )
            detections = detections[conf_mask]
            if not detections.size(0):
                continue

            if class_agnostic:
                nms_out_index = torchvision.ops.nms(
                    detections[:, :4],
                    detections[:, 4] * detections[:, 5],
                    nms_thre,
                )
            else:
                nms_out_index = torchvision.ops.batched_nms(
                    detections[:, :4],
                    detections[:, 4] * detections[:, 5],
                    detections[:, 6],
                    nms_thre,
                )

            detections = detections[nms_out_index]
            if output[i] is None:
                output[i] = detections
            else:
                output[i] = torch.cat((output[i], detections))

        return output

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
