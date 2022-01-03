"""Module with transforming of dataset to Coco format"""

import json
import os.path
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import pandas as pd
from tqdm import tqdm


class DatasetToCocoTransformer:
    ANNOTATIONS_COLUMN = 'annotations'
    VIDEO_ID_COLUMN = 'video_id'
    VIDEO_FRAME_COLUMN = 'video_frame'

    def __init__(
        self,
        annotation_dataframe: pd.DataFrame,
        image_extension: str = '.jpg',
        image_size: Tuple[int, int] = (1280, 720),
        verbose: bool = True,
    ) -> None:
        self._annotation_dataframe = annotation_dataframe

        self._image_extension = image_extension
        self._image_size = image_size
        self._verbose = verbose

    def transform(self) -> Dict[str, Any]:
        self._preprocess_dataframe()
        width_image, height_image = self._image_size

        images_info = []
        annotations_info = []
        categories_info = [{'supercategory': 'cot', 'id': 0, 'name': 'cot'}]
        last_segmentation_id = -1

        for index, curr_row in tqdm(
            self._annotation_dataframe.iterrows(),
            postfix='Transforming dataset...',
            disable=not self._verbose,
            total=self._annotation_dataframe.shape[0],
        ):
            annotations = curr_row[self.ANNOTATIONS_COLUMN]
            video_id = curr_row[self.VIDEO_ID_COLUMN]
            video_frame = curr_row[self.VIDEO_FRAME_COLUMN]

            if len(annotations) > 0:
                image_info = self._get_image_info(
                    image_id=index,
                    video_id=video_id,
                    video_frame=video_frame,
                    width_image=width_image,
                    height_image=height_image,
                )
                (
                    image_annotations_info,
                    last_segmentation_id,
                ) = self._get_image_annotations_info(
                    image_id=index,
                    last_segmentation_id=last_segmentation_id,
                    coords=annotations,
                )

                images_info.append(image_info)
                annotations_info.extend(image_annotations_info)

        print(last_segmentation_id)

        result = {
            'images': images_info,
            'annotations': annotations_info,
            'categories': categories_info,
        }

        return result

    @staticmethod
    def save_json(obj: Any, path: Path) -> None:
        with path.open(mode='w', encoding='UTF-8') as file:
            json.dump(obj, fp=file)

    def _get_image_info(
        self,
        image_id: int,
        video_id: int,
        video_frame: int,
        width_image: int,
        height_image: int,
    ) -> Dict[str, Any]:
        image_name = str(video_frame) + self._image_extension
        image_rel_path = os.path.join(f'video_{str(video_id)}', image_name)

        image_info = {
            'file_name': image_rel_path,
            'width': width_image,
            'height': height_image,
            'id': image_id,
        }

        return image_info

    def _get_image_annotations_info(
        self, image_id: int, last_segmentation_id: int, coords: List[Dict[str, int]]
    ) -> Tuple[List[Dict[str, Any]], int]:
        image_annotations = []

        for curr_coords in coords:
            transformed_coords = self._transform_bbox(bbox=curr_coords)

            last_segmentation_id += 1
            curr_image_annotation = {
                'bbox': transformed_coords,
                'category_id': 0,
                'image_id': image_id,
                'iscrowd': False,
                'area': transformed_coords[2] * transformed_coords[3],
                'id': last_segmentation_id,
            }
            image_annotations.append(curr_image_annotation)

        return image_annotations, last_segmentation_id

    def _preprocess_dataframe(self) -> None:
        if isinstance(self._annotation_dataframe[self.ANNOTATIONS_COLUMN].loc[0], str):
            self._annotation_dataframe[
                self.ANNOTATIONS_COLUMN
            ] = self._annotation_dataframe[self.ANNOTATIONS_COLUMN].apply(eval)

        self._annotation_dataframe[self.VIDEO_ID_COLUMN] = self._annotation_dataframe[
            self.VIDEO_ID_COLUMN
        ].apply(int)
        self._annotation_dataframe[
            self.VIDEO_FRAME_COLUMN
        ] = self._annotation_dataframe[self.VIDEO_FRAME_COLUMN].apply(int)

    @staticmethod
    def _transform_bbox(
        bbox: Dict[str, int], original_image_size: Tuple[int, int] = (1280, 720)
    ) -> List[float]:
        x_min = int(bbox['x'])
        y_min = int(bbox['y'])
        width = int(bbox['width'])
        height = int(bbox['height'])

        if x_min + width > original_image_size[0]:
            width = original_image_size[0] - x_min
        if y_min + height > original_image_size[1]:
            height = original_image_size[1] - y_min

        return [x_min, y_min, width, height]
