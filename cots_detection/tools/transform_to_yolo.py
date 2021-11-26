"""Module with transforming of dataset to YoloV5 format"""

import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Union

import pandas as pd
from tqdm import tqdm


class DatasetToYoloTransformer:
    ANNOTATIONS_COLUMN = 'annotations'
    VIDEO_ID_COLUMN = 'video_id'
    VIDEO_FRAME_COLUMN = 'video_frame'

    def __init__(
        self,
        annotation_dataframe: pd.DataFrame,
        images_root: Union[Path, str],
        labels_root: Union[Path, str],
        new_images_root: Union[Path, str],
        image_extension: str = '.jpg',
        image_size: Tuple[int, int] = (1280, 720),
        verbose: bool = True,
    ) -> None:
        self._annotation_dataframe = annotation_dataframe
        self._images_root = Path(images_root)
        self._labels_root = Path(labels_root)
        self._new_images_root = Path(new_images_root)

        self._image_extension = image_extension
        self._image_size = image_size
        self._verbose = verbose

    def transform(self):
        self._preprocess_dataframe()
        width_image, height_image = self._image_size

        for _, curr_row in tqdm(
            self._annotation_dataframe.iterrows(),
            postfix='Transforming dataset...',
            disable=not self._verbose,
            total=self._annotation_dataframe.shape[0],
        ):
            annotations = curr_row[self.ANNOTATIONS_COLUMN]
            video_id = curr_row[self.VIDEO_ID_COLUMN]
            video_frame = curr_row[self.VIDEO_FRAME_COLUMN]

            if len(annotations) > 0:
                bboxes = self._transform_all_bboxes(
                    bboxes=annotations,
                    width_image=width_image,
                    height_image=height_image,
                )
                self._copy_image_and_save_annotation(
                    video_id=video_id, video_frame=video_frame, bboxes=bboxes
                )

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

    def _transform_all_bboxes(
        self, bboxes: List[Dict[str, int]], width_image: int, height_image: int
    ) -> List[List[float]]:
        transformed_bboxes = []

        for curr_bbox in bboxes:
            transformed_bbox = self._transform_bbox(
                bbox=curr_bbox, width_image=width_image, height_image=height_image
            )
            transformed_bboxes.append(transformed_bbox)

        return transformed_bboxes

    def _copy_image_and_save_annotation(
        self,
        video_id: int,
        video_frame: int,
        bboxes: List[List[float]],
    ) -> None:
        image_name = str(video_frame) + self._image_extension
        image_res_name = str(video_id) + '_' + str(video_frame) + self._image_extension
        annotation_res_name = str(video_id) + '_' + str(video_frame) + '.txt'

        image_path = self._images_root.joinpath(f'video_{str(video_id)}', image_name)
        image_res_path = self._new_images_root.joinpath(image_res_name)
        annotation_path = self._labels_root.joinpath(annotation_res_name)

        self._copy_image(
            from_path=image_path,
            to_path=image_res_path,
        )
        self._save_annotation(annotation_path=annotation_path, bboxes=bboxes)

    @staticmethod
    def _copy_image(from_path: Path, to_path: Path) -> None:
        shutil.copy(from_path, to_path)

    @staticmethod
    def _save_annotation(annotation_path: Path, bboxes: List[List[float]]) -> None:
        annotation_string = ''

        for curr_bbox in bboxes:
            curr_annotation_string = ' '.join([str(item) for item in curr_bbox])
            curr_annotation_string = '0 ' + curr_annotation_string  # bbox class

            annotation_string += curr_annotation_string + '\n'

        with annotation_path.open(mode='w') as file:
            file.write(annotation_string)

    @staticmethod
    def _transform_bbox(
        bbox: Dict[str, int], width_image: int, height_image: int
    ) -> List[float]:
        x_min = bbox['x']
        y_min = bbox['y']
        width = float(bbox['width'])
        height = float(bbox['height'])

        x_center = (x_min + width / 2) / width_image
        y_center = (y_min + height / 2) / height_image

        width = width / width_image
        height = height / height_image

        return [x_center, y_center, width, height]
