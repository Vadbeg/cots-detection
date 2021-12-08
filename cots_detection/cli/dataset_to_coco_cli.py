"""Module with CLI for converting dataset to YoloV5 format"""


from pathlib import Path
from typing import Tuple

import pandas as pd
import typer

from cots_detection.tools.transform_to_coco import DatasetToCocoTransformer


def transform_dataset_to_coco_cli(
    annotations_path: Path = typer.Option(..., help='Path to .csv dataset'),
    res_path: Path = typer.Option(..., help='Path to .json COCO dataset'),
    image_extension: str = typer.Option(
        default='.jpg', help='Extension of images in `images_root`'
    ),
    image_size: Tuple[int, int] = typer.Option(default=(1280, 720), help='Images size'),
    verbose: bool = typer.Option(default=True, help='If true shows progress bar'),
) -> None:
    dataframe = pd.read_csv(annotations_path)

    dataset_to_yolo_transformer = DatasetToCocoTransformer(
        annotation_dataframe=dataframe,
        image_extension=image_extension,
        image_size=image_size,
        verbose=verbose,
    )

    result = dataset_to_yolo_transformer.transform()
    dataset_to_yolo_transformer.save_json(result, res_path)
