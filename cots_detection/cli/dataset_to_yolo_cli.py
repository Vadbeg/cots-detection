"""Module with CLI for converting dataset to YoloV5 format"""


from pathlib import Path
from typing import Tuple

import pandas as pd
import typer

from cots_detection.tools.transform_to_yolo import DatasetToYoloTransformer


def transform_dataset_to_yolo_cli(
    annotations_path: Path = typer.Option(..., help='Path to .csv dataset'),
    images_root: Path = typer.Option(..., help='Path to images root directory'),
    labels_root: Path = typer.Option(..., help='Path to new labels root directory'),
    new_images_root: Path = typer.Option(..., help='Path to new images root directory'),
    image_extension: str = typer.Option(
        default='.jpg', help='Extension of images in `images_root`'
    ),
    image_size: Tuple[int, int] = typer.Option(default=(1280, 720), help='Images size'),
    verbose: bool = typer.Option(default=True, help='If true shows progress bar'),
) -> None:
    dataframe = pd.read_csv(annotations_path)

    new_images_root.mkdir(exist_ok=True)
    labels_root.mkdir(exist_ok=True)

    dataset_to_yolo_transformer = DatasetToYoloTransformer(
        annotation_dataframe=dataframe,
        images_root=images_root,
        labels_root=labels_root,
        new_images_root=new_images_root,
        image_extension=image_extension,
        image_size=image_size,
        verbose=verbose,
    )

    dataset_to_yolo_transformer.transform()
