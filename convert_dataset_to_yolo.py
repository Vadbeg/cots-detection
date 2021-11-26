"""CLI for converting original dataset to CoreML"""

import warnings

import typer

from cots_detection.cli.dataset_to_yolo_cli import transform_dataset_to_yolo_cli

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    typer.run(transform_dataset_to_yolo_cli)
