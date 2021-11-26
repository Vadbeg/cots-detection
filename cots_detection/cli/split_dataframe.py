"""CLI for dataframe splitting on train and test"""


from pathlib import Path
from typing import List, Tuple

import pandas as pd
import typer


def _split_dataframe(
    dataframe: pd.DataFrame, train_video_ids: List[int], val_video_ids: List[int]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_dataframe = dataframe.loc[dataframe['video_id'].isin(train_video_ids)]
    val_dataframe = dataframe.loc[dataframe['video_id'].isin(val_video_ids)]

    return train_dataframe, val_dataframe


def split_dataframe_cli(
    original_dataframe_path: Path = typer.Option(
        ..., help='Path to original dataframe'
    ),
    train_name: str = typer.Option(
        default='train_part.csv', help='Train dataframe filename'
    ),
    val_name: str = typer.Option(default='val_part.csv', help='Val dataframe filename'),
) -> None:
    dataframe = pd.read_csv(original_dataframe_path)

    train_dataframe, val_dataframe = _split_dataframe(
        dataframe=dataframe, train_video_ids=[0, 1], val_video_ids=[2]
    )

    train_path = original_dataframe_path.parent.joinpath(train_name)
    val_path = original_dataframe_path.parent.joinpath(val_name)

    train_dataframe.to_csv(train_path, index=False)
    val_dataframe.to_csv(val_path, index=False)
