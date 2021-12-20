"""CLI for dataframe splitting on train and test"""


from enum import Enum
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import typer


class SplitType(Enum):
    video: str = 'video'
    length: str = 'length'


def _split_dataframe_by_videos(
    dataframe: pd.DataFrame, train_video_ids: List[int], val_video_ids: List[int]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_dataframe = dataframe.loc[dataframe['video_id'].isin(train_video_ids)]
    val_dataframe = dataframe.loc[dataframe['video_id'].isin(val_video_ids)]

    return train_dataframe, val_dataframe


def _split_dataframe_by_length(
    dataframe: pd.DataFrame, val_length: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_dataframes = []
    val_dataframes = []

    for unique_video_id in dataframe['video_id'].unique():
        dataframe_part: pd.DataFrame = dataframe.loc[
            dataframe['video_id'] == unique_video_id
        ]

        dataframe_part.sort_values(by='video_frame')

        train_part = dataframe_part.iloc[: -int(val_length * dataframe_part.shape[0])]
        val_part = dataframe_part.iloc[-int(val_length * dataframe_part.shape[0]) :]

        train_dataframes.append(train_part)
        val_dataframes.append(val_part)

    train_df = pd.concat(train_dataframes)
    val_df = pd.concat(val_dataframes)

    return train_df, val_df


def split_dataframe_cli(
    original_dataframe_path: Path = typer.Option(
        ..., help='Path to original dataframe'
    ),
    train_name: str = typer.Option(
        default='train_part.csv', help='Train dataframe filename'
    ),
    val_name: str = typer.Option(default='val_part.csv', help='Val dataframe filename'),
    split_type: SplitType = typer.Option(
        default=SplitType.video,
        help=(
            'If video - takes 0 and 1 video to train, 2 - test. '
            'If length - takes 20% of every video as test. Other - train. '
        ),
    ),
) -> None:
    dataframe = pd.read_csv(original_dataframe_path)

    if split_type == SplitType.video:
        train_dataframe, val_dataframe = _split_dataframe_by_videos(
            dataframe=dataframe, train_video_ids=[0, 1], val_video_ids=[2]
        )
    elif split_type == SplitType.length:
        train_dataframe, val_dataframe = _split_dataframe_by_length(
            dataframe=dataframe, val_length=0.3
        )
    else:
        raise ValueError('No such dataset type')

    train_path = original_dataframe_path.parent.joinpath(train_name)
    val_path = original_dataframe_path.parent.joinpath(val_name)

    train_dataframe.to_csv(train_path, index=False)
    val_dataframe.to_csv(val_path, index=False)
