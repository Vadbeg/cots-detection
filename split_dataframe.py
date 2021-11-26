"""CLI for splitting original dataset on train and val parts"""

import warnings

import typer

from cots_detection.cli.split_dataframe import split_dataframe_cli

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    typer.run(split_dataframe_cli)
