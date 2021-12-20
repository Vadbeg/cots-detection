import pandas as pd

if __name__ == '__main__':
    dataframe_path = '/Users/vadim.tsitko/Data/tensorflow-great-barrier-reef/train.csv'
    data = pd.read_csv(dataframe_path)

    dataset_part: pd.DataFrame = data.loc[data['video_id'] == 0]

    max_value = dataset_part['video_frame'].max()

    dataset_part.sort_values(by='video_frame')

    val_percent = 0.3
    train_part = dataset_part.iloc[: -int(val_percent * dataset_part.shape[0])]
    val_part = dataset_part.iloc[-int(val_percent * dataset_part.shape[0]) :]

    print(train_part.shape)
    print(val_part.shape)
