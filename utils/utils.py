import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
from discordwebhook import Discord
import datetime, time
import csv


def send_message_to_discord(message):
    # Discordサーバーにメッセージを送信
    discord = Discord(url=load_json("discord_url"))
    discord.post(content=message)


def send_error_to_discord():
    import traceback
    traceback.format_exc()
    send_message_to_discord(":no_entry_sign:" + "-"*50 + "\n" + traceback.format_exc() + "\n" + "-"*50)


def add_csv(file_name):
    # ファイル拡張子をcsvに変更
    return file_name.split(".")[0] + ".csv"


def add_feather(file_name):
    # ファイル拡張子をftrに変更
    return file_name.split(".")[0] + ".ftr"


def add_pkl(file_name):
    # ファイル拡張子をpklに変更
    return file_name.split(".")[0] + ".pkl"


def csv_to_feather(csv_names):
    input_path = load_path("inputs_path")
    for file in csv_names:
        csv_file = pd.read_csv(input_path/add_csv(file))
        reduce_mem_usage(csv_file).to_feather(input_path/add_feather(file))

        del csv_file


def load_path(path_name):
    # default.jsonに保存されているパスを返す
    path = Path(load_json("root")) / Path(load_json(path_name))
    return path


def load_json(column):
    # default.jsonのcolumnの値を返す
    with open("../config/default.json", "r") as json_file:
        json_data = json.load(json_file)
        return json_data[column]


def clear_json(features, classes):
    config_path = load_path("config_path")
    if config_path.exists():
        with open(config_path, "r") as json_file:
            json_data = json.load(json_file)
    else:
        json_data = {}

    for (feature, class_) in zip(features, classes):
        json_data[feature] = class_()

    with open(config_path, "w") as json_file:
        json.dump(json_data, json_file, indent=4)


def add_json(feature, data, overwrite=False):
    config_path = load_path("config_path")
    if config_path.exists():
        with open(config_path, "r") as json_file:
            json_data = json.load(json_file)
    else:
        if type(data) is list:
            json_data = {feature: []}
        elif type(data) is dict:
            json_data = {feature: {}}
        else:
            json_data = {feature: []}
    if overwrite:
        if type(data) is list:
            json_data[feature] = []
        elif type(data) is dict:
            json_data[feature] = {}
        else:
            json_data[feature] = []

    if type(data) is list:
        if feature in json_data:
            json_data[feature].extend(data)
        else:
            json_data[feature] = data
    elif type(data) is dict:
        json_data[feature].update(data)
    else:
        json_data[feature].extend([data])

    with open(config_path, "w") as json_file:
        json.dump(json_data, json_file, indent=4)


def remove_id_and_target(columns):
    if type(columns) is not list:
        columns = list(columns)

    id_name = load_json("ID_name")
    target_name = load_json("target_name")

    if id_name in columns:
        columns.remove(id_name)
    if target_name in columns:
        columns.remove(target_name)
    return columns


def reduce_mem_usage(df, logger=None, level=logging.DEBUG):
    print_ = print if logger is None else lambda msg: logger.log(level, msg)
    start_mem = df.memory_usage().sum() / 1024**2
    print_('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype
        if col_type != 'object' and col_type != 'datetime64[ns]':
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32)  # feather-format cannot accept float16
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        elif col_type == 'object':
            df[col] = df[col].astype("object")

    end_mem = df.memory_usage().sum() / 1024**2
    print_('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print_('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df


def load_datasets(suffix):
    features = load_json("features")

    assert suffix == "csv" or suffix == "ftr"
    if suffix == "csv":
        dfs = [pd.read_csv(load_path("inputs_path") / f"train_test.csv", dtype=load_json("dtypes"))]
        print("load train_test.csv")
    elif suffix == "ftr":
        dfs = [pd.read_feather(load_path("features_path")/f"{feature}_train_test.ftr") for feature in features]
        print("load train_test.ftr")
    df = pd.concat(dfs, axis=1)
    del dfs
    size = load_json("size")

    train = df.iloc[size[0]:size[1]].copy()
    test = df.iloc[size[1]:size[2]].copy()
    del df
    test.drop([load_json("target_name")], axis=1, inplace=True)

    return train, test


def add_time(model_name):
    today = datetime.datetime.fromtimestamp(time.time())
    now = today.strftime("%Y-%m-%d_%H-%M-%S")
    return model_name + f"_{now}"


def get_use_columns(skip_features=[]):
    # skip_columnsで使用しない特徴量を選択し
    # default.jsonで指定されているId_nameとtarget_nameは除いた
    # カラム名のリストを返す
    features = load_json("features")
    use_columns = []
    for key, value in features.items():
        if not(key in skip_features):
            use_columns.extend(value)
    use_columns = remove_id_and_target(use_columns)

    return use_columns


def save_evaluation(output_file, evaluation_value, use_columns):
    evaluation_value_path = load_path("memos_path")/"cross_validation.csv"
    if not evaluation_value_path.exists():
        header = ["model", "evaluation_value", "used_columns", "memo"]
    else:
        header = None

    with open(evaluation_value_path, "a", newline="") as f:
        writer = csv.writer(f)
        if header is not None:
            writer.writerow(header)
        writer.writerow([output_file, evaluation_value, use_columns, ""])
