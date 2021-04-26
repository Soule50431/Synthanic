import xgboost
import pickle

from utils import *


def train_model(output_file, skip_columns=[], num_boost_round=100, early_stopping_rounds=20):
    # 学習データ読み込み
    train, _ = load_datasets()

    # XGBoost用のDMatrixを作成
    use_columns = get_use_columns(skip_columns=skip_columns)
    target = load_json("target_name")
    x = train[use_columns]
    y = train[target]
    dtrain = xgboost.DMatrix(x, label=y, feature_names=use_columns)

    # 学習
    watch_list = [(dtrain, "train")]
    model = xgboost.train(load_json("parameters"), dtrain,
                          num_boost_round=num_boost_round, early_stopping_rounds=early_stopping_rounds,
                          evals=watch_list)

    # モデルの保存
    with open(load_path("models_path")/add_pkl(output_file), "wb") as f:
        pickle.dump(model, f)
    return model
