import xgboost
import pickle

from utils.utils import *
from utils.models import *


def train_model(output_file, training_algorithm,suffix="ftr", skip_features=[]):
    # 学習データ読み込み
    train, _ = load_datasets(suffix)

    # XGBoost用のDMatrixを作成
    use_columns = get_use_columns(skip_features=skip_features)
    target = load_json("target_name")
    x = train[use_columns]
    y = train[target]
    # dtrain = xgboost.DMatrix(x, label=y, feature_names=use_columns)

    # 学習
    # watch_list = [(dtrain, "train")]
    # model = xgboost.train(load_json("parameters"), dtrain,
    #                       num_boost_round=num_boost_round, early_stopping_rounds=early_stopping_rounds,
    #                       evals=watch_list)
    model = training_algorithm(x, y)
    print("trained")
    # モデルの保存
    with open(load_path("models_path")/add_pkl(output_file), "wb") as f:
        pickle.dump(model, f)
    return model
