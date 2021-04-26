import numpy as np
from utils import *
from predict import predict_
from sklearn.model_selection import KFold
import xgboost


def cross_validate(output_file, evaluation_function, skip_columns=[],
                   num_boost_round=100, early_stopping_rounds=20, n_splits=10, shuffle=True,seed=0):
    # 学習データを読み込む
    train, _ = load_datasets()
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)

    use_columns = get_use_columns(skip_columns=skip_columns)
    target = load_json("target_name")
    results = []

    # クロスバリデーション
    print("cross validation")
    for i, idx in enumerate(kf.split(train)):
        # 学習データを学習用とバリデーション用データに分割
        train_idx, validation_idx = idx
        x_train, y_train = train.iloc[train_idx][use_columns], train.iloc[train_idx][target]
        x_validation, y_validation = train.iloc[validation_idx][use_columns], train.iloc[validation_idx][target]

        # XGBoost用にDMatrixを作成
        dtrain = xgboost.DMatrix(x_train, label=y_train, feature_names=use_columns)
        dvalidation = xgboost.DMatrix(x_validation, label=y_validation, feature_names=use_columns)

        # 学習
        watch_list = [(dtrain, "train"), (dvalidation, "eval")]
        model = xgboost.train(load_json("parameters"), dtrain,
                              num_boost_round=num_boost_round, early_stopping_rounds=early_stopping_rounds,
                              evals=watch_list, verbose_eval=False)

        # 予測を求め評価値を計算
        prediction = predict_(model, dvalidation)
        result = evaluation_function(y_validation.to_list(), prediction)
        print(f"{i}:", result)
        results.append(result)

    evaluation_value = np.average(results)
    print("average:", evaluation_value)

    save_evaluation(output_file, evaluation_value, use_columns)
    return evaluation_value
