import numpy as np
from utils import *
from predict import predict_
from sklearn.model_selection import KFold, StratifiedKFold
import xgboost


def cross_validate(output_file, training_algorithm, evaluation_function, suffix="ftr", skip_features=[],
                   num_boost_round=100, early_stopping_rounds=20, n_splits=10, shuffle=True,seed=0):
    # 学習データを読み込む
    train, _ = load_datasets(suffix)
    kf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)

    use_columns = get_use_columns(skip_features=skip_features)
    target = load_json("target_name")
    results = []
    for column in train[use_columns].columns:
        print(column)
    # クロスバリデーション
    print("cross validation")
    for i, idx in enumerate(kf.split(train[use_columns],train[target])):
        # 学習データを学習用とバリデーション用データに分割
        train_idx, validation_idx = idx
        x_train, y_train = train.iloc[train_idx][use_columns], train.iloc[train_idx][target]
        x_validation, y_validation = train.iloc[validation_idx][use_columns], train.iloc[validation_idx][target]

        # XGBoost用にDMatrixを作成
        # dtrain = xgboost.DMatrix(x_train, label=y_train, feature_names=use_columns)
        dvalidation = xgboost.DMatrix(x_validation, label=y_validation, feature_names=use_columns)
        #
        # # 学習
        # watch_list = [(dtrain, "train"), (dvalidation, "eval")]
        # model = xgboost.train(load_json("parameters"), dtrain,
        #                       num_boost_round=num_boost_round, early_stopping_rounds=early_stopping_rounds,
        #                       evals=watch_list, verbose_eval=False)
        model = training_algorithm(x_train, y_train, x_validation, y_validation)
        # 予測を求め評価値を計算
        prediction = predict_(model, dvalidation)
        result = evaluation_function(y_validation.to_list(), prediction)
        print(f"{i}:", result)
        results.append(result)
    print("results:", results)
    evaluation_value = np.average(results)
    print("average:", evaluation_value)

    save_evaluation(output_file, evaluation_value, use_columns)
    return evaluation_value
