from utils import *
from sklearn.model_selection import StratifiedKFold
import numpy as np


def cross_validate(output_file, training_algorithm, evaluation_function, suffix="ftr", skip_features=[],
                   num_boost_round=100, early_stopping_rounds=20, n_splits=10, shuffle=True, seed=0):
    # 学習データを読み込む
    train, test = load_datasets(suffix)
    kf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)

    use_columns = get_use_columns(skip_features=skip_features)
    target = load_json("target_name")
    results = []
    predictions_x = []
    predictions_y = []
    for column in train[use_columns].columns:
        print(column)

    # クロスバリデーション
    print("cross validation")
    for i, idx in enumerate(kf.split(train[use_columns], train[target])):
        # 学習データを学習用とバリデーション用データに分割
        train_idx, validation_idx = idx
        x_train, y_train = train.iloc[train_idx][use_columns], train.iloc[train_idx][target]
        x_validation, y_validation = train.iloc[validation_idx][use_columns], train.iloc[validation_idx][target]

        # モデルで学習
        algorithm = training_algorithm()
        algorithm.train(x_train, y_train, x_validation, y_validation)

        # 予測を求め評価値を計算
        prediction_x = algorithm.predict(x_validation)
        prediction_y = algorithm.predict(test[use_columns])
        predictions_x.extend(prediction_x)
        predictions_y.append(prediction_y)

        result = evaluation_function(y_validation.to_list(), [1 if x > 0.5 else 0 for x in prediction_x])
        print(f"{i}:", result)
        results.append(result)

    print("results:", results)
    evaluation_value = np.average(results)
    print("average:", evaluation_value)

    # save_evaluation(output_file, evaluation_value, use_columns)
    return predictions_x, np.average(predictions_y, axis=0)
