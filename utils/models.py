import lightgbm as lgb
import xgboost

from utils.utils import *


def xgboost_train(train_x, train_y, use_columns, validation_x=None, validation_y=None, log=10):
    get_use_columns()
    dtrain = xgboost.DMatrix(train_x, label=train_y, feature_names=use_columns)
    watch_list = [(dtrain, "train")]

    if validation_x is not None and validation_y is not None:
        devaluation = xgboost.DMatrix(validation_x, label=validation_y, feature_names=use_columns)
        watch_list.append((devaluation, "eval"))
    # 学習

    model = xgboost.train(load_json("parameters"), dtrain,
                          num_boost_round=1000, early_stopping_rounds=10,
                          evals=watch_list, verbose_eval=10)
    return model


def lightgbm_train(train_x, train_y, validation_x=None, validation_y=None, log=10):
    categorical_features = load_json("categorical_features")

    lgbtrain = lgb.Dataset(train_x, train_y, categorical_feature=categorical_features)
    valid_sets = [lgbtrain]

    params = load_json("parameters")
    if validation_x is not None and validation_y is not None:
        lgbevaluation = lgb.Dataset(validation_x, validation_y, categorical_feature=categorical_features)
        valid_sets.append(lgbevaluation)
        params["verbose"] = -1
        log = False
    else:
        valid_sets.append(lgbtrain)

    model = lgb.train(params, lgbtrain, valid_sets=valid_sets,
                     verbose_eval=log, num_boost_round=1000, early_stopping_rounds=10)
    return model
