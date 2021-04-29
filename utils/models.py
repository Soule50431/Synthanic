import lightgbm as lgb
import xgboost
from sklearn import linear_model

from utils.base import *


class LogisticRegression(Model):
    def train(self, train_x, train_y, validation_x=None, validation_y=None, log=10):
        self.model = linear_model.LogisticRegression(penalty='l2', solver='sag', random_state=0)
        self.model.fit(train_x, train_y)

    def predict(self, x):
        prediction = self.model.predict(x)
        prediction = [1 if i > 0.5 else 0 for i in prediction]
        return prediction


class Xgboost(Model):
    def train(self, train_x, train_y, skip_features=[], validation_x=None, validation_y=None, log=10):
        use_columns = get_use_columns(skip_features=skip_features)
        dtrain = xgboost.DMatrix(train_x, label=train_y, feature_names=use_columns)
        watch_list = [(dtrain, "train")]

        if validation_x is not None and validation_y is not None:
            devaluation = xgboost.DMatrix(validation_x, label=validation_y, feature_names=use_columns)
            watch_list.append((devaluation, "eval"))
        # 学習

        self.model = xgboost.train(load_json("parameters"), dtrain, num_boost_round=1000, early_stopping_rounds=10,
                              evals=watch_list, verbose_eval=10)

    def predict(self, x, skip_features=[]):
        # 予測用のDMatrixを作成
        use_columns = get_use_columns(skip_features=skip_features)
        dtest = xgboost.DMatrix(x[use_columns], feature_names=use_columns)

        prediction = self.model.predict(dtest)
        prediction = [1 if i > 0.5 else 0 for i in prediction]
        return prediction


class LightGBM(Model):
    def train(self, train_x, train_y, validation_x=None, validation_y=None, log=10):
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

        self.model = lgb.train(params, lgbtrain, valid_sets=valid_sets,
                          verbose_eval=log, num_boost_round=2000, early_stopping_rounds=10)

    def predict(self, x):
        prediction = self.model.predict(x)
        # prediction = [1 if i > 0.5 else 0 for i in prediction]
        return prediction


