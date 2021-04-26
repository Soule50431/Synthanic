from base import Feature, generate_features
from utils import *

Feature.dir = load_path("features_path")


class Test(Feature):
    def create_features(self, df):
        self.all["Age"] = df["Age"]
        self.all["Fare"] = df["Fare"]
        self.all["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2}).astype(int)
        self.all["Sex"] = df["Sex"].map({"male": 0, "female":1}).astype(int)
        self.use_columns = ["Age", "Fare", "Embarked", "Sex"]


class IdTarget(Feature):
    def create_features(self, df):
        id_name = load_json("ID_name")
        target = load_json("target_name")

        self.all[id_name] = df[id_name]
        self.all[target] = df[target]
        self.use_columns.extend([id_name,target])


def feature_engineer(overwrite):
    # boolean : overwrite 既に計算した特徴量を計算しなおすかどうか
    df = pd.read_feather(load_path("inputs_path")/"train_test.ftr")
    generate_features(globals(), df, overwrite)
