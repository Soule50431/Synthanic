from utils import *
from validation import *

Feature.dir = load_path("features_path")


class Numerical(Feature):
    def create_features(self, df):
        self.all["Age"] = df["Age"]
        self.all["Fare"] = df["Fare"]
        self.use_columns = ["Age", "Fare"]


class Relative(Feature):
    def create_features(self, df):
        self.all["SibSp"] = df["SibSp"]
        self.all["Parch"] = df["Parch"]
        self.all["have_SibSp"] = (df["SibSp"] > 0).astype(int)
        self.all["have_Parch"] = (df["Parch"] > 0).astype(int)
        self.use_columns = ["SibSp", "Parch", "have_SibSp", "have_Parch"]


class FamilySize(Feature):
    def create_features(self, df):
        self.all["FamilySize"] = df["SibSp"] + df["Parch"] + 1
        self.use_columns = ["FamilySize"]


class IsAlone(Feature):
    def create_features(self, df):
        self.all["FamilySize"] = df["SibSp"] + df["Parch"] + 1
        self.all["isAlone"] = (self.all["FamilySize"] == 0).astype(int)
        self.use_columns = ["isAlone"]


class Cabin(Feature):
    def create_features(self, df):
        df["Cabin"] =df["Cabin"].map(lambda x: x[0])
        df["Cabin"] = df["Cabin"].map(lambda x: "AD" if x == "D" or x == "A" else "Other")
        self.all = pd.concat([self.all, pd.get_dummies(df["Cabin"], prefix="Cabin")])
        self.use_columns.extend(self.all.columns)


class Ticket(Feature):
    def create_features(self, df):
        df["Ticket"] = df["Ticket"].map(lambda x: str(x).split()[0] if len(str(x).split()) > 1 else "X")
        df["Ticket"] = df["Ticket"].map(lambda x: "PPC" if x == "PC" or x == "PP" else "Other")
        self.all = pd.concat([self.all, pd.get_dummies(df["Ticket"], prefix="Ticket")])
        self.use_columns.extend(self.all.columns)


class Category(Feature):
    def create_features(self, df):
        self.all["Embarked"] = df["Embarked"].fillna("S").map({"S":0, "Q":1, "C":2}).astype(int)
        self.all["Sex"] = df["Sex"].map({"male":0,"female":1}).astype(int)
        self.all["Pclass"] = df["Pclass"].astype(int)
        self.categorical_features = ["Embarked", "Sex", "Pclass"]
        self.use_columns = ["Embarked", "Sex", "Pclass"]


class IdTarget(Feature):
    def create_features(self, df):
        id_name = load_json("ID_name")
        target = load_json("target_name")

        self.all[id_name] = df[id_name]
        self.all[target] = df[target]
        self.use_columns.extend([id_name,target])


class Stacking(Feature):
    def create_features(self, df):
        prediction_x, prediction_y = cross_validate(None, training_algorithm=LightGBM,
                                      evaluation_function=accuracy_score)
        temp =[]
        temp.extend(prediction_x)
        temp.extend(prediction_y)
        self.all["lgbm"] = temp
        self.use_columns = ["lgbm"]


def feature_engineer(overwrite, suffix="ftr"):
    # boolean : overwrite ????????????????????????????????????????????????????????????
    assert suffix == "csv" or suffix == "ftr", "suffix must be csv or ftr"
    if suffix == "csv":
        df = pd.read_csv(load_path("inputs_path") / "train_test.csv")
    elif suffix == "ftr":
        df = pd.read_feather(load_path("inputs_path")/"train_test.ftr")
    generate_features(globals(), df, overwrite)
