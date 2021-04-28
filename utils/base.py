from abc import ABCMeta, abstractmethod
from contextlib import contextmanager
import inspect
import pickle
from utils.utils import *


@contextmanager
def timer(name):
    t0 = time.time()
    print(f'[{name}] start')
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')


class Model(metaclass=ABCMeta):
    def __init__(self):
        self.model = None

    @abstractmethod
    def train(self, train_x, train_y, validation_x=None, validation_y=None, log=10):
        raise NotImplementedError

    @abstractmethod
    def predict(self, x):
        raise NotImplementedError

    def load_model(self, model_name):
        with open(load_path("models_path") / add_pkl(model_name), "rb") as f:
            self.model = pickle.load(f)
        print("Model loaded.")

    def save_model(self, model_name):
        assert self.model is not None, "Model isn't defined."
        with open(load_path("models_path") / add_pkl(model_name), "wb") as f:
            pickle.dump(self.model, f)
        print("Model saved.")


class Feature(metaclass=ABCMeta):
    prefix = ''
    suffix = ''
    dir = '../scripts'

    def __init__(self):
        self.name = self.__class__.__name__
        self.all = pd.DataFrame()
        preprocessed_file = load_json("preprocessed_file")
        self.all_path = Path(self.dir) / f'{self.name}_{preprocessed_file}.ftr'
        self.use_columns = []
        self.categorical_features = []

    def run(self, df):
        # create_features()で特徴量抽出を行い、timerでその計算時間を計測
        with timer(self.name):
            self.create_features(df)
            prefix = self.prefix + '_' if self.prefix else ''
            suffix = '_' + self.suffix if self.suffix else ''
            self.all.columns = prefix + self.all.columns + suffix
        return self

    @abstractmethod
    def create_features(self, df):
        # 特徴量抽出の処理を書く
        raise NotImplementedError

    def save(self):
        # allをfeatherファイルで保存し、特徴量クラス名と学習に使用するカラム名を保存
        self.all[self.use_columns].to_feather(str(self.all_path))
        assert len(self.use_columns) != 0, "using columns aren't set"
        add_json("features", {self.name:self.use_columns})
        add_json("categorical_features", self.categorical_features)
        add_json("dtypes", {column:str(self.all[column].dtype) for column in self.all.columns})


def get_features(namespace):
    # Featureを継承しており、抽象的でないクラスのインスタンスを返す
    for k, v in namespace.items():
        if inspect.isclass(v) and issubclass(v, Feature) and not inspect.isabstract(v):
            yield v()


def generate_features(namespace, dataframe, overwrite):
    if overwrite:
        # default.jsonのfeaturesとuse_columnsを初期化
        clear_json(["features", "categorical_features", "dtypes"], [dict, list, dict])

    for f in get_features(namespace):
        # Featureを継承したクラスに対して処理を行う
        # Featureを継承したクラスはfeature_engineering.pyに書く
        if f.all_path.exists() and not overwrite:
            print(f.name, 'was skipped')
        else:
            f.run(dataframe).save()
