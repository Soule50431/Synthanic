from utils.models import *
from validation import cross_validate
from utils.evaluation_functions import *


def train_model(output_file, training_algorithm, suffix="ftr", skip_features=[]):
    # 学習データ読み込み
    train, test = load_datasets(suffix)

    use_columns = get_use_columns(skip_features=skip_features)
    target = load_json("target_name")
    x = train[use_columns]
    y = train[target]
    # pseudo_label = pd.read_csv(load_path("outputs_path")/"logistic_regression_2021-04-27_21-51-24.csv")
    # x = pd.concat([x, test[use_columns]])
    # y = pd.concat([y, pseudo_label[target]])

    algorithm = training_algorithm()
    algorithm.train(x, y)
    algorithm.save_model(output_file)
