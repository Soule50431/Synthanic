from utils.models import *


def train_model(output_file, training_algorithm, suffix="ftr", skip_features=[]):
    # 学習データ読み込み
    train, test = load_datasets(suffix)

    # XGBoost用のDMatrixを作成
    use_columns = get_use_columns(skip_features=skip_features)
    target = load_json("target_name")
    x = train[use_columns]
    y = train[target]
    pseudo_label = pd.read_csv(load_path("outputs_path")/"ensemble_soule_logistic80043_lgbm78630_udon_rogistic80051_2021-04-27_22-13-38.csv")
    x = pd.concat([x, test[use_columns]])
    y = pd.concat([y, pseudo_label[target]])

    algorithm = training_algorithm()
    algorithm.train(x, y)
    algorithm.save_model(output_file)
