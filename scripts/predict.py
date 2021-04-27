import xgboost
import pickle

from utils.utils import *


def predict_(model, dtest):
    prediction = model.predict(dtest)
    prediction = [1 if i > 0.5 else 0 for i in prediction]
    return prediction


def _write_submission(output_file, prediction, test):
    submission = pd.DataFrame()
    submission[load_json("ID_name")] = test[load_json("ID_name")]
    submission[load_json("target_name")] = prediction
    submission.to_csv(load_path("outputs_path")/add_csv(output_file), index=False)
    print("write to csv")


def predict(output_file, skip_features=[], suffix="ftr"):
    # テストデータの読み込み
    _, test = load_datasets(suffix)
    # モデルの読み込み
    with open(load_path("models_path")/add_pkl(output_file), "rb") as f:
        model = pickle.load(f)

    # 予測用のDMatrixを作成
    use_columns = get_use_columns(skip_features=skip_features)
    x = test[use_columns]
    # dtest = xgboost.DMatrix(x, feature_names=use_columns)

    # 予測
    prediction = predict_(model, x)

    # 提出ファイル作成
    _write_submission(output_file, prediction, test)
