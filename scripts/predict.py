from utils.utils import *


def _write_submission(output_file, prediction, test):
    submission = pd.DataFrame()
    submission[load_json("ID_name")] = test[load_json("ID_name")]
    submission[load_json("target_name")] = prediction
    submission.to_csv(load_path("outputs_path")/add_csv(output_file), index=False)
    print("write to csv")


def predict(output_file, training_algorithm, skip_features=[], suffix="ftr"):
    # テストデータの読み込み
    _, test = load_datasets(suffix)
    use_columns = get_use_columns(skip_features)
    # モデルの読み込み
    algorithm = training_algorithm()
    algorithm.load_model(output_file)

    # 予測
    prediction = algorithm.predict(test[use_columns])

    # 提出ファイル作成
    _write_submission(output_file, prediction, test)
