from base import *

input_path = load_path("inputs_path")


def _preprocess(df):
    # 前処理の具体的な処理
    delete_columns = ["Name"]
    df.drop(delete_columns, axis=1, inplace=True)
    df["Age"].fillna(df["Age"].mean(), inplace=True)
    df["Fare"].fillna(df["Fare"].median(), inplace=True)
    df["Embarked"].fillna("S", inplace=True)
    df["Cabin"].fillna("X", inplace=True)
    df["Ticket"].fillna("X", inplace=True)


def preprocess(overwrite, suffix="ftr"):
    # file_names : 入力ファイル名のList 拡張子は付いていても付いていなくてもよい
    # boolean overwrite : ファイルの上書きをするか
    assert suffix == "csv" or suffix == "ftr"
    if suffix == "csv":
        preprocessed_path = input_path/add_csv(load_json("preprocessed_file"))
    elif suffix == "ftr":
        preprocessed_path = input_path/add_feather(load_json("preprocessed_file"))

    if (not overwrite) and preprocessed_path.exists:
        # 上書きしない時、かつ前処理を行ったふぃあるが既に存在するときは処理をスキップ
        print("preprocessing was skipped")
        return

    size = [0]  # 学習データとテストデータの長さを格納
    file_names = load_json("input_files")
    # 学習データとテストデータをdfに結合
    df = pd.DataFrame()
    for i, file_name in enumerate(file_names):
        file_name = add_csv(file_name)
        temp = pd.read_csv(input_path/file_name)
        print(f"read {file_name}")
        size.append(size[i]+len(temp))
        df = pd.concat([df, temp])
    add_json("size", size, overwrite=True)

    # 前処理
    _preprocess(df)

    # 上書きする場合か、前処理を行ったファイルが存在しない時に、
    # dfをtrain_test.ftrに保存
    if overwrite or not preprocessed_path.exists():
        df.reset_index(inplace=True)
        if suffix == "csv":
            df.to_csv(preprocessed_path)
            file_name = add_csv(load_json("preprocessed_file"))
        elif suffix == "ftr":
            df.to_feather(preprocessed_path)
            file_name = add_feather(load_json("preprocessed_file"))
        print(f"saved to {file_name}")
