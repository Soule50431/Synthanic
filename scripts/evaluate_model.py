import shap

from utils import *

shap.initjs()

train, test = load_datasets(suffix="ftr")

# XGBoost用のDMatrixを作成
use_columns = get_use_columns()
target = load_json("target_name")
x = train[use_columns]
y = train[target]
pseudo_label = pd.read_csv(load_path("outputs_path") / "logistic_regression_2021-04-27_21-51-24.csv")
x = pd.concat([x, test[use_columns]])
y = pd.concat([y, pseudo_label[target]])

with open("D:\\Documents\\Kaggle\\Synthanic\\models\\lightgbm_pseudo_2021-04-28_16-20-20.pkl", "rb") as f:
    model = pickle.load(f)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(x)  # X_trainは訓練データのpandas.DataFrame

shap.force_plot(explainer.expected_value, shap_values[0, :], x.iloc[0, :])