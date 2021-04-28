import math
from utils.utils import *


def ensemble(file_names, output_file):
    submission = pd.read_csv(load_path("inputs_path")/"sample_submission.csv")
    target = load_json("target_name")

    outputs_path = load_path("outputs_path")
    submission[target] = 0
    for file_name in file_names:
        submission[target] += pd.read_csv(outputs_path/add_csv(file_name))[target]

    submission[target] = (submission[target] >= math.ceil(len(file_names) / 2)).astype(int)
    submission.to_csv(outputs_path/add_csv(output_file),index=False)
    print("ensembled")


if __name__ == "__main__":
    inputs = ["submission_LogisticRegression.csv",
                   "lightgbm_2021-04-27_22-04-25.csv",
                   "logistic_regression_2021-04-27_21-51-24.csv"]

    output = add_time("ensemble_soule_logistic80043_lgbm78630_udon_rogistic80051")
    ensemble(inputs, output)

