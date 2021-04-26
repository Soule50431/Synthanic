from preprocessing import *
from feature_engineering import *
from validation import *
from train_model import *
from predict import *
from utils.utils import *
from utils.evaluation_functions import *


input_files = ["train_data", "test_data"]
output_file = add_time("xgboost_baseline")
# skip_columns = ["IdTarget"]


if __name__ == "__main__":
    preprocess(overwrite=False)
    feature_engineer(overwrite=True)
    cross_validate(output_file, evaluation_function=accuracy_score)

    # train_model(output_file)
    # predict(output_file)
