from preprocessing import *
from feature_engineering import *
from validation import *
from train_model import *
from predict import *
from utils.utils import *
from utils.evaluation_functions import *
from utils.models import *


output_file = add_time("lightgbm")
# skip_columns = ["IdTarget"]


if __name__ == "__main__":
    preprocess(overwrite=False)
    feature_engineer(overwrite=True)
    cross_validate(output_file, training_algorithm=xgboost_train, evaluation_function=accuracy_score)

    # train_model(output_file, training_algorithm=lightgbm_train)
    # predict(output_file)
