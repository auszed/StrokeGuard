from typing import List, Tuple, Any
import pandas as pd
import numpy as np
import joblib

# importing all the required ML packages
from sklearn.base import BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.pipeline import make_pipeline

# model importation
# ensemble_model_001 = joblib.load('models/ensemble_model_002.pkl')
GNB_002 = joblib.load('models/GNB_002.pkl')

def preprocess_pipeline(dataset: pd.DataFrame, num_features: List, binary_cols: List,
                        categorical_cols: List, numeric_preprocess: str = "StandardScaler",
                        strategy_imputer: str = "most_frequent") -> Tuple[pd.DataFrame, np.ndarray]:
    """Create a pipeline with selectable model and preprocessing steps."""

    # preprocess the pipelines
    numerical_preprocess = Pipeline(
        steps=[
            ("imputation_mode", SimpleImputer(missing_values=np.nan, strategy=strategy_imputer)),
            ("scaler", StandardScaler())
        ],
    )
    categorical_binary_preprocess = Pipeline(
        steps=[
            ("imputation_constant", SimpleImputer(fill_value="missing", strategy="constant")),
            ("binary_imputer", OrdinalEncoder(unknown_value=None))
        ]
    )
    categorical_variables_preprocess = Pipeline(
        steps=[
            ("imputation_constant", SimpleImputer(fill_value="missing", strategy="constant")),
            ("binary_imputer", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    # combine the pipelines
    preprocess = ColumnTransformer(
        [
            ("numerical_preprocess", numerical_preprocess, num_features),
            ("categorical_binary_preprocess", categorical_binary_preprocess, binary_cols),
            ("categorical_variables_preprocess", categorical_variables_preprocess, categorical_cols)
        ]
    )

    pre_process_pipeline = make_pipeline(preprocess)
    numpy_dataset = pre_process_pipeline.fit_transform(dataset)

    pre_process_pipeline.fit(dataset)
    num_normalize_transformed_cols = ["normalize" + '_' + item for item in num_features]
    bin_transformed_cols = binary_cols
    cat_transformed_cols = preprocess.named_transformers_['categorical_variables_preprocess'].named_steps[
        'binary_imputer'].get_feature_names_out(categorical_cols)
    transformed_column_names = np.concatenate([num_features, bin_transformed_cols, cat_transformed_cols])

    # we transformed in a pandas but before was in a numpy array
    transformed_df = pd.DataFrame(numpy_dataset, columns=transformed_column_names)

    return transformed_df, numpy_dataset


def predict_or_analyze(data: pd.DataFrame, model_selection:str) -> tuple:
    """This method will help to predict the values and the possibilities of having a stroke."""

    # Sample data that has all the variables so that we process it in the pipeline works fine.
    sample_data = {
        'gender': ['Male', 'Female', 'Other', 'Other', 'Other'],
        'age': [0, 40, 50, 80, 100],
        'hypertension': [1, 0, 1, 0, 1],
        'heart_disease': [1, 0, 1, 0, 0],
        'ever_married': ['Yes', 'No', 'Yes', 'No', 'No'],
        'work_type': ['Private', 'Self-employed', 'Govt_job', 'Children', 'Never_worked'],
        'Residence_type': ['Urban', 'Rural', 'Urban', 'Rural', 'Urban'],
        'avg_glucose_level': [50, 100, 150, 200, 300],
        'bmi': [10, 20, 50, 80, 100],
        'smoking_status': ['never smoked', 'formerly smoked', 'smokes', 'Unknown', 'Unknown']
    }
    sample_data_all = pd.DataFrame(sample_data)
    df_concat = pd.concat([data, sample_data_all], ignore_index=True)

    # Pipeline transformation
    dataset, np_array = preprocess_pipeline(
        dataset=df_concat,
        num_features=["age", "avg_glucose_level", "bmi"],
        binary_cols=["ever_married", "hypertension", "heart_disease"],
        categorical_cols=["work_type", "Residence_type", "smoking_status", "gender"],
        numeric_preprocess="StandardScaler",
        strategy_imputer="most_frequent"
    )

    # extract the value from the pipeline
    data_for_prediction = dataset.iloc[0]
    data_for_prediction_array = data_for_prediction.values.reshape(1, -1)

    # Predictions
    if model_selection == "GNB":

        value = GNB_002.predict(data_for_prediction_array)
        value_probabilities = GNB_002.predict_proba(data_for_prediction_array)

    # if model_selection == "Ensemble_model":
    #     value = ensemble_model_001.predict(data_for_prediction_array)
    #     value_probabilities = ensemble_model_001.predict_proba(data_for_prediction_array)

    return value, value_probabilities


