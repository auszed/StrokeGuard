from typing import List, Tuple, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import qqplot
import seaborn as sns
import scipy.stats as stats
from sklearn.model_selection import train_test_split

# importing all the required ML packages
from sklearn.base import BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.pipeline import make_pipeline
from sklearn.metrics import recall_score, precision_recall_curve, confusion_matrix, classification_report, f1_score

# models baseline
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import LinearSVC

# color setup
custom_colors = ['#36CE8A', "#7436F5", "#3736F4", "#36AEF5", "#B336F5", "#f8165e", "#36709A", "#3672F5", "#7ACE5D"]
color_palette_custom = sns.set_palette(custom_colors)
theme_color = sns.color_palette(color_palette_custom, 9)

"""
functions for statistics
"""


def chi_square_test_results(dataset: pd.DataFrame, independent_var: str, dependent_var: str, alpha: float = 0.05) -> (
        Tuple):
    """Chi square test results with different parameters that we had as an exit"""
    proportion = pd.crosstab(dataset[independent_var], dataset[dependent_var])
    chi2, p, dof, expected = stats.chi2_contingency(proportion)
    expected_vals = pd.DataFrame(expected)

    if p < alpha:
        result = (f"Reject the null hypothesis. "
                  f"There is an association between {dependent_var} and {independent_var}, so they are independent")
    else:
        result = (f"Fail to reject the null hypothesis. "
                  f"There is no significant relationship between {dependent_var} and {independent_var}, so they are dependent")

    return proportion, expected_vals, chi2, p, dof, result


def qq_plot_verification_groups(dataset_a: pd.DataFrame, dataset_b: pd.DataFrame, independent_var: str,
                                dependent_var: str) -> None:
    """QQ Plot verification groups plot"""
    # Create a figure and two subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Q-Q plot for Stroke Group
    qqplot(dataset_a, line='s', ax=axs[0])
    axs[0].set_title(f'Q-Q plot for {independent_var} of {dependent_var} Group')

    # Q-Q plot for No-Stroke Group
    qqplot(dataset_b, line='s', ax=axs[1])
    axs[1].set_title(f'Q-Q plot for {independent_var} of No-{dependent_var} Group')

    # Adjust layout
    plt.tight_layout()
    plt.show()


def dependent_variables_test(dataset_a: pd.DataFrame, dataset_b: pd.DataFrame, alpha: float = 0.05) -> None:
    """Test dependent variables to validate the dependency using different tests"""

    # Shapiro-Wilk test
    shapiro_A = stats.shapiro(dataset_a)
    shapiro_B = stats.shapiro(dataset_b)

    # Levene's Test
    levene_test = stats.levene(dataset_a, dataset_b)

    print("==========================")
    print("=====Shapiro-Wilk test=====")

    if shapiro_A[1] > alpha and shapiro_B[1] > alpha:
        print(
            f'Shapiro-Wilk test for dataset A: p-value = {shapiro_A[1]:.4f}, dataset B: p-value = {shapiro_B[1]:.4f},'
            f'suggests that the data is normally distributed')
        shapiro_validation = True
    else:
        print(
            f"Shapiro-Wilk test for dataset A: p-value = {shapiro_A[1]:.4f}, dataset B: p-value = {shapiro_B[1]:.4f},"
            f" implies that it's not normally distributed")
        shapiro_validation = False

    print("=====Levene’s test=====")
    if levene_test[1] > alpha:
        print(
            f'Levene’s test for equality of variances: p-value= {levene_test[1]:.4f}, suggests homogeneity in variance,'
            f' so no significant difference across the groups being compared')
        levene_validation = True
    else:
        print(
            f'Levene’s test for inequality of variances: p-value= {levene_test[1]:.4f}, there is heterogeneity meaning '
            f'a significant difference across the groups being compared')
        levene_validation = False

    if shapiro_validation and levene_validation:
        t_stat, p_value = stats.ttest_ind(dataset_a, dataset_b, equal_var=True)
        print("=====T-Test=====")
        print(f't-statistic: {t_stat}, p-value: {p_value}')

    elif shapiro_validation == False and levene_validation == False:
        t_stat, p_value = stats.ttest_ind(dataset_a, dataset_b, equal_var=True)
        print(f'=====Mann-Whitney U test=====')
        print(f't-statistic: {t_stat}, p-value: {p_value}')

    else:
        t_stat, p_value = stats.ttest_ind(dataset_a, dataset_b, equal_var=False)
        print("=====Welch's t-test=====")
        print(f"Welch's t-test: t-statistic = {t_stat}, p-value = {p_value}")

    print("==========================")


"""
Drawings for different plots
"""


def plot_roc_curve(fpr: List[float], tpr: List[float], roc_auc: float) -> None:
    """Plot the ROC curve"""
    plt.figure(figsize=(15, 6))
    plt.plot(fpr, tpr, color='red', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()


def conf_heatmap(conf_matrix: np.ndarray) -> None:
    """Heatmap of confusion matrix"""
    plt.figure(figsize=(15, 2))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Predicted Positive","Predicted Negative"],
                yticklabels=["Actual Positive","Actual Negative" ])
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix')
    plt.show()


def thresholds_recall_curve(y_prediction: np.ndarray, y_true: np.ndarray, rearrange_intersection: float= 0.50) -> None:
    """Adding the comparator between threshold and the recall precision and f1 score curve"""

    # Predicting probabilities instead of classes
    f1_scores = []
    precision, recall, thresholds = precision_recall_curve(y_true, y_prediction)

    for threshold in thresholds:
        predicted_labels = (y_prediction >= threshold).astype(int)
        f1 = f1_score(y_true, predicted_labels)
        f1_scores.append(f1)

    # Find the intersection point
    precision_recall_diff = np.abs(precision[:-1] - recall[:-1])
    intersection_index = np.argmin(precision_recall_diff)
    intersection_threshold = thresholds[intersection_index]
    intersection_precision = precision[intersection_index]
    intersection_recall = recall[intersection_index]

    # Plot the chart with precision, recall, and F1 scores against thresholds
    plt.figure(figsize=(15, 6))
    plt.plot(thresholds, precision[:-1], color=theme_color[0], alpha=0.8, linewidth=2, label="Precision")
    plt.plot(thresholds, recall[:-1], color=theme_color[3], alpha=0.8, linewidth=2, label="Recall")
    plt.plot(thresholds, f1_scores, color=theme_color[5], alpha=0.8, linewidth=2, label="F1 Score")

    # Add vertical line at the intersection point
    plt.axvline(x=intersection_threshold, color='yellow', linestyle='--', linewidth=2, label="Intersection Point")

    # Move the text units above the intersection point
    text_y_position = max(intersection_precision, intersection_recall) + rearrange_intersection
    plt.text(intersection_threshold, text_y_position,
             f' Intersection: {intersection_threshold:.2f}',
             horizontalalignment='left',
             verticalalignment='bottom',
             color='yellow',
             fontsize=12)

    plt.title('Precision-Recall Curve')
    plt.xlabel('Thresholds')
    plt.ylabel('Precision, Recall and F1 Score')
    plt.grid(True)
    plt.legend()
    plt.show()


"""
functions for the modeling
"""


def split_dataset(dependent_variable: List, independent_variable: List, split_train_validation: float = 0.3,
                  split_validation_test: float = 0.5, state_reproducibility: int = 100) -> Tuple:
    """Split the dataset independently for training, validation and test"""

    # Split data into training and the rest (validation and test combined)
    X_train, X_vt, y_train, y_vt = train_test_split(independent_variable, dependent_variable,
                                                    test_size=split_train_validation,
                                                    random_state=state_reproducibility)

    # Split X_vt, y_vt into validation and test sets
    X_validation, X_test, y_validation, y_test = train_test_split(X_vt, y_vt, test_size=split_validation_test,
                                                                  random_state=state_reproducibility)

    return X_train, X_validation, X_test, y_train, y_validation, y_test


# List of models with class_weight parameter
def baseline_models(independent_x: pd.DataFrame, dependent_y:np.ndarray, independent_x_prediction: pd.DataFrame,
                    independent_y_prediction: np.ndarray)->None:
    models_with_class_weight = {
        'LogisticRegression': LogisticRegression(class_weight='balanced', random_state=42),
        'RidgeClassifier': RidgeClassifier(class_weight='balanced'),
        'SGDClassifier': SGDClassifier(class_weight='balanced', random_state=42),
        'LinearSVC': LinearSVC(class_weight='balanced', random_state=42),
        'DecisionTreeClassifier': DecisionTreeClassifier(class_weight='balanced', random_state=42),
        'RandomForestClassifier': RandomForestClassifier(class_weight='balanced', random_state=42),
        'ExtraTreesClassifier': ExtraTreesClassifier(class_weight='balanced', random_state=42)
    }
    # train models
    for name, model in models_with_class_weight.items():
        model.fit(independent_x, dependent_y)
        y_pred = model.predict(independent_x_prediction)
        print(f'{name}:\n', classification_report(independent_y_prediction, y_pred))


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


def predict_recall_test_values(model: Any, X_val: pd.DataFrame, y_val: np.ndarray, X_test: pd.DataFrame,
                               y_test: np.ndarray, Model_Name:str) -> pd.DataFrame:
    """Predict the values from the test values"""
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)
    recall_val = recall_score(y_val, y_pred_val)
    recall_test = recall_score(y_test, y_pred_test)
    data = {
        'Model': [str(Model_Name)],
        'Recall_Validation': [recall_val],
        'Recall_Test': [recall_test]
    }
    data_df = pd.DataFrame(data)
    return data_df


def threshold_values_change(threshold: list, predicted_probabilities: list, y_validation: list,
                            value_above_threshold: int = 1, value_below_threshold: int = 0) -> pd.DataFrame:
    """Extract a table with the best thresholds and probabilities for each model"""
    new_dataset = pd.DataFrame({
        "thresholds": [],
        "precision": [],
        "recall": [],
        "false_negative": [],
        "true_negative": [],
        "false_positive": [],
        "true_positive": []
    })

    # loop for verify the best metric in each threshold
    for tresh in threshold:
        probabilities = []
        for value in predicted_probabilities:
            if value >= tresh:
                probabilities.append(value_above_threshold)
            else:
                probabilities.append(value_below_threshold)

        # create the confusion matrix to extract the values
        conf_recall = confusion_matrix(y_validation, probabilities)

        # preparation if we got an error
        precision_denominator = (conf_recall[1][0] + conf_recall[1][1])
        recall_denominator = (conf_recall[0][1] + conf_recall[1][1])
        if precision_denominator == 0:
            precision = 0
        else:
            precision = conf_recall[1][1] / precision_denominator
        if recall_denominator == 0:
            recall = 0
        else:
            recall = conf_recall[1][1] / recall_denominator

        false_negative = conf_recall[0][1]
        true_negative = conf_recall[1][1]
        false_positive = conf_recall[0][0]
        true_positive = conf_recall[1][0]
        new_row = pd.DataFrame({
            "thresholds": threshold,
            "precision": precision,
            "recall": recall,
            "false_negative": false_negative,
            "true_negative": true_negative,
            "false_positive": false_positive,
            "true_positive": true_positive
        })
        new_dataset = pd.concat([new_dataset, new_row], ignore_index=True)

    return new_dataset


def classification_report_summary(model: BaseEstimator, X_train: pd.DataFrame, y_train: pd.Series,
                                  X_validation: pd.DataFrame, y_validation: pd.Series) -> tuple:
    """Adding a summary transformation for the classification report"""
    train_report = classification_report(y_train, model.predict(X_train), output_dict=True, labels=[1, 0],
                                         target_names=["stroke", "no stroke"])
    validation_report = classification_report(y_validation, model.predict(X_validation), output_dict=True,
                                              labels=[1, 0], target_names=["stroke", "no stroke"])
    train_report = pd.DataFrame(train_report)
    validation_report = pd.DataFrame(validation_report)

    # print both dataset
    return train_report, validation_report


"""
Interpretability of the models
"""


def fail_predictions_df(model: BaseEstimator, x_val: pd.DataFrame, y_val: np.ndarray, extract_rows_examples: int=5) -> tuple:
    """This method help us to extract the columns from the values that the model fails to predict"""

    # conversions and extracting the rows of the values that predict on each row
    y_pred = model.predict(x_val)
    correct_predictions = (y_pred == y_val)
    prediction_df = x_val.copy()
    prediction_df['correct_predictions'] = correct_predictions
    prediction_df['predicted_values'] = y_pred
    prediction_df['real_values'] = y_val
    prediction_df = prediction_df[prediction_df['correct_predictions'] == False]

    # extract the values the recall values
    prediction_positive = prediction_df[prediction_df['real_values'] == 1]
    prediction_negative = prediction_df[prediction_df['real_values'] == 0]

    prediction_positive = prediction_positive.drop(columns=['correct_predictions', 'predicted_values', 'real_values'])
    prediction_negative = prediction_negative.drop(columns=['correct_predictions', 'predicted_values', 'real_values'])

    prediction_positive = prediction_positive.head(extract_rows_examples)
    prediction_negative = prediction_negative.head(extract_rows_examples)

    return prediction_positive, prediction_negative
