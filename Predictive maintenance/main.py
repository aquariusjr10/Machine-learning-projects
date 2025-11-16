import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
import numpy as np
# --- SKLearn Imports for Machine Learning ---
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
)
# --- Model Algorithms ---
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier


df = pd.read_csv("Dataset/predictive_maintenance_dataset.csv")
if df.empty:
    print("Dataset is empty")
else:
    print("Dataset loaded successfully")

# EDA
df.head()
df.info()
df.describe()
print("Number of null values:", df.isnull().sum())
print("Duplicates rows:", df.duplicated().sum())

# Data Cleaning
df.dropna(inplace=True)
print("Number of rows that have failure:", df[df["failure"] == 1].shape[0])

print("Dropping duplicates...")
df.drop_duplicates(inplace=True)
print("Duplicates rows after dropping:", df.duplicated().sum())


df["device_model"] = df["device"].apply(lambda x: x[:4])
df["device_model"].value_counts()


# Create a figure with 1 row and 2 columns
plt.figure(figsize=(12, 6))
# Plot the distribution of 'failure' with respect to 'device' for failure=1
plt.subplot(1, 2, 1)
sns.countplot(x="device_model", data=df.loc[df["failure"] == 1])
plt.title("Distribution of Failure (failure=1) with respect to Device")

# Plot the distribution of 'failure' with respect to 'device' for failure=0
plt.subplot(1, 2, 2)
sns.countplot(x="device_model", data=df.loc[df["failure"] == 0])
plt.title("Distribution of Failure (failure=0) with respect to Device")

# Adjust layout for better spacing
plt.tight_layout()

# Show the plots
plt.show()


# Create histograms to visualize the distribution of selected metrics with 'failure' as hue
plt.figure(figsize=(4 * 5, 2 * 5))
print("Distribution for failure is 0")
mask = df.failure > 0
for i, col in enumerate(
    [
        "metric1",
        "metric2",
        "metric3",
        "metric4",
        "metric5",
        "metric6",
        "metric7",
        "metric9",
    ]
):
    plt.subplot(2, 4, i + 1)
    sns.histplot(data=df.loc[mask], x=col, kde=True)
    plt.title(f"Distribution of {col}")
plt.tight_layout()
plt.show()

# Convert the 'date' column to datetime format
df["date"] = pd.to_datetime(df["date"])

# Extract and format the 'month' column for plotting
df["month"] = df["date"].dt.to_period("M")
df["month"] = df["month"].dt.strftime("%Y-%m")

# Create a line plot to visualize 'failure' over time by month
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x="month", y="failure")
plt.xticks(rotation=45)
plt.title("Failure over Time by Month")

# Select only numeric columns for the correlation matrix
numeric_cols = df.select_dtypes(include=[np.number])

# Compute the correlation matrix
correlation_matrix = numeric_cols.corr()

# Create a heatmap to visualize the correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix")


# --- Function to Train and Evaluate Multiple Models ---
def train_and_evaluate_models(df):
    """
    Trains and evaluates multiple ML models on the provided dataframe.
    """
    print("\n" + "=" * 50)
    print("Starting Model Training & Evaluation")
    print("=" * 50)

    # 1. Define Features (X) and Target (y)
    # We'll use the numeric metrics and the 'device_model' we created
    numeric_features = [
        "metric1", "metric2", "metric3", "metric4",
        "metric5", "metric6", "metric7", "metric9",
    ]
    categorical_features = ["device_model"]

    # Drop rows where these features might be NaN (if any slipped through)
    # and where failure is NaN
    df_model = df.dropna(subset=numeric_features + categorical_features + ["failure"])

    X = df_model[numeric_features + categorical_features]
    y = df_model["failure"]

    if X.empty or y.empty:
        print("No data left for modeling after dropping NaNs. Exiting.")
        return None, None

    # 2. Train-Test Split
    # We use stratify=y to ensure both train and test sets get a proportional
    # number of 'failure' samples, which is crucial for imbalanced datasets.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples.")
    print(f"Failure rate in training data: {y_train.mean():.2%}")
    print(f"Failure rate in test data: {y_test.mean():.2%}")

    # 3. Create Preprocessing Pipelines
    # Numeric features need to be scaled (important for LogReg and NN)
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

    # Categorical features need to be one-hot encoded
    categorical_transformer = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )

    # Use ColumnTransformer to apply different transformers to different columns
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # 4. Define Models to Train
    # We use class_weight='balanced' for models that support it
    # to help with the imbalanced nature of failure data.
    models = {
        "Logistic Regression": LogisticRegression(
            random_state=42, class_weight="balanced", max_iter=1000
        ),
        "Random Forest": RandomForestClassifier(
            random_state=42, class_weight="balanced"
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Neural Network (MLP)": MLPClassifier(
            random_state=42, max_iter=500, early_stopping=True
        ),
    }

    trained_models = {}
    results = {}

    # 5. Loop, Train, and Evaluate Each Model
    for name, model in models.items():
        # Create the full pipeline: Preprocess -> Classify
        clf = Pipeline(
            steps=[("preprocessor", preprocessor), ("classifier", model)]
        )

        print(f"\n--- Training {name} ---")
        clf.fit(X_train, y_train)

        # Make predictions
        y_pred = clf.predict(X_test)

        # Get prediction probabilities for ROC-AUC
        # (Handle models that might not have predict_proba)
        if hasattr(clf, "predict_proba"):
            y_pred_proba = clf.predict_proba(X_test)[:, 1]
        else:
            y_pred_proba = clf.decision_function(X_test)

        # Store the trained model
        trained_models[name] = clf

        # Evaluate
        print(f"\n--- Results for {name} ---")
        print(classification_report(y_test, y_pred, digits=4))

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        print("Confusion Matrix:")
        print(f"  True Negatives: {tn} (Correctly predicted no failure)")
        print(f" False Positives: {fp} (Incorrectly predicted failure)")
        print(f" False Negatives: {fn} (MISSED a real failure)")
        print(f"  True Positives: {tp} (Correctly predicted failure)")

        # Store metrics
        # We focus on F1, Recall, and Precision because accuracy
        # is misleading in imbalanced datasets.
        results[name] = {
            "f1": f1_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_pred_proba),
        }

    # 6. Compare Models
    results_df = pd.DataFrame(results).T.sort_values(by="f1", ascending=False)
    print("\n" + "=" * 50)
    print("Model Comparison (Sorted by F1-Score)")
    print("=" * 50)
    print(results_df)

    return trained_models, results_df


# --- Function to Predict Failure for a Single New Part ---
def predict_failure(trained_pipeline, single_data_point):
    """
    Uses a trained pipeline to predict failure for a single data point.

    :param trained_pipeline: A trained sklearn Pipeline object.
    :param single_data_point: A dictionary of features for one part.
    """
    # Convert the dictionary to a DataFrame (pipeline expects it)
    data_df = pd.DataFrame([single_data_point])

    # Predict
    prediction = trained_pipeline.predict(data_df)[0]

    # Predict probability
    prediction_proba = trained_pipeline.predict_proba(data_df)[0]

    failure_probability = prediction_proba[1]  # Probability of class '1' (failure)

    if prediction == 1:
        print(f"PREDICTION: \t ** FAILURE LIKELY **")
    else:
        print(f"PREDICTION: \t No Failure Likely")

    print(f"Confidence (Failure Prob): {failure_probability:.2%}")

    return prediction, failure_probability


# --- Main execution ---
if __name__ == "__main__":

    # (Your plotting code is skipped here for brevity in the console output)
    # (To show plots, uncomment the plt.show() lines from your original code)

    # --- 3. Run Model Training and Evaluation ---
    trained_models, results_df = train_and_evaluate_models(df)

    if trained_models:
        # --- 4. Example: Predict on a new part ---
        print("\n" + "=" * 50)
        print("Example Single Prediction")
        print("=" * 50)

        # Get the best model (first one in the sorted results_df)
        best_model_name = results_df.index[0]
        best_model_pipeline = trained_models[best_model_name]

        print(f"Using best model ({best_model_name}) for a sample prediction...")

        # A sample part matching the features
        # We'll invent one that looks like it might fail
        sample_part_data = {
            'metric1': 48467332,
            'metric2': 64776,
            'metric3': 0,
            'metric4': 900,
            'metric5': 8,
            'metric6': 39267,
            'metric7': 70,
            'metric8': 69,
            'metric9': 1,
            'device_model': 'sm-j'
        }

        predict_failure(best_model_pipeline, sample_part_data)

        # A sample part that looks healthy
        print("\n---")
        print("Predicting on a healthy-looking sample...")
        healthy_part_data = {
            'metric1': 225420172,
            'metric2': 60,
            'metric3': 0,
            'metric4': 58,
            'metric5': 2,
            'metric6': 404538,
            'metric7': 0,
            'metric8': 1,
            'metric9': 20,
            'device_model': 'sm-g'
        }
        predict_failure(best_model_pipeline, healthy_part_data)
