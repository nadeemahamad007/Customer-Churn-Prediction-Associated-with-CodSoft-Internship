from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = ROOT_DIR / "data" / "Churn_Modelling.csv"
DEFAULT_MODEL_PATH = ROOT_DIR / "models" / "churn_model.pkl"

NUMERICAL_FEATURES = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
]

CATEGORICAL_FEATURES = ["Geography", "Gender"]
DROP_COLUMNS = ["RowNumber", "CustomerId", "Surname"]
TARGET_COLUMN = "Exited"


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def build_candidate_models() -> dict[str, object]:
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }


def load_dataset(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df.drop(columns=DROP_COLUMNS)


def evaluate_pipeline(name: str, pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    predictions = pipeline.predict(X_test)
    print(f"\n{name}")
    print(classification_report(y_test, predictions))


def train_and_save(data_path: Path, model_path: Path, test_size: float, n_jobs: int) -> None:
    df = load_dataset(data_path)
    X = df.drop(columns=TARGET_COLUMN)
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor()

    for name, model in build_candidate_models().items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )
        pipeline.fit(X_train, y_train)
        evaluate_pipeline(name, pipeline, X_test, y_test)

    tuned_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", RandomForestClassifier(random_state=42)),
        ]
    )

    param_grid = {
        "model__n_estimators": [50, 100, 200],
        "model__max_depth": [None, 10, 20, 30],
        "model__min_samples_split": [2, 5, 10],
    }

    search = GridSearchCV(
        estimator=tuned_pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=n_jobs,
    )
    search.fit(X_train, y_train)

    print("\nTuned Random Forest")
    print(f"Best parameters: {search.best_params_}")
    evaluate_pipeline("Classification report", search.best_estimator_, X_test, y_test)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(search.best_estimator_, model_path)
    print(f"\nSaved trained model to: {model_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a customer churn prediction model.")
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to the CSV dataset.",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path where the trained model will be saved.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.30,
        help="Proportion of data to reserve for testing.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel jobs for grid search. Use 1 for best compatibility.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_and_save(args.data, args.model_out, args.test_size, args.n_jobs)
