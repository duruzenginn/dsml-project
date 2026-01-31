import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def build_text(df: pd.DataFrame) -> pd.Series:
    """Create the text feature used by the model (title + article)."""
    return df["title"].fillna("") + " " + df["article"].fillna("")


def main() -> None:
    # 1) Load labeled training data
    train_df = pd.read_csv("development.csv")
    X_train = build_text(train_df)
    y_train = train_df["label"]

    # 2) Final model (chosen hyperparameters)
    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.6,
            max_features=100000,
            strip_accents=None,
        )),
        ("clf", LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced"
        )),
    ])

    # 3) Train on all labeled data
    model.fit(X_train, y_train)

    # 4) Load evaluation data and build text
    eval_df = pd.read_csv("evaluation.csv")
    X_eval = build_text(eval_df)

    # 5) Predict and write submission
    preds = model.predict(X_eval)

    submission = pd.DataFrame({
        "Id": eval_df["Id"],
        "Predicted": preds
    })
    submission.to_csv("submission.csv", index=False)

    # 6) Minimal sanity prints
    print("✅ submission.csv created")
    print("Rows:", len(submission), "| Columns:", submission.columns.tolist())


if __name__ == "__main__":
    main()