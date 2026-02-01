import pandas as pd
import numpy as np
import time
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def build_text(df: pd.DataFrame) -> pd.Series:
    """
    Build text using domain knowledge:
    source + title + article
    """
    return (
        df["source"].fillna("") + " " +
        df["title"].fillna("") + " " +
        df["article"].fillna("")
    )


def main() -> None:
    # =========================
    # 1) Load training data
    # =========================
    train_df = pd.read_csv("development.csv")
    X_train = build_text(train_df)
    y_train = train_df["label"]

    # =========================
    # 2) WORD-LEVEL MODEL (strong)
    # =========================
    word_model = Pipeline([
        ("tfidf", TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 3),
            min_df=3,
            sublinear_tf=True,
            max_features=100000,
        )),
        ("clf", LogisticRegression(
            C=2.0,
            max_iter=2000,
            class_weight="balanced"
        )),
    ])

    # =========================
    # 3) CHAR-LEVEL MODEL (complementary)
    # =========================
    char_model = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(4, 6),
            min_df=3,
            sublinear_tf=True,
            max_features=200000,
        )),
        ("clf", LogisticRegression(
            C=2.0,
            max_iter=2000,
            class_weight="balanced"
        )),
    ])

    # Train both models
    print("Fitting word_model...")
    start_time = time.time()
    word_model.fit(X_train, y_train)
    print(f"Word model trained in {time.time() - start_time:.2f} seconds")

    print("Fitting char_model...")
    start_time = time.time()
    char_model.fit(X_train, y_train)
    print(f"Char model trained in {time.time() - start_time:.2f} seconds")

    # =========================
    # 4) Load evaluation data
    # =========================
    eval_df = pd.read_csv("evaluation.csv")
    X_eval = build_text(eval_df)

    # =========================
    # 5) Ensemble prediction
    # =========================
    word_proba = word_model.predict_proba(X_eval)
    char_proba = char_model.predict_proba(X_eval)

    # Weighted average (word model is stronger)
    final_proba = 0.7 * word_proba + 0.3 * char_proba
    final_preds = np.argmax(final_proba, axis=1)

    # =========================
    # 6) Submission
    # =========================
    submission = pd.DataFrame({
        "Id": eval_df["Id"],
        "Predicted": final_preds
    })

    out_path = "submission_ensemble_last.csv"
    submission.to_csv(out_path, index=False)

    print(f"✅ {out_path} created")
    print("Rows:", len(submission), "| Columns:", submission.columns.tolist())
    print(submission.head())


if __name__ == "__main__":
    start = time.time()
    main()
    print("Elapsed seconds:", round(time.time() - start, 2))