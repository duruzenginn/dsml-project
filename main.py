# %%
import pandas as pd

df = pd.read_csv("development.csv")
df.head()
df.columns
df.info()

# %%
df['label'].value_counts()

# %%
df['text'] = df['title'].fillna('') + ' ' + df['article'].fillna('')
df['text'].head()

# %%
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=50000
)

X_tfidf = vectorizer.fit_transform(df["text"])
X_tfidf.shape

# %% [markdown]
# Description: 
# Rows = articles
# Columns = words (features)
# 
# 🔹 79997 → number of articles
# You have 79,997 news articles in development.csv.
# Each row corresponds to one article.
# 
# 🔹 50000 → number of features (words)
# You told TF-IDF:
# 	•	The system selected the 50,000 most important words/word-patterns
# 	•	Each column corresponds to one word (or word combination)
# 
# One Row looks like this: [0.0, 0.12, 0.0, 0.87, 0.03, 0.0, ...]

# %% [markdown]
# This line confirms that:
# 	•	✅ Your text → numbers conversion worked
# 	•	✅ You now have a valid ML input
# 	•	✅ Each article is represented consistently
# 
# This is a big milestone, even if it looks simple.

# %%
from sklearn.model_selection import train_test_split

X = X_tfidf          # features (numbers)
y = df['label']     # target labels

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_train.shape, X_val.shape

# %% [markdown]
# X = X_tfidf
# y = df['label']
# What this means:
# 	•	X → the input features
# 	•	Here: the TF-IDF matrix (numbers representing article text)
# 	•	y → the target variable
# 	•	The label (0–6) indicating the news category
# 
# In ML notation:
# 	•	X = inputs
# 	•	y = correct answers
# 
# X_train, X_val, y_train, y_val = train_test_split(
#     X,
#     y,
#     test_size=0.2,
#     random_state=42,
#     stratify=y
# )
# 1️⃣ Splits the data
# 	•	X_train, y_train → used to train the model
# 	•	X_val, y_val → used to evaluate the model on unseen data
# 2️⃣ test_size=0.2
# 	•	20% of the data goes to validation
# 	•	80% remains for training
# This is a standard choice in the lectures.
# 3️⃣ random_state=42
# 	•	Fixes the randomness of the split
# 	•	Ensures reproducibility
# 	•	Running the code again gives the same split
# 
# Very important for:
# 	•	debugging
# 	•	fair comparison of models
# 4️⃣ stratify=y
# 	•	Keeps the class proportions the same in train and validation
# 	•	Important because the dataset is imbalanced (e.g. Health is rare)
# 
# Without this:
# 	•	validation set could miss rare classes
# 	•	evaluation would be misleading
# 
# X_train.shape, X_val.shape
# What this checks:
# 	•	Confirms the split worked
# 	•	Shows how many samples are in each set
# 	•	Number of columns (features) stays the same
# 
# 

# %%
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    max_iter=1000,
    n_jobs=-1
)

model.fit(X_train, y_train)

# %% [markdown]
# 
# Each row is a hyperparameter of Logistic Regression.
# 
# Hyperparameters are:
# 	•	chosen by you
# 	•	fixed before training
# 	•	they control how the model learns
#     

# %% [markdown]
# 
# 🔹 solver = 'lbfgs'
# 
# This is the optimization algorithm used to find the weights.
# 
# Plain English:
# 
# This is the math engine that adjusts the model until it fits the data.
# 
# Why this is good:
# 	•	lbfgs is standard
# 	•	works well for multiclass classification
# 	•	handles many features (like TF-IDF)
# 
# ✔️ Fully aligned with course defaults.

# %% [markdown]
# 🔹 max_iter = 1000
# 
# This is very important.
# 
# Plain English:
# 
# Maximum number of steps the optimizer is allowed to take.
# 
# Why we increased it:
# 	•	TF-IDF has 50,000 features
# 	•	Default (100) is often not enough
# 	•	1000 prevents premature stopping
# 
# ✔️ Correct and recommended.

# %% [markdown]
# 🔹 C = 1.0
# 
# This controls regularization strength.
# 
# Plain English:
# 	•	Large C → model fits data more closely
# 	•	Small C → model is more conservative
# 
# C = 1.0 means:
# 
# “Use a balanced, default amount of regularization.”
# 
# ✔️ Perfect baseline choice
# We’ll maybe tune this later, not now.

# %% [markdown]
# 🔹 penalty = 'deprecated'
# 
# This looks scary but it is not a problem.
# 
# What it really means:
# 	•	You did not explicitly set a penalty
# 	•	The solver default (l2) is used
# 
# So effectively:
# 
# You are using L2 regularization, which is standard.
# 
# You can safely ignore this for now.

# %% [markdown]
# 🔹 class_weight = None
# 
# This means:
# 
# All classes are treated equally during training.
# 
# Is this okay?
# 	•	Yes, for a baseline
# 	•	Later we may try class_weight='balanced' as an improvement
# 
# Right now:
# ✔️ Totally fine.
# 

# %% [markdown]
# 🔹 n_jobs = -1
# 
# Plain English:
# 
# Use all available CPU cores.
# 
# This only affects speed, not results.
# 
# ✔️ Good practice.
# 

# %% [markdown]
# 	•	LogisticRegression(...)
# → creates the model object
# 	•	max_iter=1000
# → allows more training iterations so the model converges
# (important with many features like TF-IDF)
# 	•	n_jobs=-1
# → uses all available CPU cores (faster)
# 	•	model.fit(X_train, y_train)
# → this is where learning happens
# The model finds patterns linking word features to labels

# %%
y_val_pred = model.predict(X_val)

# %%
from sklearn.metrics import f1_score

f1_macro = f1_score(y_val, y_val_pred, average="macro")
f1_macro

# %% [markdown]
# 
# 0.6475 means:
# On unseen validation data, your model is doing a reasonably good job at correctly classifying articles across all 7 categories, giving equal importance to each category.
# 

# %% [markdown]
# “The baseline Logistic Regression model achieves a Macro F1 of approximately 0.65 on the validation set, indicating that it generalizes reasonably well across all news categories, including underrepresented ones. This confirms that the feature extraction and learning pipeline is correct.”

# %%
from sklearn.linear_model import LogisticRegression

model_balanced = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

model_balanced.fit(X_train, y_train)

# %%
from sklearn.metrics import f1_score

y_val_pred_balanced = model_balanced.predict(X_val)
f1_macro_balanced = f1_score(y_val, y_val_pred_balanced, average="macro")
f1_macro_balanced

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

Cs = [0.25, 0.5, 1, 2, 4, 8]

results = []
for C in Cs:
    m = LogisticRegression(max_iter=1000, class_weight="balanced", C=C)
    m.fit(X_train, y_train)
    pred = m.predict(X_val)
    f1 = f1_score(y_val, pred, average="macro")
    results.append((C, f1))
    print(f"C={C:<4}  MacroF1={f1:.5f}")

best = max(results, key=lambda x: x[1])
print("\nBEST:", best)

# %%
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("clf", LogisticRegression(
        C=1,
        max_iter=1000,
        class_weight="balanced"
    ))
])

# %%
X_text = df["text"]
y = df["label"]

# %%
from sklearn.model_selection import train_test_split

X_train_text, X_val_text, y_train, y_val = train_test_split(
    X_text,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# %%
configs = [
    {"tfidf__ngram_range": (1,1)},
    {"tfidf__ngram_range": (1,2)},
]

for cfg in configs:
    pipe.set_params(**cfg)
    pipe.fit(X_train_text, y_train)
    preds = pipe.predict(X_val_text)
    f1 = f1_score(y_val, preds, average="macro")
    print(cfg, "→ Macro F1:", round(f1, 5))

# %%
configs = [
    {"tfidf__min_df": 1},
    {"tfidf__min_df": 2},
    {"tfidf__min_df": 3},
]

for cfg in configs:
    pipe.set_params(tfidf__ngram_range=(1,2), **cfg)
    pipe.fit(X_train_text, y_train)
    preds = pipe.predict(X_val_text)
    f1 = f1_score(y_val, preds, average="macro")
    print(cfg, "→ Macro F1:", round(f1, 5))

# %% [markdown]
# We choose mind-df: 2 beacuse it is safer to go on with the easier model
# 

# %%
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

final_pipe = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        ngram_range=(1,2),
        min_df=2
    )),
    ("clf", LogisticRegression(
        C=1,
        max_iter=1000,
        class_weight="balanced"
    ))
])

# %%
X_text_all = df["text"]          # already title+article
y_all = df["label"]

final_pipe.fit(X_text_all, y_all)

# %%
import pandas as pd

eval_df = pd.read_csv("evaluation.csv")
eval_df["text"] = eval_df["title"].fillna("") + " " + eval_df["article"].fillna("")

# %%
eval_pred = final_pipe.predict(eval_df["text"])

submission = pd.DataFrame({
    "Id": eval_df["Id"],
    "label": eval_pred
})

submission.to_csv("submission.csv", index=False)
submission.head()

# %%
print("evaluation rows:", len(eval_df))
print("submission rows:", len(submission))
print("columns:", submission.columns.tolist())
print(submission["label"].unique())


