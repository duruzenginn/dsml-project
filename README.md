# dsml-project
### Experiment Log (Validation Macro-F1)

- Baseline Logistic Regression (TF-IDF unigrams, 50k features): **0.6475**
- + class_weight="balanced": **0.6600**
- Best C among {0.25,0.5,1,2,4,8}: **C=1** (0.6600)
- TF-IDF ngrams: (1,2) improves to **0.6698**
- TF-IDF min_df: best stable choice **min_df=2** (0.6747)
- Final submission (leaderboard): **0.687**