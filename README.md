# Titanic Survival Predictor

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn&logoColor=white)
![Kaggle Score](https://img.shields.io/badge/Kaggle%20score-0.77272-20BEFF?logo=kaggle&logoColor=white)

An end-to-end machine learning project that predicts which passengers survived the Titanic disaster. It covers exploratory analysis, feature engineering, a reproducible preprocessing pipeline, model comparison, hyperparameter tuning, and a validated submission file for the [Kaggle Titanic competition](https://www.kaggle.com/competitions/titanic).

**Result:** 0.77272 accuracy on the Kaggle public leaderboard (418 held-out passengers).

---

## Project overview

| | |
|---|---|
| **Task** | Binary classification: did the passenger survive (1) or not (0)? |
| **Data** | 891 labelled passengers (train), 418 unlabelled (test) |
| **Metric** | Accuracy (Kaggle's competition metric); ROC AUC and F1 tracked during development |
| **Stack** | Python, pandas, scikit-learn, Jupyter |

## Approach

### 1. Data cleaning
- Dropped the 2 training rows with a missing `Embarked` value.
- Converted `Cabin` (77% missing) into a boolean "has a recorded cabin" flag for analysis.
- Test rows are **never dropped**: Kaggle requires a prediction for every `PassengerId`, so missing test values are imputed instead.

### 2. Exploratory analysis
Survival rate by group in the training data:

| Feature | Finding |
|---|---|
| Sex | Women 74% vs. men 19% (strongest signal) |
| Passenger class | 1st: 63%, 2nd: 47%, 3rd: 24% |
| Port of embarkation | Cherbourg 55%, Queenstown 39%, Southampton 34% |
| Recorded cabin | With cabin 66% vs. without 30% |
| Fare | Positive correlation with survival (0.26) |

I also engineered `Age_bins`, `fam_size` (SibSp + Parch + 1) and a passenger `Title` extracted from names, and checked their relationship with survival.

### 3. Preprocessing pipeline
A single scikit-learn `ColumnTransformer` keeps training and test preprocessing identical and avoids data leakage:

- **Numeric** (`Age`, `SibSp`, `Parch`, `Fare`): median imputation, then standard scaling
- **Categorical** (`Sex`, `Embarked`): one-hot encoding
- **Ordinal** (`Pclass`): ordinal encoding

The pipeline is fitted on the training set only. The test set goes through `transform()`, never `fit_transform()`.

### 4. Model comparison
Four classifiers were compared with 5-fold cross-validation using default settings:

| Model | ROC AUC | Accuracy | F1 |
|---|---|---|---|
| SGD Classifier | 0.821 | 0.755 | 0.658 |
| **Gradient Boosting** | **0.868** | **0.830** | **0.762** |
| Random Forest | 0.853 | 0.811 | 0.745 |
| SVM (RBF) | 0.848 | 0.827 | 0.759 |

### 5. Hyperparameter tuning
The three strongest models were tuned with stratified 5-fold cross-validation, scored on accuracy:

- **Gradient Boosting:** `RandomizedSearchCV` (40 iterations) over `n_estimators`, `learning_rate`, `max_depth`, `min_samples_leaf`, `subsample`
- **Random Forest:** `RandomizedSearchCV` (40 iterations) over `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`
- **SVM:** `GridSearchCV` over `C` and `gamma`

The model with the best cross-validated accuracy is selected automatically and refit on the full training set.

### 6. Submission
The tuned model predicts on `test.csv` and writes `submission.csv` with the required `PassengerId,Survived` format. Built-in assertions check the row count (418), column names, binary values and unique IDs before the file is saved.

## Repository structure

```
TitanicSurvivalPredictor/
├── README.md
├── titanicv2.ipynb            # Main notebook: EDA, pipeline, tuning, submission
├── data/
│   ├── train.csv              # Kaggle training data
│   ├── test.csv               # Kaggle test data
│   ├── gender_submission.csv  # Kaggle sample submission
│   └── submission.csv         # Final predictions submitted to Kaggle
└── archive/
    └── titanic.ipynb          # Earlier version (EDA, pipeline, model comparison only)
```

## How to run

```bash
git clone https://github.com/yogesh71/TitanicSurvivalPredictor.git
cd TitanicSurvivalPredictor

pip install pandas scikit-learn jupyter
jupyter notebook titanicv2.ipynb
```

The notebook reads the Kaggle files from the `data/` folder. Before running, make sure the path to `train.csv` in the first cell and `data_dir` in the submission cell point to `data/`. Then run all cells from top to bottom.

## Limitations and next steps

I would rather be upfront about where this stands:

- Cross-validated accuracy is higher than the 0.773 leaderboard score. That gap is common on a dataset this small (891 rows), and it suggests some overfitting to the training folds.
- The final model uses only the seven original features. `Title` and `fam_size` were explored but not yet added to the pipeline.

Planned improvements:

- Add `Title` (with rare titles grouped), `fam_size` and an "is alone" flag to the pipeline and measure the change
- Impute `Age` using group medians (by title and class) instead of a global median
- Try ensembling (voting or stacking) of the tuned models
- Replace hardcoded file paths with relative paths so the notebook runs anywhere

## Author

**Yogesh Sah**
GitHub: [@yogesh71](https://github.com/yogesh71)
