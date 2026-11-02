# Interview Preparation: Car Price Prediction

Answers are tied to what this project actually does and measured, so they can be defended
under follow-up questions. Numbers come from `artifacts/metrics.json`,
`artifacts/model_comparison.csv` and `artifacts/target_transform_ablation.csv`.

---

## The 60-second project pitch

> "I built an end-to-end regression system that predicts used-car resale prices in India from
> the CarDekho Kaggle dataset. While inspecting the data I found that a third of the rows,
> 101 of 301, were motorcycles and scooters, so I scoped the problem to the 198 actual cars.
> I put feature engineering, scaling and encoding inside one scikit-learn pipeline to prevent
> leakage, then compared four algorithms with repeated cross-validation. The biggest win was
> reframing the target: instead of predicting the absolute price, the model predicts the
> share of the new-car price a car retains. That cut Random Forest's CV error by 45%, because
> tree models can't extrapolate to price levels they never saw. The selected Random Forest gets
> R² 0.954 and MAE of about ₹69k on a held-out test set. It is served through a validated CLI
> and a Streamlit app, with 56 pytest tests and CI."

---

## Technical architecture (walkthrough)

1. **`download_data.py`**: fetches the CSV and verifies its SHA-256, so every run trains on
   identical bytes.
2. **`data_preprocessing.py`**:
   - `validate_schema`: required columns, numeric coercion with clear errors.
   - `clean_data`: deterministic row-level cleaning only (whitespace, missing values,
     duplicates, two-wheelers). No statistics are computed, so it is safe before the split.
   - `CarFeatureEngineer`: custom sklearn transformer → `Car_Age`, `Kms_Per_Year`. The
     reference year is *learned in `fit`*.
   - `build_preprocessor`: `ColumnTransformer` with log1p+scale for skewed quantities, scaling
     for age/owner, one-hot for categoricals (`handle_unknown="ignore"`).
   - `train_test_split_stratified`: 80/20 split stratified on price quintiles.
3. **`model.py`**: `CarPriceRegressor` meta-estimator that trains the pipeline on a raw, log or
   ratio target scale and always predicts in lakh INR. It is sklearn-compatible (works with
   `clone`, `GridSearchCV`, `cross_validate` and `permutation_importance`).
4. **`train.py`**: per algorithm, choose the target scale by CV → tune (Grid/Randomized
   search) → re-score with 5×3 repeated K-fold → select the lowest mean CV RMSE → score the
   test set once → save the bundle.
5. **`predict.py`**: `load_model` (clear errors for missing/corrupt files), `validate_vehicle`
   (collects *all* errors; warns on extrapolation), `predict_price`, `predict_batch`, CLI.
6. **`app/streamlit_app.py`**: cached model loading, a form, INR formatting, error display,
   and a model-performance tab.

---

## Core questions

### Why regression?
The target, selling price, is a **continuous quantity**, and we care about *how far off* a
prediction is (₹50k vs ₹5 lakh), not just whether it falls in a bucket. Classification would
throw that information away. Regression losses (squared/absolute error) directly optimise
closeness in rupees, and the metrics (MAE, RMSE) are in rupees, which stakeholders understand.

### Why Random Forest?
- It captures **non-linear effects and interactions** (e.g. depreciation per year differs by
  segment) without hand-crafting them.
- **Bagging** (bootstrap samples + random feature subsets, averaged over 500 trees) reduces the
  variance of a single deep tree. In this project the single Decision Tree had CV RMSE
  1.57 ± 0.89; the forest got 1.05 ± 0.21: lower error and far more stable.
- Needs little preprocessing, is robust to outliers in the inputs, and gives feature importances.
- **Its weakness, and how I handled it:** a forest predicts averages of training targets, so
  it **cannot extrapolate**. On the raw price scale, a fold containing the 92.6-lakh Land
  Cruiser broke it (CV RMSE 1.90). On the ratio scale (value retained, 0–1) that car is
  ordinary, and CV RMSE dropped to 1.04.
- **Be honest:** Gradient Boosting was statistically tied (1.08 ± 0.29). RF won on the
  pre-declared criterion (mean CV RMSE) and had lower variance. A tuned linear model on the
  same ratio target was not far behind (1.20), which shows how much of the problem is solved by
  the target framing rather than the algorithm.

### How does train-test splitting work here?
- 20% of the 198 cars (40) are held out **before** any EDA or modelling and used exactly once,
  for the final report.
- The split is **stratified on price quintiles** so both sets cover cheap and expensive cars;
  with ~200 rows a random split can easily skew.
- Inside the 158 training cars, **K-fold cross-validation** produces many train/validation
  splits for choosing the target scale, hyperparameters and the final model. Repeated 5-fold ×
  3 gives 15 scores → a mean *and* a standard deviation.
- Duplicates were removed **before** splitting, otherwise one copy could land in train and its
  twin in test, inflating the score.

### What is data leakage, and how did you prevent it?
Leakage is when information that would not be available at prediction time (or information
from the evaluation data) influences training, so offline scores look better than real-world
performance. Prevention in this project:
1. **Preprocessing is fitted inside the pipeline**, so in CV the scaler, encoder and reference
   year are fitted on training folds only. A test asserts the scaler's means equal the
   training-split means.
2. **EDA uses the training split only.**
3. **Model selection uses CV, not the test set.** GB had the better test score, and picking it
   for that reason would have been leakage through model selection.
4. **Reference year learned from training data** instead of `datetime.now()`. This also keeps
   predictions stable over time.
5. **Present_Price in the ratio target is not leakage:** it is an input the user supplies at
   prediction time, not derived from the target.
6. **`Car_Name` excluded:** not leakage, but 98 unique names in ~200 rows would let the model
   memorise identities.

### Explain MAE, RMSE and R²
- **MAE** = mean |actual − predicted|. Robust, same units as price. Test MAE = 0.687 lakh ≈
  ₹68,700: "on average the estimate is off by about ₹69k."
- **MSE** = mean squared error, in lakh² (hard to interpret, but it is what least squares
  optimises).
- **RMSE** = √MSE, back in lakh. Squaring before averaging makes it **sensitive to large
  errors**. Test RMSE 0.921 > MAE 0.687, which tells you some errors are notably bigger than
  average.
- **R²** = 1 − SS_residual / SS_total: the fraction of price variance explained relative to
  always predicting the mean. 0.954 means 95.4% explained. Caveat: R² depends on the target's
  spread; a dataset with very varied prices makes R² look good easily, so always pair it with
  an error metric in rupees.

### How does feature engineering help?
- **`Car_Age = reference_year − Year`**: for a linear model this is only a re-parameterisation
  of `Year`, and trees are invariant to it. Its value is **interpretability and stability**:
  "per year of age" is the business concept, and fixing the reference year keeps it stable.
  Be upfront about this; interviewers like candidates who know when a feature *doesn't* add
  information.
- **`Kms_Per_Year`**: separates heavy use from old age, information not linearly available
  from the raw columns.
- **log1p of prices and kilometres**: makes skewed features closer to linear in log-price
  space (which matters for the linear model).
- **The ratio target is target engineering**, and it was the single biggest improvement
  (RF −45% CV RMSE). It encodes domain knowledge: depreciation is a *fraction* of the new price.

### Why use Pipeline and ColumnTransformer?
- **`ColumnTransformer`** applies different preprocessing to different column types (log+scale
  numerics, one-hot categoricals) in one object.
- **`Pipeline`** chains feature engineering → preprocessing → model so that:
  1. `fit` learns every parameter **from training data only**, including inside each CV fold
     (leakage prevention);
  2. `GridSearchCV` can tune the model's hyperparameters on the whole chain;
  3. **one object is saved**, so inference replays exactly the training transformations (no
     train/serve skew);
  4. the code is shorter and less error-prone than manual `fit_transform` calls.

### How does model persistence work?
- `joblib.dump` serialises (pickles) the fitted estimator, including the fitted scaler means,
  one-hot categories, reference year and 500 trees, plus a metadata dict (feature schema with
  allowed categories and training ranges, metrics, sklearn version, timestamp).
- `joblib.load` reconstructs it. The custom classes (`CarFeatureEngineer`,
  `CarPriceRegressor`) must be importable, which is why the Streamlit app adds the project root
  to `sys.path` before loading.
- Production concerns I handled: version check with a warning on scikit-learn mismatch; clear
  errors for missing or corrupt files; a test that save → load gives identical predictions.
- Security: unpickling executes code, so only load model files you trust.
- The binary is git-ignored and **reproducible** from code + data + seed.

### How would you improve this model?
1. **More, newer data**: the biggest lever. 198 cars from ≤2018 is small and stale.
2. **Richer features**: brand/model via target encoding (fitted in-fold), engine size,
   condition, city.
3. **Uncertainty**: quantile regression or conformal prediction intervals instead of a single
   number.
4. **Nested CV** to remove the small optimism from tuning and selecting on the same CV data.
5. **Monotonic constraints** so price can never rise with age or mileage.
6. **Time-based validation** (train on older listings, test on newer) to measure drift.
7. **Serving**: FastAPI + Docker, model registry (MLflow), input drift monitoring.

---

## Likely follow-ups

**Q: Your test RMSE (0.92) is lower than CV RMSE (1.05). Isn't that odd?**
The 40-car test set's most expensive car is 20.75 lakh, while the training folds include the
92.6-lakh Land Cruiser, which produces the largest errors when it is in a validation fold. With
40 test rows, that is normal sampling noise; I quote the CV number as the realistic expectation.

**Q: Train RMSE is 0.38 but CV RMSE is 1.05. Is the forest overfitting?**
It fits training data closely (as forests with `min_samples_leaf=1` do), but what matters is
generalisation. CV error is low and stable across 15 folds (std 0.21), and the test score
agrees. Regularising (larger leaves, shallower trees) was in the search space and did not
improve CV error.

**Q: Why remove the two-wheelers instead of keeping them as extra data?**
They are a different product with different price dynamics, and the app is for cars. Keeping
them would also distort features: in the full file, `Seller_Type=Individual` was mostly bikes,
so the model would have learned "individual seller → cheap" for the wrong reason. I verified
the rule exactly: it flags 101 rows, all in one contiguous block (rows 100–200).

**Q: Why did tuning barely help Random Forest?**
Default RF on the ratio target scored 1.041 and the tuned version 1.049, a difference well
within noise. Random forests are fairly insensitive to hyperparameters; the target framing
mattered far more than tuning. I report that rather than claim tuning gains.

**Q: How do you handle a fuel type the model has never seen, like "Electric"?**
Validation rejects it with a clear message listing the allowed values. As a second safety net,
the encoder has `handle_unknown="ignore"`, so the pipeline itself would not crash.

**Q: What did you debug along the way?**
- A virtualenv failed to install on Windows because of the 260-character path limit; I moved
  it to a short path.
- The first training run picked a log-target Linear Regression. The tree models fit the
  training data far better than they generalised (Gradient Boosting's training RMSE was ≈0)
  and had high, unstable CV error (std 0.6–0.8 lakh). I tested the hypothesis that one
  extreme car (new price 92.6 lakh) was the cause: excluding it from validation scoring cut
  RF's CV RMSE from 1.90 to 1.49. That led to the ratio target, which fixed the root cause
  instead of deleting a genuine data point.
- To make sure the tests actually catch bugs, I injected an off-by-one in the reference year
  and a broken ratio decode; 3 and 9 tests failed respectively.
