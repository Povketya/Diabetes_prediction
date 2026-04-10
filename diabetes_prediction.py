# =============================================================
#  DIABETES PREDICTION WITH MACHINE LEARNING + EXPLAINABLE AI
#  Based on:
#    "Towards Transparent and Accurate Diabetes Prediction
#     Using Machine Learning and Explainable AI"
#
#  Steps in this file:
#    1.  Load the dataset
#    2.  Pre-process (impute, scale, balance with SMOTE)
#    3.  Split into Train / Validation / Test
#    4.  Train 7 individual ML models with hyperparameter tuning
#    5.  Build an Ensemble model (RF + XGBoost + LightGBM)
#    6.  Compare all models (Table II from paper)
#    7.  Explain predictions with SHAP, LIME, EBM, PDP, Anchors
#    8.  Compute Explainability Metrics (Table IV from paper)
#
#  Dataset: Diabetes Binary Health Indicators (BRFSS 2015)
#  Download from Kaggle and place the CSV in the same folder:
#    https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset
#  File name: diabetes_binary_health_indicators_BRFSS2015.csv
#
#  Install libraries once:
#    pip install -r requirements.txt
# =============================================================

import warnings
warnings.filterwarnings("ignore")

# ---------- basic tools ----------
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # save plots to files (no pop-up window needed)
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- scikit-learn ----------
from sklearn.model_selection import (train_test_split, RandomizedSearchCV,
                                     StratifiedKFold)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, classification_report)
from sklearn.inspection import permutation_importance, PartialDependenceDisplay

# ---------- handle class imbalance ----------
from imblearn.over_sampling import SMOTE

# ---------- boosting models ----------
import xgboost as xgb
import lightgbm as lgb

# ---------- explainability ----------
import shap
import lime
import lime.lime_tabular
from interpret.glassbox import ExplainableBoostingClassifier
from interpret import show as ebm_show


# =============================================================
#  SECTION 1 — LOAD DATASET
# =============================================================
print("=" * 60)
print("STEP 1: Loading dataset")
print("=" * 60)

CSV_FILE = "diabetes_binary_health_indicators_BRFSS2015.csv"

try:
    df = pd.read_csv(CSV_FILE)
except FileNotFoundError:
    raise FileNotFoundError(
        f"\nCannot find '{CSV_FILE}'.\n"
        "Please download it from Kaggle and put it in the same folder as this script.\n"
        "URL: https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset"
    )

print(f"Rows: {df.shape[0]}   Columns: {df.shape[1]}")
print("\nClass distribution (0 = Non-Diabetic, 1 = Diabetic):")
print(df["Diabetes_binary"].value_counts())
print(f"\nClass percentages:")
print((df["Diabetes_binary"].value_counts(normalize=True) * 100).round(2))
print(f"\nMissing values total: {df.isnull().sum().sum()}")


# =============================================================
#  SECTION 2 — PRE-PROCESSING
# =============================================================
print("\n" + "=" * 60)
print("STEP 2: Pre-processing")
print("=" * 60)

# --- 2a. Split features (X) and target (y) ---
TARGET = "Diabetes_binary"
X = df.drop(TARGET, axis=1)
y = df[TARGET]
feature_names = X.columns.tolist()

# --- 2b. Fill missing values with MEDIAN ---
# Median is better than mean for medical data because it is not
# affected by extreme outlier values (e.g. very high BMI)
imputer = SimpleImputer(strategy="median")
X = pd.DataFrame(imputer.fit_transform(X), columns=feature_names)
print("Missing values after imputation:", X.isnull().sum().sum())

# --- 2c. Scale features to mean=0, std=1 (StandardScaler) ---
# This makes sure BMI, blood pressure, cholesterol etc. are
# on the same scale so no feature dominates due to its unit
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_names)
print("Feature scaling done (mean≈0, std≈1).")


# =============================================================
#  SECTION 3 — TRAIN / VALIDATION / TEST SPLIT
# =============================================================
print("\n" + "=" * 60)
print("STEP 3: Splitting data  →  70% train | 15% val | 15% test")
print("=" * 60)

# First cut off 15% for the final test set
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X_scaled, y,
    test_size=0.15,
    random_state=42,
    stratify=y          # keeps class ratio the same in every split
)

# From the remaining 85%, cut 15% of original = ~17.6% of 85% for validation
val_ratio = 0.15 / 0.85
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val,
    test_size=val_ratio,
    random_state=42,
    stratify=y_train_val
)

total = len(X_scaled)
print(f"Train:      {len(X_train):>6}  ({len(X_train)/total*100:.1f}%)")
print(f"Validation: {len(X_val):>6}  ({len(X_val)/total*100:.1f}%)")
print(f"Test:       {len(X_test):>6}  ({len(X_test)/total*100:.1f}%)")

# --- 3b. Apply SMOTE ONLY on the training set ---
# SMOTE creates synthetic (artificial) samples for the minority class
# so both classes have equal numbers during training.
# NEVER apply SMOTE to val/test — that would fake the real-world evaluation.
print("\nApplying SMOTE to balance training classes...")
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
X_train_bal = pd.DataFrame(X_train_bal, columns=feature_names)

print(f"Before SMOTE: {dict(y_train.value_counts())}")
print(f"After  SMOTE: {dict(pd.Series(y_train_bal).value_counts())}")


# =============================================================
#  SECTION 4 — MODEL TRAINING & HYPERPARAMETER TUNING
# =============================================================
print("\n" + "=" * 60)
print("STEP 4: Training individual models")
print("=" * 60)

def evaluate_model(model, name, X_v, y_v, X_te, y_te):
    """Return a dict of metrics for validation and test sets."""
    val_pred  = model.predict(X_v)
    test_pred = model.predict(X_te)

    # predict_proba gives probabilities — needed for ROC-AUC
    if hasattr(model, "predict_proba"):
        val_prob  = model.predict_proba(X_v)[:, 1]
        test_prob = model.predict_proba(X_te)[:, 1]
    else:
        # SVC with decision_function fallback
        val_prob  = model.decision_function(X_v)
        test_prob = model.decision_function(X_te)

    return {
        "Model":              name,
        "Val Accuracy (%)":   round(accuracy_score(y_v,  val_pred)  * 100, 2),
        "Test Accuracy (%)":  round(accuracy_score(y_te, test_pred) * 100, 2),
        "Val ROC-AUC":        round(roc_auc_score(y_v,  val_prob),  3),
        "Test ROC-AUC":       round(roc_auc_score(y_te, test_prob), 3),
        "Precision":          round(precision_score(y_te, test_pred, zero_division=0), 2),
        "Recall":             round(recall_score(y_te, test_pred),   2),
        "F1-Score":           round(f1_score(y_te, test_pred),       2),
    }


def train_with_tuning(model, param_grid, name,
                      X_tr, y_tr, X_v, y_v, X_te, y_te):
    """
    Run RandomizedSearchCV with 3-fold CV to find the best
    hyperparameters, then evaluate on validation and test sets.
    """
    print(f"\n  [{name}]", end=" ", flush=True)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    if param_grid:
        searcher = RandomizedSearchCV(
            estimator  = model,
            param_distributions = param_grid,
            n_iter     = 10,           # try 10 random combinations
            cv         = cv,
            scoring    = "roc_auc",
            n_jobs     = -1,           # use all CPU cores
            random_state = 42,
            verbose    = 0,
        )
        searcher.fit(X_tr, y_tr)
        best = searcher.best_estimator_
        print(f"best params: {searcher.best_params_}")
    else:
        best = model
        best.fit(X_tr, y_tr)
        print("no tuning needed")

    metrics = evaluate_model(best, name, X_v, y_v, X_te, y_te)
    print(f"     Val Acc={metrics['Val Accuracy (%)']:.2f}%  "
          f"Test Acc={metrics['Test Accuracy (%)']:.2f}%  "
          f"AUC={metrics['Test ROC-AUC']:.3f}")
    return best, metrics


# ---- Define each model and the values to try for its settings ----
models_to_train = {
    "Random Forest": (
        RandomForestClassifier(random_state=42, n_jobs=-1),
        {
            "n_estimators":    [100, 200, 300],
            "max_depth":       [None, 10, 20],
            "min_samples_split": [2, 5, 10],
        }
    ),
    "XGBoost": (
        xgb.XGBClassifier(eval_metric="logloss", random_state=42,
                          use_label_encoder=False, n_jobs=-1),
        {
            "n_estimators":  [100, 200, 300],
            "max_depth":     [3, 5, 7],
            "learning_rate": [0.01, 0.1, 0.2],
            "subsample":     [0.8, 1.0],
        }
    ),
    "LightGBM": (
        lgb.LGBMClassifier(random_state=42, verbose=-1, n_jobs=-1),
        {
            "n_estimators":  [100, 200, 300],
            "max_depth":     [-1, 5, 10],
            "learning_rate": [0.01, 0.1, 0.2],
            "num_leaves":    [31, 63, 127],
        }
    ),
    "Decision Tree": (
        DecisionTreeClassifier(random_state=42),
        {
            "max_depth":         [3, 5, 10, None],
            "min_samples_split": [2, 5, 10],
            "criterion":         ["gini", "entropy"],
        }
    ),
    "SVM": (
        SVC(probability=True, random_state=42),
        {
            "C":      [0.1, 1, 10],
            "kernel": ["rbf", "linear"],
            "gamma":  ["scale", "auto"],
        }
    ),
    "Logistic Regression": (
        LogisticRegression(max_iter=1000, random_state=42),
        {"C": [0.01, 0.1, 1, 10]}
    ),
    "Naive Bayes": (
        GaussianNB(),
        None        # Naive Bayes has no important hyperparameters to tune
    ),
}

trained_models = {}
results_list   = []

for model_name, (model_obj, param_grid) in models_to_train.items():
    best_model, metrics = train_with_tuning(
        model_obj, param_grid, model_name,
        X_train_bal, y_train_bal,
        X_val, y_val,
        X_test, y_test,
    )
    trained_models[model_name] = best_model
    results_list.append(metrics)


# =============================================================
#  SECTION 5 — ENSEMBLE MODEL  (RF + XGBoost + LightGBM)
# =============================================================
print("\n" + "=" * 60)
print("STEP 5: Building Ensemble Model (Soft Voting)")
print("=" * 60)
# Soft voting = average the predicted probabilities of all 3 models.
# This is better than hard voting (majority wins) because it uses
# the confidence of each model, not just its final yes/no answer.

ensemble = VotingClassifier(
    estimators=[
        ("rf",  trained_models["Random Forest"]),
        ("xgb", trained_models["XGBoost"]),
        ("lgb", trained_models["LightGBM"]),
    ],
    voting="soft",
    n_jobs=-1,
)
ensemble.fit(X_train_bal, y_train_bal)
ens_metrics = evaluate_model(ensemble, "Ensemble Model",
                              X_val, y_val, X_test, y_test)
results_list.append(ens_metrics)
trained_models["Ensemble Model"] = ensemble

print(f"  Ensemble  Val Acc={ens_metrics['Val Accuracy (%)']:.2f}%  "
      f"Test Acc={ens_metrics['Test Accuracy (%)']:.2f}%  "
      f"AUC={ens_metrics['Test ROC-AUC']:.3f}")


# =============================================================
#  SECTION 6 — RESULTS TABLE  (Table II from paper)
# =============================================================
print("\n" + "=" * 60)
print("STEP 6: Performance Comparison Table (Table II)")
print("=" * 60)

results_df = pd.DataFrame(results_list).set_index("Model")
print(results_df.to_string())

# Save table to CSV
results_df.to_csv("model_comparison.csv")
print("\nSaved: model_comparison.csv")

# Bar chart of test accuracy
plt.figure(figsize=(10, 5))
results_df["Test Accuracy (%)"].plot(kind="bar", color="steelblue", edgecolor="black")
plt.ylabel("Test Accuracy (%)")
plt.title("Model Test Accuracy Comparison")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("model_accuracy_comparison.png", dpi=150)
plt.close()
print("Saved: model_accuracy_comparison.png")


# =============================================================
#  SECTION 7 — EXPLAINABILITY
# =============================================================

# We use XGBoost as the base model for SHAP because TreeExplainer
# works fastest with tree-based models.
xgb_model = trained_models["XGBoost"]

# Use a smaller sample of test data so SHAP runs faster
SAMPLE_SIZE = 500
X_test_sample = X_test.iloc[:SAMPLE_SIZE].reset_index(drop=True)
y_test_sample = y_test.iloc[:SAMPLE_SIZE].reset_index(drop=True)

# ------------------------------------------------------------------
#  7A — SHAP  (SHapley Additive exPlanations)
#       Shows which features push a prediction up (+) or down (-)
#       Based on game theory — each feature gets a "fair share" of credit
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 7A: SHAP Explanations")
print("=" * 60)

explainer_shap = shap.TreeExplainer(xgb_model)
shap_values    = explainer_shap.shap_values(X_test_sample)

# ---- SHAP Summary Plot (Fig. 2 from paper) ----
# Each dot = one patient.  Red = high feature value, Blue = low.
# Dots on the right = pushed prediction TOWARDS diabetic.
print("  Generating SHAP Summary Plot (Fig. 2)...")
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test_sample,
                  feature_names=feature_names, show=False)
plt.title("SHAP Summary Plot — Global Feature Importance (Fig. 2)")
plt.tight_layout()
plt.savefig("shap_summary_plot.png", dpi=150)
plt.close()
print("  Saved: shap_summary_plot.png")

# ---- SHAP Waterfall Plot (Fig. 7 from paper) ----
# Shows how each feature moves the prediction for ONE patient
# starting from the average prediction (base value).
print("  Generating SHAP Waterfall Plot (Fig. 7)...")
shap_explanation = explainer_shap(X_test_sample)
plt.figure(figsize=(10, 6))
shap.plots.waterfall(shap_explanation[0], show=False)
plt.title("SHAP Waterfall — Patient 0 (Fig. 7)")
plt.tight_layout()
plt.savefig("shap_waterfall.png", dpi=150)
plt.close()
print("  Saved: shap_waterfall.png")

# ---- SHAP Force Plot (Fig. 6 from paper) ----
# A horizontal view of the same idea — arrows push left or right.
print("  Generating SHAP Force Plot (Fig. 6)...")
shap.plots.force(
    shap_explanation[0],
    matplotlib=True,
    show=False
)
plt.title("SHAP Force Plot — Patient 0 (Fig. 6)")
plt.tight_layout()
plt.savefig("shap_force_plot.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: shap_force_plot.png")

# ---- SHAP Decision Plot (Fig. 8 from paper) ----
# Shows the cumulative path from base value to final prediction
# for multiple patients at once.
print("  Generating SHAP Decision Plot (Fig. 8)...")
plt.figure(figsize=(10, 8))
shap.decision_plot(
    explainer_shap.expected_value,
    shap_values[:20],
    X_test_sample.iloc[:20],
    feature_names=feature_names,
    show=False,
)
plt.title("SHAP Decision Plot — 20 Patients (Fig. 8)")
plt.tight_layout()
plt.savefig("shap_decision_plot.png", dpi=150)
plt.close()
print("  Saved: shap_decision_plot.png")

# Print top-5 most important features by average SHAP value
mean_abs_shap = np.abs(shap_values).mean(axis=0)
shap_importance = (
    pd.DataFrame({"Feature": feature_names, "Mean |SHAP|": mean_abs_shap})
    .sort_values("Mean |SHAP|", ascending=False)
)
print("\n  Top 5 features by SHAP importance:")
print(shap_importance.head(5).to_string(index=False))


# ------------------------------------------------------------------
#  7B — Permutation Importance  (Fig. 4 from paper)
#       Shuffle one feature at a time; see how much accuracy drops.
#       Big drop = that feature was very important.
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 7B: Permutation Importance (Fig. 4)")
print("=" * 60)

perm = permutation_importance(
    ensemble, X_test, y_test,
    n_repeats=10, random_state=42, n_jobs=-1
)
perm_df = (
    pd.DataFrame({"Feature": feature_names,
                  "Mean Decrease in Accuracy": perm.importances_mean})
    .sort_values("Mean Decrease in Accuracy", ascending=True)
)

plt.figure(figsize=(10, 8))
plt.barh(perm_df["Feature"], perm_df["Mean Decrease in Accuracy"],
         color="steelblue")
plt.xlabel("Mean Decrease in Accuracy")
plt.title("Permutation Importance — Ensemble Model (Fig. 4)")
plt.tight_layout()
plt.savefig("permutation_importance.png", dpi=150)
plt.close()
print("  Saved: permutation_importance.png")


# ------------------------------------------------------------------
#  7C — EBM  (Explainable Boosting Machine)  (Fig. 3 from paper)
#       EBM is like a boosted model but you can read exactly what
#       each feature contributes — no black box.
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 7C: Explainable Boosting Machine (EBM) — Fig. 3")
print("=" * 60)

ebm = ExplainableBoostingClassifier(random_state=42)
ebm.fit(X_train_bal, y_train_bal)

ebm_pred = ebm.predict(X_test)
ebm_acc  = accuracy_score(y_test, ebm_pred) * 100
ebm_auc  = roc_auc_score(y_test, ebm.predict_proba(X_test)[:, 1])
print(f"  EBM Test Accuracy: {ebm_acc:.2f}%  |  ROC-AUC: {ebm_auc:.3f}")

# EBM feature importance bar chart (mirror of Fig. 3)
ebm_importances = pd.DataFrame({
    "Feature":           feature_names,
    "Mean Abs Score":    ebm.term_importances(),
}).sort_values("Mean Abs Score", ascending=True)

plt.figure(figsize=(10, 8))
plt.barh(ebm_importances["Feature"], ebm_importances["Mean Abs Score"],
         color="orange")
plt.xlabel("Mean Absolute Score (Weighted)")
plt.title("EBM Feature Importance — Top Global Predictors (Fig. 3)")
plt.tight_layout()
plt.savefig("ebm_feature_importance.png", dpi=150)
plt.close()
print("  Saved: ebm_feature_importance.png")

# EBM interactive explanation (opens in browser)
print("  Opening EBM global explanation in browser...")
ebm_global = ebm.explain_global(name="EBM Global Importance")
ebm_show(ebm_global)

ebm_local = ebm.explain_local(
    X_test.iloc[:5], y_test.iloc[:5], name="EBM Local (5 patients)"
)
ebm_show(ebm_local)


# ------------------------------------------------------------------
#  7D — LIME  (Local Interpretable Model-Agnostic Explanations)
#             (Fig. 5 from paper)
#       LIME explains ONE prediction by building a simple model
#       in the neighbourhood of that data point.
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 7D: LIME Explanations (Fig. 5)")
print("=" * 60)

lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data  = X_train_bal.values,
    feature_names  = feature_names,
    class_names    = ["Non-Diabetic", "Diabetic"],
    mode           = "classification",
    random_state   = 42,
)

# Explain Instance 0
instance_0 = X_test.iloc[0].values
lime_exp   = lime_explainer.explain_instance(
    data_row   = instance_0,
    predict_fn = ensemble.predict_proba,
    num_features = 10,
)

fig = lime_exp.as_pyplot_figure()
plt.title("LIME Explanation — Patient 0 (Fig. 5)")
plt.tight_layout()
plt.savefig("lime_explanation_instance0.png", dpi=150)
plt.close()
print("  Saved: lime_explanation_instance0.png")

print("\n  Top 10 features for Patient 0:")
for feature_rule, weight in lime_exp.as_list():
    direction = "↑ DIABETIC" if weight > 0 else "↓ SAFE"
    print(f"    {feature_rule:45s}  {weight:+.5f}  {direction}")


# ------------------------------------------------------------------
#  7E — Partial Dependence Plots (PDPs)  (Fig. 9 from paper)
#       Shows the average effect of one feature on the prediction
#       while keeping all other features at their average values.
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 7E: Partial Dependence Plots (Fig. 9)")
print("=" * 60)

highbp_idx   = feature_names.index("HighBP")
highchol_idx = feature_names.index("HighChol")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
PartialDependenceDisplay.from_estimator(
    ensemble,
    X_test,
    features      = [highbp_idx, highchol_idx],
    feature_names = feature_names,
    ax            = axes,
)
plt.suptitle(
    "Partial Dependence — HighBP & HighChol interaction with Diabetes (Fig. 9)",
    fontsize=12
)
plt.tight_layout()
plt.savefig("pdp_highbp_highchol.png", dpi=150)
plt.close()
print("  Saved: pdp_highbp_highchol.png")


# ------------------------------------------------------------------
#  7F — Anchor Explanations  (Fig. 10 from paper)
#       Anchors give IF-THEN rules:
#         "IF BMI > 30 AND PhysActivity = 0  THEN  Diabetic (90% sure)"
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 7F: Anchor Explanations (Fig. 10)")
print("=" * 60)

try:
    from alibi.explainers import AnchorTabular

    anchor_exp = AnchorTabular(
        predictor     = ensemble.predict,
        feature_names = feature_names,
    )
    anchor_exp.fit(X_train_bal.values, disc_perc=(25, 50, 75))

    anchor_result = anchor_exp.explain(
        X_test.iloc[0].values, threshold=0.90
    )
    print(f"  Rule:      IF  {' AND '.join(anchor_result.anchor)}")
    print(f"  Precision: {anchor_result.precision:.2f}")
    print(f"  Coverage:  {anchor_result.coverage:.2f}")

except ImportError:
    print("  alibi not installed — skipping. Run: pip install alibi[tensorflow]")

# Plot distribution of PhysActivity with an anchor threshold line (Fig. 10)
phys_vals = X_test["PhysActivity"]
threshold = float(phys_vals.median())

plt.figure(figsize=(8, 5))
plt.hist(phys_vals, bins=30, color="steelblue", alpha=0.7, label="PhysActivity")
plt.axvline(threshold, color="red", linestyle="--",
            label=f"Anchor Threshold = {threshold:.2f}")
plt.xlabel("PhysActivity (scaled)")
plt.ylabel("Count")
plt.title("Distribution of PhysActivity with Anchor Threshold (Fig. 10)")
plt.legend()
plt.tight_layout()
plt.savefig("anchor_threshold_physactivity.png", dpi=150)
plt.close()
print("  Saved: anchor_threshold_physactivity.png")


# ------------------------------------------------------------------
#  7G — Counterfactual Explanations  (Table III from paper)
#       "What would need to change for you to NOT be diabetic?"
#       e.g.  lower your BMI by 0.24,  increase PhysActivity by 0.57
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 7G: Counterfactual Explanations (Table III)")
print("=" * 60)

try:
    import dice_ml
    from dice_ml import Dice

    # Combine balanced train features + labels into one dataframe for DiCE
    train_df_for_dice = X_train_bal.copy()
    train_df_for_dice[TARGET] = y_train_bal.values

    dice_data  = dice_ml.Data(
        dataframe          = train_df_for_dice,
        continuous_features = feature_names,
        outcome_name       = TARGET,
    )
    dice_model = dice_ml.Model(model=ensemble, backend="sklearn")
    dice_exp   = Dice(dice_data, dice_model, method="random")

    # Find the first diabetic patient in the test set
    diabetic_instance = X_test[y_test == 1].iloc[:1]

    cf_result = dice_exp.generate_counterfactuals(
        diabetic_instance,
        total_CFs     = 3,
        desired_class = "opposite",   # flip from Diabetic → Non-Diabetic
    )
    print("  Counterfactual explanations (changes that would flip prediction):")
    cf_result.visualize_as_dataframe(show_only_changes=True)

except ImportError:
    print("  dice-ml not installed — showing paper values from Table III instead.")

# Always print the coefficients from the paper (Table III) as reference
print("\n  Counterfactual coefficients from paper (Table III):")
cf_paper = pd.DataFrame([
    {
        "HighBP": 1.15, "HighChol": 1.17, "CholCheck": -2.89,
        "BMI": 0.24, "Smoker": -0.89, "Stroke": -0.21,
        "HeartDiseaseorAttack": -0.32, "PhysActivity": 0.57,
        "Fruits": -1.32, "Veggies": -0.32, "HvyAlcoholConsump": -0.24,
        "AnyHealthcare": 0.23, "NoDocbcCost": -0.30,
        "GenHlth": 1.39, "MentHlth": -0.16,
    },
    {
        "HighBP": 1.15, "HighChol": 1.17, "CholCheck": 0.20,
        "BMI": 0.24, "Smoker": -0.89, "Stroke": -0.21,
        "HeartDiseaseorAttack": -0.32, "PhysActivity": 0.57,
        "Fruits": -1.32, "Veggies": -2.07, "HvyAlcoholConsump": -0.24,
        "AnyHealthcare": 0.23, "NoDocbcCost": -0.30,
        "GenHlth": 1.39, "MentHlth": -0.16,
    },
    {
        "HighBP": 1.15, "HighChol": -0.21, "CholCheck": 0.20,
        "BMI": 0.24, "Smoker": -0.89, "Stroke": 1.26,
        "HeartDiseaseorAttack": -0.32, "PhysActivity": 0.57,
        "Fruits": -1.32, "Veggies": -2.07, "HvyAlcoholConsump": -0.24,
        "AnyHealthcare": 0.23, "NoDocbcCost": -0.30,
        "GenHlth": 1.39, "MentHlth": -0.16,
    },
])
print(cf_paper.to_string(index=False))


# =============================================================
#  SECTION 8 — EXPLAINABILITY METRICS  (Table IV from paper)
#  These numbers tell us HOW GOOD the explanations are,
#  not just how good the model is.
# =============================================================
print("\n" + "=" * 60)
print("STEP 8: Explainability Metrics (Table IV)")
print("=" * 60)

METRIC_SAMPLE = 100     # use 100 test patients to keep it fast

# ---- FIDELITY ----
# How closely does LIME's local prediction match the ensemble prediction?
# Score close to 1.0 = LIME is accurately describing the model.
fidelity_scores = []
for i in range(METRIC_SAMPLE):
    instance = X_test.iloc[i].values
    exp      = lime_explainer.explain_instance(
        instance, ensemble.predict_proba, num_features=len(feature_names)
    )
    lime_prob      = exp.local_pred[0]          # LIME's probability estimate
    ensemble_prob  = ensemble.predict_proba([instance])[0][1]
    fidelity_scores.append(1.0 - abs(lime_prob - ensemble_prob))
fidelity = float(np.mean(fidelity_scores))

# ---- FAITHFULNESS ----
# Mask the top-5 most important features (set them to 0) and measure
# how much the prediction changes.  Bigger change = more faithful explanation
# because the features identified really DO matter.
def compute_faithfulness(model, X_sample, shap_vals, top_k=5):
    scores = []
    for i in range(len(X_sample)):
        original_prob = model.predict_proba(X_sample.iloc[[i]])[0][1]
        top_feat_idx  = np.argsort(np.abs(shap_vals[i]))[-top_k:]
        X_masked      = X_sample.iloc[[i]].copy()
        X_masked.iloc[0, top_feat_idx] = 0.0
        masked_prob   = model.predict_proba(X_masked)[0][1]
        scores.append(abs(original_prob - masked_prob))
    return float(np.mean(scores))

faithfulness = compute_faithfulness(
    xgb_model, X_test_sample, shap_values
)

# ---- SPARSITY ----
# How many features does LIME actually use (weight > tiny threshold)?
# Fewer features = simpler, easier-to-read explanation.
sparsity_counts = []
for i in range(METRIC_SAMPLE):
    instance = X_test.iloc[i].values
    exp      = lime_explainer.explain_instance(
        instance, ensemble.predict_proba, num_features=len(feature_names)
    )
    used = sum(1 for _, w in exp.as_list() if abs(w) > 1e-4)
    sparsity_counts.append(used)
sparsity = float(np.mean(sparsity_counts))

# ---- STABILITY ----
# Run LIME twice on the same patient.  How different are the explanations?
# Close to 0 = very stable (same answer every time) — good.
stability_diffs = []
for i in range(min(20, METRIC_SAMPLE)):
    instance = X_test.iloc[i].values
    exp1 = lime_explainer.explain_instance(
        instance, ensemble.predict_proba, num_features=10
    )
    exp2 = lime_explainer.explain_instance(
        instance, ensemble.predict_proba, num_features=10
    )
    w1 = dict(exp1.as_list())
    w2 = dict(exp2.as_list())
    common_keys = set(w1) & set(w2)
    if common_keys:
        diff = np.mean([abs(w1[k] - w2[k]) for k in common_keys])
        stability_diffs.append(diff)
stability = float(np.mean(stability_diffs)) if stability_diffs else 0.0

# ---- CONSISTENCY ----
# Do SHAP and LIME agree on which features are most important?
# Score = fraction of top-5 features that both methods agree on.
shap_top5 = set(
    pd.DataFrame({"Feature": feature_names, "Score": mean_abs_shap})
    .nlargest(5, "Score")["Feature"]
)
lime_weights_all = dict(lime_exp.as_list())
lime_top5 = set(
    sorted(lime_weights_all, key=lambda k: abs(lime_weights_all[k]), reverse=True)[:5]
)

# lime keys look like "0.08 < BMI <= 0.70" — extract feature name from them
def extract_feature(rule_str):
    for feat in feature_names:
        if feat in rule_str:
            return feat
    return rule_str

lime_top5_clean = {extract_feature(k) for k in lime_top5}
consistency = len(shap_top5 & lime_top5_clean) / 5.0

# ---- Print results ----
metrics_table = pd.DataFrame([
    {"Metric": "Fidelity",     "Value": round(fidelity,     3),
     "Meaning": "LIME matches ensemble predictions (closer to 1 = better)"},
    {"Metric": "Faithfulness", "Value": round(faithfulness, 3),
     "Meaning": "Top features actually change prediction when removed (higher = better)"},
    {"Metric": "Sparsity",     "Value": round(sparsity,     1),
     "Meaning": "Avg features used per explanation (lower = simpler)"},
    {"Metric": "Stability",    "Value": f"{stability:.2e}",
     "Meaning": "Explanation difference on same input (closer to 0 = more stable)"},
    {"Metric": "Consistency",  "Value": round(consistency,  2),
     "Meaning": "SHAP and LIME agree on top features (closer to 1 = better)"},
]).set_index("Metric")

print(metrics_table.to_string())

# Save to CSV
metrics_table.to_csv("explainability_metrics.csv")
print("\nSaved: explainability_metrics.csv")


# =============================================================
#  FINAL SUMMARY
# =============================================================
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print("\nModel Performance (Table II):")
print(results_df.to_string())

print("\nTop 5 Predictors (SHAP global importance):")
print(shap_importance.head(5).to_string(index=False))

print("\nExplainability Metrics (Table IV):")
print(metrics_table[["Value"]].to_string())

print("\nOutput files saved:")
output_files = [
    "model_comparison.csv",
    "model_accuracy_comparison.png",
    "shap_summary_plot.png",
    "shap_waterfall.png",
    "shap_force_plot.png",
    "shap_decision_plot.png",
    "permutation_importance.png",
    "ebm_feature_importance.png",
    "lime_explanation_instance0.png",
    "pdp_highbp_highchol.png",
    "anchor_threshold_physactivity.png",
    "explainability_metrics.csv",
]
for f in output_files:
    print(f"  {f}")

print("\nAll done!")
