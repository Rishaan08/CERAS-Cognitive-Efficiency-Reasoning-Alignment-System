#Imports
import os
import pandas as pd
import joblib
from sklearn.experimental import enable_iterative_imputer  
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from lightgbm import LGBMRegressor

#Load Data
df = pd.read_parquet("./data/processed/oulad_behavior.parquet")

target = "cognitive_efficiency"
id = "student_id"

#Imputation
impute_col = "oulad_n_assessments"

impute_features = [
    "oulad_avg_assessment_score",
    "oulad_avg_clicks",
    "oulad_active_days",
    "registration_duration_days",
    "effort_norm"  
]

impute_cols = impute_features + [impute_col]

imputer = IterativeImputer(
    estimator=ExtraTreesRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=1
    ),
    max_iter=10,
    initial_strategy="constant",
    fill_value=1,
    random_state=42
)

df[impute_cols] = imputer.fit_transform(df[impute_cols])

#Post-process n_assessments
df[impute_col] = (
    df[impute_col]
    .clip(lower=1)
    .round()
    .astype(int)
)

#Remove CE-derived / leaky features
df.drop(
    columns=[
        "score_norm",
        "score_w",
        "effort_norm",
        "clicks_norm",
        "days_norm",
        "reg_norm",
        "clicks_log",
        "has_reg_norm"
    ],
    inplace=True
)

#Train / Val / Test Split
students = df[id].unique()

#80% train pool, 20% test
train_pool_ids, test_ids = train_test_split(
    students,
    test_size=0.20,
    random_state=42
)

train_pool_df = df[df[id].isin(train_pool_ids)].reset_index(drop=True)
test_df = df[df[id].isin(test_ids)].reset_index(drop=True)

#80% train, 20% validation (inside train pool)
train_ids, val_ids = train_test_split(
    train_pool_df[id].unique(),
    train_size=0.80,
    random_state=42
)

train_df = train_pool_df[train_pool_df[id].isin(train_ids)].reset_index(drop=True)
val_df = train_pool_df[train_pool_df[id].isin(val_ids)].reset_index(drop=True)

#Feature / Target Split
feature_cols = [
    c for c in df.columns
    if c not in [id, target]
]

X_train = train_df[feature_cols]
y_train = train_df[target]

X_val = val_df[feature_cols]
y_val = val_df[target]

X_test = test_df[feature_cols]
y_test = test_df[target]

#Sanity Check (no leakage)
assert set(train_df[id]).isdisjoint(val_df[id])
assert set(train_df[id]).isdisjoint(test_df[id])
assert set(val_df[id]).isdisjoint(test_df[id])

# Normalization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

#Train CEPM (LightGBM)
cepm_model = LGBMRegressor(
    n_estimators=300,
    learning_rate=0.05,
    random_state=42
)

cepm_model.fit(X_train, y_train)

#Validation Performance
y_val_pred = cepm_model.predict(X_val)

print("\nValidation Performance")
print("- MAE :", mean_absolute_error(y_val, y_val_pred))
print("- MSE:", mean_squared_error(y_val, y_val_pred))
print("- R2  :", r2_score(y_val, y_val_pred))

#Test Performance
y_test_pred = cepm_model.predict(X_test)

print("\nTEST SET PERFORMANCE")
print("- MAE :", mean_absolute_error(y_test, y_test_pred))
print("- MSE:", mean_squared_error(y_test, y_test_pred))
print("- R2  :", r2_score(y_test, y_test_pred))


#Save Model & Scaler
os.makedirs("./artifacts", exist_ok=True)

joblib.dump(cepm_model, "./artifacts/cepm_lightgbm.pkl")
joblib.dump(scaler, "./artifacts/cepm_scaler.pkl")

print("\nCEPM model and scaler saved successfully.")