#Imports
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_selection import mutual_info_regression
from lightgbm import LGBMRegressor

#Path
artifact_dir = "./artifacts"
os.makedirs(artifact_dir, exist_ok=True)

#Load dataset
print("\nLoading PISA student-level dataset")
df = pd.read_parquet("./data/processed/pisa_student.parquet")

target = "ce_score"
id_col = "cntstuid"

print("Dataset shape:", df.shape)

#Feature matrix
X = df.drop(columns=[id_col, target])
y = df[target]

print("\nInitial features:")
print(X.columns.tolist())

#Train / Val / Test split
students = df[id_col].unique()

train_pool_ids, test_ids = train_test_split(
    students,
    test_size=0.20,
    random_state=42
)

train_pool_df = df[df[id_col].isin(train_pool_ids)].reset_index(drop=True)
test_df = df[df[id_col].isin(test_ids)].reset_index(drop=True)

train_ids, val_ids = train_test_split(
    train_pool_df[id_col].unique(),
    train_size=0.80,
    random_state=42
)

train_df = train_pool_df[train_pool_df[id_col].isin(train_ids)]
val_df   = train_pool_df[train_pool_df[id_col].isin(val_ids)]

X_train = X.loc[train_df.index].copy()
y_train = y.loc[train_df.index]

X_val = X.loc[val_df.index].copy()
y_val = y.loc[val_df.index]

X_test = X.loc[test_df.index].copy()
y_test = y.loc[test_df.index]

#Mutual Information
mi = mutual_info_regression(X, y, random_state=42)
mi_scores = pd.Series(mi, index=X.columns)

threshold = np.percentile(mi_scores, 60)

selected_features = mi_scores[mi_scores >= threshold].index
X_l1 = X[selected_features]

print(f"MI threshold: {threshold:.6f}")
print(f"Features retained: {len(selected_features)}")
print("Selected features:", selected_features.tolist())

#Save feature list
np.save(f"{artifact_dir}/cepm_features.npy", selected_features.to_numpy())

#Apply feature selection
X_train = X_train[selected_features]
X_val   = X_val[selected_features]
X_test  = X_test[selected_features]

#Iterative Imputation
imputer = IterativeImputer(
    estimator=ExtraTreesRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=1
    ),
    max_iter=10,
    random_state=42
)

X_train = imputer.fit_transform(X_train)
X_val   = imputer.transform(X_val)
X_test  = imputer.transform(X_test)

#Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

#Train CEPM
print("\nTraining CEPM model")

cepm_model = LGBMRegressor(
    n_estimators=300,
    learning_rate=0.05,
    random_state=42,
    n_jobs=4,
    force_row_wise=True
)

cepm_model.fit(X_train, y_train)

#Evaluation
print("\nTrain Performance")
y_train_pred = cepm_model.predict(X_train)
print("MAE:", mean_absolute_error(y_train, y_train_pred))
print("MSE:", mean_squared_error(y_train, y_train_pred))
print("R2 :", r2_score(y_train, y_train_pred))

print("\nValidation Performance")
y_val_pred = cepm_model.predict(X_val)
print("MAE:", mean_absolute_error(y_val, y_val_pred))
print("MSE:", mean_squared_error(y_val, y_val_pred))
print("R2 :", r2_score(y_val, y_val_pred))

print("\nTest Performance")
y_test_pred = cepm_model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_test_pred))
print("MSE:", mean_squared_error(y_test, y_test_pred))
print("R2 :", r2_score(y_test, y_test_pred))

#Save artifacts
joblib.dump(cepm_model, f"{artifact_dir}/cepm_lightgbm.pkl")
joblib.dump(scaler, f"{artifact_dir}/cepm_scaler.pkl")