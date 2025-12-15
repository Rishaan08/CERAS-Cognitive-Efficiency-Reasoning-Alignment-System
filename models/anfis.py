#Imports
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.feature_selection import mutual_info_regression, RFE
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from boruta import BorutaPy
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Paths
ARTIFACT_DIR = "./artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

#Load data
df = pd.read_parquet("./data/processed/features_final.parquet")

#Missingness analysis
missing_ratio = df.isnull().mean()

low_missing = missing_ratio[(missing_ratio >= 0) & (missing_ratio <= 0.30)].index.tolist()
mid_missing = missing_ratio[(missing_ratio > 0.30) & (missing_ratio <= 0.80)].index.tolist()
high_missing = missing_ratio[(missing_ratio > 0.80) & (missing_ratio <= 0.95)].index.tolist()
extreme_missing = missing_ratio[(missing_ratio > 0.95)].index.tolist()

#Iterative Imputation
numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
low_missing_numeric = [c for c in low_missing if c in numeric_cols]

df[low_missing_numeric] = df[low_missing_numeric].astype("float64").replace({pd.NA: np.nan})

imputer = IterativeImputer(
    estimator=ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=1),
    max_iter=10,
    initial_strategy="constant",
    fill_value=1,
    random_state=0
)
df[low_missing_numeric] = imputer.fit_transform(df[low_missing_numeric])

#Mid / High missing handling
for col in mid_missing:
    df[f"{col}_is_present"] = df[col].notnull().astype(int)
    df[col] = df[col].fillna(df[col].median())

for col in high_missing:
    df[f"{col}_is_present"] = df[col].notnull().astype(int)
    df[col] = df[col].fillna(0)

df.drop(columns=extreme_missing, inplace=True)

#Feature matrix
target = "cognitive_efficiency"
id = "student_id"

X = df.drop(columns=[id, target, "dataset"])
y = df[target]

#Layer 1 — Mutual Information
mi = mutual_info_regression(X, y, random_state=42)
mi_scores = pd.Series(mi, index=X.columns)
threshold = np.percentile(mi_scores, 70)

selected_l1 = mi_scores[mi_scores >= threshold].index
X_l1 = X[selected_l1]

#Layer 2 — Recursive Feature Elimination
rfe = RFE(
    estimator=LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        random_state=42,
        n_jobs=-1
    ),
    n_features_to_select=30,
    step=0.1
)
rfe.fit(X_l1, y)
selected_l2 = X_l1.columns[rfe.support_]
X_l2 = X_l1[selected_l2]

#Layer 3 — LASSO
scaler = StandardScaler()
X_l2_scaled = scaler.fit_transform(X_l2)

lasso = LassoCV(
    alphas=np.logspace(-4, 0, 50),
    cv=5,
    random_state=42,
    n_jobs=-1
)
lasso.fit(X_l2_scaled, y)

lasso_coef = pd.Series(lasso.coef_, index=X_l2.columns)
selected_l3 = lasso_coef[lasso_coef != 0].index
X_l3 = X_l2[selected_l3]

#Layer 4 — Boruta
boruta = BorutaPy(
    estimator=RandomForestRegressor(
        n_estimators=500,
        random_state=42,
        n_jobs=-1
    ),
    n_estimators="auto",
    perc=100,
    random_state=42
)

boruta.fit(X_l3.values, y.values)
selected_l4 = X_l3.columns[boruta.support_]
X_final = X_l3[selected_l4]

#Save final features
np.save(f"{ARTIFACT_DIR}/anfis_features_l4.npy", selected_l4.to_numpy())

#ANFIS Data Preparation
df1 = df[df[target].notna()].reset_index(drop=True)
anfis_df = df1[selected_l4.tolist() + [target]].copy()

#Iterative imputer
iter_imputer = IterativeImputer(
    estimator=ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    initial_strategy="constant",
    fill_value=0,
    max_iter=10,
    random_state=42
)
anfis_df[selected_l4] = iter_imputer.fit_transform(anfis_df[selected_l4])

#Outlier clipping
for col in selected_l4:
    anfis_df[col] = anfis_df[col].clip(
        anfis_df[col].quantile(0.01),
        anfis_df[col].quantile(0.99)
    )

#MinMax scaling (required for ANFIS)
mm_scaler = MinMaxScaler(feature_range=(0, 1))
anfis_df[selected_l4] = mm_scaler.fit_transform(anfis_df[selected_l4])

#Train / Val / Test split
X = anfis_df[selected_l4].values
y = anfis_df[target].values

X_train_pool, X_test, y_train_pool, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_pool, y_train_pool, test_size=0.20, random_state=42
)

#ANFIS MODEL 
class SimpleANFIS:
    def __init__(self, n_inputs, n_mfs=3):
        self.n_inputs = n_inputs
        self.n_mfs = n_mfs
        self.rules = n_mfs ** n_inputs

        self.mf_params = {
            i: {
                j: {"mean": np.random.uniform(0, 1), "sigma": np.random.uniform(0.1, 0.3)}
                for j in range(n_mfs)
            }
            for i in range(n_inputs)
        }
        self.consequents = None

    #ANFIS Architecture: Layer 1 : Fuzzification and membership function
    def gaussian_mf(self, x, mean, sigma):
        return np.exp(-0.5 * ((x - mean) / sigma) ** 2)

    #ANFIS Architecture: Layer 2-3 : Rule Firing Strength and Normalization
    def forward(self, X):
        firing = []
        for x in X:
            rule_strengths = []
            for rule in range(self.rules):
                strength = 1.0
                tmp = rule
                for i in range(self.n_inputs):
                    mf_idx = tmp % self.n_mfs
                    tmp //= self.n_mfs
                    p = self.mf_params[i][mf_idx]
                    strength *= self.gaussian_mf(x[i], p["mean"], p["sigma"])
                rule_strengths.append(strength)
            firing.append(rule_strengths)
        firing = np.array(firing)
        return firing / (np.sum(firing, axis=1, keepdims=True) + 1e-8)

    #ANFIS Architecture: Layer 4 : Comsequent Layer
    def fit(self, X, y):
        W = self.forward(X)
        X_aug = np.hstack([X, np.ones((X.shape[0], 1))])
        Phi = np.vstack([np.hstack([W[i, r] * X_aug[i] for r in range(W.shape[1])])
                         for i in range(X.shape[0])])
        self.consequents = np.linalg.pinv(Phi) @ y

    #ANFIS Architecture: Layer 5 : Defuzzification
    def predict(self, X):
        W = self.forward(X)
        X_aug = np.hstack([X, np.ones((X.shape[0], 1))])
        Phi = np.vstack([np.hstack([W[i, r] * X_aug[i] for r in range(W.shape[1])])
                         for i in range(X.shape[0])])
        return Phi @ self.consequents

#Train ANFIS
anfis = SimpleANFIS(n_inputs=X_train.shape[1], n_mfs=3)
anfis.fit(X_train, y_train)

#Evaluation
for split, Xs, ys in [
    ("Train", X_train, y_train),
    ("Validation", X_val, y_val),
    ("Test", X_test, y_test),
]:
    yp = anfis.predict(Xs)
    print(f"\nANFIS {split}")
    print("MAE:", mean_absolute_error(ys, yp))
    print("R2 :", r2_score(ys, yp))

#Save model
joblib.dump(anfis, f"{ARTIFACT_DIR}/anfis_model.pkl")