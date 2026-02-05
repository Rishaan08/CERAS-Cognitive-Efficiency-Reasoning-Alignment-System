#Imports
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression, RFE
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from lightgbm import LGBMRegressor
from boruta import BorutaPy

#Path
artifact_dir = "./artifacts"
os.makedirs(artifact_dir, exist_ok=True)

id_col = "cntstuid"
target = "ce_score"

#Phase A: PISA Only (Cognitive)
print("\nLoading PISA student features")
pisa_df = pd.read_parquet("./data/processed/pisa_student.parquet")
pisa_df.columns = pisa_df.columns.str.lower()

X_pisa = pisa_df.drop(columns=[id_col, target])
y_pisa = pisa_df[target]

print("PISA feature count:", X_pisa.shape[1])

# ---------------- Layer 1 — Mutual Information ----------------
mi = mutual_info_regression(X_pisa, y_pisa, random_state=42)
mi_scores = pd.Series(mi, index=X_pisa.columns)

selected_l1 = mi_scores[mi_scores >= np.percentile(mi_scores, 50)].index
X_l1 = X_pisa[selected_l1]

print("\n[PISA] Layer 1 — Mutual Information")
print("Retained:", len(selected_l1))
print(selected_l1.tolist())

# ---------------- Layer 2 — RFE ----------------
rfe = RFE(
    estimator=LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        random_state=42,
        n_jobs=-1
    ),
    n_features_to_select=min(10, X_l1.shape[1]),
    step=1
)

rfe.fit(X_l1, y_pisa)
selected_l2 = X_l1.columns[rfe.support_]
X_l2 = X_l1[selected_l2]

print("\n[PISA] Layer 2 — RFE")
print("Retained:", len(selected_l2))
print(selected_l2.tolist())

# ---------------- Layer 3 — LASSO ----------------
scaler_lasso = StandardScaler()
X_l2_scaled = scaler_lasso.fit_transform(X_l2)

lasso = LassoCV(
    alphas=np.logspace(-4, 0, 50),
    cv=5,
    random_state=42,
    n_jobs=-1,
)

lasso.fit(X_l2_scaled, y_pisa)

lasso_coef = pd.Series(lasso.coef_, index=X_l2.columns)
selected_l3 = lasso_coef[lasso_coef.abs() > 1e-4].index
X_l3 = X_l2[selected_l3]

print("\n[PISA] Layer 3 — LASSO")
print("Retained:", len(selected_l3))
print(selected_l3.tolist())

# ---------------- Layer 4 — Boruta ----------------
boruta = BorutaPy(
    estimator=RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ),
    n_estimators="auto",
    perc=70,
    max_iter=30,
    random_state=42,
    verbose=0
)

boruta.fit(X_l3.values, y_pisa.values)
pisa_features = X_l3.columns[boruta.support_].tolist()

print("\n[PISA] Layer 4 — Boruta")
print("Final PISA features:", pisa_features)

# ---------------- Impute + MinMax Scale ----------------
pisa_refined = pisa_df[[id_col, target] + pisa_features].copy()

imputer = IterativeImputer(
    estimator=ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    max_iter=10,
    random_state=42)

pisa_refined[pisa_features] = imputer.fit_transform(pisa_refined[pisa_features])
pisa_refined[pisa_features] = MinMaxScaler(feature_range=(0, 1)).fit_transform(pisa_refined[pisa_features])

# ============================================================
# Phase 0B — QQQ ONLY
# ============================================================
print("\nLoading QQQ student features")
qqq_df = pd.read_parquet("./data/processed/qqq_student_features.parquet")
qqq_df.columns = qqq_df.columns.str.lower()

X_qqq = qqq_df.drop(columns=[id_col])
y_qqq = pisa_df.set_index(id_col).loc[qqq_df[id_col], target].values

# ---------------- Layer 1 — Mutual Information ----------------
mi = mutual_info_regression(X_qqq, y_qqq, random_state=42)
mi_scores = pd.Series(mi, index=X_qqq.columns)

selected_l1 = mi_scores[mi_scores >= np.percentile(mi_scores, 65)].index
X_l1 = X_qqq[selected_l1]

print("\n[QQQ] Layer 1 — Mutual Information")
print("Retained:", len(selected_l1))
print(selected_l1.tolist())

# ---------------- Layer 2 — RFE ----------------
rfe.fit(X_l1, y_qqq)
selected_l2 = X_l1.columns[rfe.support_]
X_l2 = X_l1[selected_l2]

print("\n[QQQ] Layer 2 — RFE")
print("Retained:", len(selected_l2))
print(selected_l2.tolist())

# ---------------- Layer 3 — LASSO ----------------
X_l2_scaled = scaler_lasso.fit_transform(X_l2)
lasso.fit(X_l2_scaled, y_qqq)

lasso_coef = pd.Series(lasso.coef_, index=X_l2.columns)
selected_l3 = lasso_coef[lasso_coef.abs() > 1e-4].index
X_l3 = X_l2[selected_l3]

print("\n[QQQ] Layer 3 — LASSO")
print("Retained:", len(selected_l3))
print(selected_l3.tolist())

# ---------------- Layer 4 — Boruta ----------------
boruta.fit(X_l3.values, y_qqq)
qqq_features = X_l3.columns[boruta.support_].tolist()

print("\n[QQQ] Layer 4 — Boruta")
print("Final QQQ features:", qqq_features)

# ---------------- Impute + MinMax Scale ----------------
qqq_refined = qqq_df[[id_col] + qqq_features].copy()
qqq_refined[qqq_features] = imputer.fit_transform(qqq_refined[qqq_features])
qqq_refined[qqq_features] = MinMaxScaler(feature_range=(0, 1)).fit_transform(qqq_refined[qqq_features])

# ------------------------------------------------------------
# Final ANFIS feature set (NO artificial limit)
# ------------------------------------------------------------
print("Safe merge")
df = pisa_refined.merge(qqq_refined, on=id_col, how="inner")

final_features = pisa_features + qqq_features

np.save(f"{artifact_dir}/anfis_features.npy", np.array(final_features))

print("Final merged feature count:", len(final_features))
print("Final features:", final_features)

print("Final ANFIS features:")
print(final_features)
print("Total features:", len(final_features))

# ------------------------------------------------------------
# Train / Test split (ANFIS)
# ------------------------------------------------------------
X = df[final_features].values
y = df[target].values
X_train_pool, X_test, y_train_pool, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(X_train_pool, y_train_pool, test_size=0.20, random_state=42)

#ANFIS Model
class SimpleANFIS:
    def __init__(self, n_inputs, n_mfs=2):
        self.n_inputs = n_inputs
        self.n_mfs = n_mfs
        self.rules = n_mfs ** n_inputs
        self.mf_params = {
            i: {
                j: {"mean": 0.5, "sigma": 0.25}
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
            rs = []
            for rule in range(self.rules):
                s = 1.0
                tmp = rule
                for i in range(self.n_inputs):
                    mf = tmp % self.n_mfs
                    tmp //= self.n_mfs
                    p = self.mf_params[i][mf]
                    s *= self.gaussian_mf(x[i], p["mean"], p["sigma"])
                rs.append(s)
            firing.append(rs)
        firing = np.array(firing)
        return firing / (np.sum(firing, axis=1, keepdims=True) + 1e-8)

    #ANFIS Architecture: Layer 4 : Consequent Layer
    def fit(self, X, y):
        W = self.forward(X)
        X_aug = np.hstack([X, np.ones((X.shape[0], 1))])
        Phi = np.vstack([
            np.hstack([W[i, r] * X_aug[i] for r in range(W.shape[1])])
            for i in range(X.shape[0])
        ])
        self.consequents = np.linalg.pinv(Phi) @ y

    #ANFIS Architecture: Layer 5 : Defuzzification
    def predict(self, X):
        W = self.forward(X)
        X_aug = np.hstack([X, np.ones((X.shape[0], 1))])
        Phi = np.vstack([
            np.hstack([W[i, r] * X_aug[i] for r in range(W.shape[1])])
            for i in range(X.shape[0])
        ])
        return Phi @ self.consequents

print("\nTraining & Evaluating ANFIS")
anfis = SimpleANFIS(n_inputs=len(final_features))
anfis.fit(X_train, y_train)

for name, Xs, ys in [
    ("Train", X_train, y_train),
    ("Validation", X_val, y_val),
    ("Test", X_test, y_test)
]:
    yp = anfis.predict(Xs)
    print(f"\n{name}")
    print("MAE:", mean_absolute_error(ys, yp))
    print("R2 :", r2_score(ys, yp))

joblib.dump(anfis, f"{artifact_dir}/anfis_model.pkl")