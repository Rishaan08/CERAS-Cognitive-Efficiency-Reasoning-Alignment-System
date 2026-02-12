#Imports
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.cluster import KMeans
from sklearn.feature_selection import mutual_info_regression, RFE
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from lightgbm import LGBMRegressor
from boruta import BorutaPy
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Reshape, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import shap
import matplotlib.pyplot as plt

#Paths
graph_dir = "./graphs"
artifact_dir = "./artifacts"
os.makedirs(graph_dir, exist_ok=True)
os.makedirs(artifact_dir, exist_ok=True)

#Load OULAD + MEU dataset
df = pd.read_parquet("./data/processed/features_final.parquet")
df = df[df["cognitive_efficiency"].notna()].reset_index(drop=True)

target = "cognitive_efficiency"
id_col = "student_id"

print("Dataset loaded:", df.shape)

#Missing Value Handling
missing_ratio = df.isnull().mean()

low_missing = missing_ratio[(missing_ratio >= 0) & (missing_ratio <= 0.30)].index.tolist()
mid_missing = missing_ratio[(missing_ratio > 0.30) & (missing_ratio <= 0.80)].index.tolist()
high_missing = missing_ratio[(missing_ratio > 0.80) & (missing_ratio <= 0.95)].index.tolist()
extreme_missing = missing_ratio[(missing_ratio > 0.95)].index.tolist()

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

for col in mid_missing:
    df[f"{col}_is_present"] = df[col].notnull().astype(int)
    df[col] = df[col].fillna(df[col].median())

for col in high_missing:
    df[f"{col}_is_present"] = df[col].notnull().astype(int)
    df[col] = df[col].fillna(0)

df.drop(columns=extreme_missing, inplace=True)

#Feature matrix
X_full = df.drop(columns=[id_col, target])
y = df[target]

#Ensure numeric-only features
non_numeric_cols = X_full.select_dtypes(exclude=["number"]).columns.tolist()

if len(non_numeric_cols) > 0:
    print("Dropping non-numeric columns:", non_numeric_cols)
    X_full = X_full.drop(columns=non_numeric_cols)

#Cognitive Feature Refinement Selection (Four-Layer Pipeline)
#MI → RFE → LASSO → Boruta
#Layer 1: Mutual Information
mi = mutual_info_regression(X_full, y, random_state=42)
mi_scores = pd.Series(mi, index=X_full.columns)

mi_threshold = np.percentile(mi_scores, 65)
selected_l1 = mi_scores[mi_scores >= mi_threshold].index
X_l1 = X_full[selected_l1]

print("Layer 1 retained:", len(selected_l1))

#Layer 2: Recursive Feature Elimination
rfe = RFE(
    estimator=LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        random_state=42,
        n_jobs=-1
    ),
    n_features_to_select=min(20, X_l1.shape[1]),
    step=1
)
rfe.fit(X_l1, y)
selected_l2 = X_l1.columns[rfe.support_]
X_l2 = X_l1[selected_l2]

print("Layer 2 retained:", len(selected_l2))

#Layer 3: LASSO
scaler_lasso = StandardScaler()
X_l2_scaled = scaler_lasso.fit_transform(X_l2)

lasso = LassoCV(
    alphas=np.logspace(-4, 0, 50),
    cv=5,
    random_state=42,
    n_jobs=-1
)
lasso.fit(X_l2_scaled, y)

lasso_coef = pd.Series(lasso.coef_, index=X_l2.columns)
selected_l3 = lasso_coef[lasso_coef.abs() > 1e-4].index
X_l3 = X_l2[selected_l3]

print("Layer 3 retained:", len(selected_l3))

#Layer 4: Boruta (Optional)
if X_l3.shape[1] > 10:
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
    boruta.fit(X_l3.values, y.values)
    selected_l4 = X_l3.columns[boruta.support_]

    if len(selected_l4) < 10:
        selected_l4 = selected_l3
else:
    selected_l4 = selected_l3

print("Final CNN features:", selected_l4.tolist())

np.save(f"{artifact_dir}/cnn_features.npy", selected_l4.to_numpy())

#CNN Data Preparation
X = df[selected_l4].values

X_train_pool, X_test, y_train_pool, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_pool, y_train_pool, test_size=0.20, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

#CNN Model
n_features = X_train.shape[1]

model = Sequential([
    Reshape((n_features, 1), input_shape=(n_features,)),
    Conv1D(32, 3, activation="relu", padding="same"),
    MaxPooling1D(2),
    Conv1D(64, 3, activation="relu", padding="same"),
    MaxPooling1D(2),
    Flatten(),
    Dense(128, activation="relu"),
    BatchNormalization(),
    Dropout(0.4),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(1)
])

model.compile(
    optimizer=Adam(0.001),
    loss="mse",
    metrics=["mae"]
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True
)

model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=64,
    callbacks=[early_stop],
    verbose=1
)

#Evaluation
def evaluate(name, Xs, ys):
    preds = model.predict(Xs)
    print(f"\n{name}")
    print("MAE:", mean_absolute_error(ys, preds))
    print("R2 :", r2_score(ys, preds))
    return preds

y_train_pred = evaluate("Train", X_train, y_train)
y_val_pred   = evaluate("Validation", X_val, y_val)
y_test_pred  = evaluate("Test", X_test, y_test)

#Save model
model.save(os.path.join(artifact_dir, "cnn_ce_model.keras"))

np.save("./data/validation/y_val_true.npy", y_val)
np.save("./data/validation/y_val_pred_cnn.npy", y_val_pred)

#SHAP + Intention Modeling
def cnn_predict(x):
    return model.predict(x).reshape(-1)

background = X_train[np.random.choice(X_train.shape[0], 100, replace=False)]
explainer = shap.KernelExplainer(cnn_predict, background)
shap_values = explainer.shap_values(X_test[:50], nsamples=200)

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test[:50], feature_names=selected_l4.tolist(), show=False)
plt.savefig(f"{graph_dir}/shap_summary.png", bbox_inches="tight")
plt.close()

embedding_layer = model.layers[-5].output
embedding_model = Model(inputs=model.layers[0].input, outputs=embedding_layer)

embeddings = embedding_model.predict(X_test[:50])
clusters = KMeans(n_clusters=4, random_state=42).fit_predict(embeddings)

pd.DataFrame({
    "cluster": clusters,
    "ce_score": y_test.iloc[:50].values
}).groupby("cluster").mean().to_csv(
    f"{graph_dir}/intention_cluster_means.csv"
)

cluster_mean = pd.read_csv(
    f"{graph_dir}/intention_cluster_means.csv",
    index_col=0
)

plt.figure(figsize=(8, 5))

colors = ["#4CAF50", "#FF9800", "#03A9F4", "#9C27B0"]

cluster_mean["ce_score"].plot(
    kind="bar",
    color=colors[:len(cluster_mean)],
    edgecolor="black"
)

plt.title("Intention Clusters vs Cognitive Efficiency", fontsize=14)
plt.xlabel("Intention Cluster", fontsize=12)
plt.ylabel("Mean Cognitive Efficiency", fontsize=12)
plt.xticks(rotation=0)
plt.grid(axis="y", linestyle="--", alpha=0.6)

#Value labels
for i, v in enumerate(cluster_mean["ce_score"]):
    plt.text(i, v + 0.02 * cluster_mean["ce_score"].max(),
             f"{v:.2f}", ha="center", fontsize=11)

plt.tight_layout()
plt.savefig(f"{graph_dir}/intention_cluster_means.png", dpi=300)
plt.close()
