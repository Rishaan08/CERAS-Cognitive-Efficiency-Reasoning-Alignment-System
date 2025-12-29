#Imports
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.cluster import KMeans
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Reshape, Conv1D, MaxPooling1D,Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import shap
import matplotlib.pyplot as plt

# Paths
graph_dir = "./graphs"
artifact_dir = "./artifacts"
os.makedirs(graph_dir, exist_ok=True)
os.makedirs(artifact_dir, exist_ok=True)

# Load data
df = pd.read_parquet("./data/processed/features_final.parquet")
df = df[df["cognitive_efficiency"].notna()].reset_index(drop=True)

anfis_features = np.load(
    "./artifacts/anfis_features_l4.npy",
    allow_pickle=True
).tolist()

target = "cognitive_efficiency"

X = df[anfis_features].values
y = df[target].values

#Train / Val / Test split
X_train_pool, X_test, y_train_pool, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_pool, y_train_pool, test_size=0.20, random_state=42
)

#Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

#CNN Model
n_features = X_train.shape[1]

model = Sequential([
    Reshape((n_features, 1), input_shape=(n_features,)),

    Conv1D(32, kernel_size=3, activation="relu", padding="same"),
    MaxPooling1D(pool_size=2),

    Conv1D(64, kernel_size=3, activation="relu", padding="same"),
    MaxPooling1D(pool_size=2),

    Flatten(),
    Dense(128, activation="relu"),
    BatchNormalization(),
    Dropout(0.4),

    Dense(64, activation="relu"),
    Dropout(0.3),

    Dense(1, activation="linear")
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="mse",
    metrics=["mae"]
)

#Early Stopping
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=64,
    callbacks=[early_stop],
    verbose=1
)

#Evaluation
def evaluate(split_name, X_split, y_split):
    preds = model.predict(X_split)
    print(f"\n{split_name}")
    print("MAE:", mean_absolute_error(y_split, preds))
    print("R2 :", r2_score(y_split, preds))

evaluate("Train", X_train, y_train)
evaluate("Validation", X_val, y_val)
evaluate("Test", X_test, y_test)

#Save model
model.save(os.path.join(artifact_dir, "cnn_ce_model.keras"))

#SHAP Explainability
def cnn_predict(x):
    return model.predict(x).reshape(-1)

background = X_train[np.random.choice(X_train.shape[0], 100, replace=False)]

explainer = shap.KernelExplainer(cnn_predict, background)

shap_values = explainer.shap_values(X_test[:50], nsamples=200)

#SHAP summary
plt.figure(figsize=(10, 6))
shap.summary_plot(
    shap_values,
    X_test[:50],
    feature_names=anfis_features,
    show=False
)
plt.tight_layout()
plt.savefig(f"{graph_dir}/shap_summary.png", bbox_inches="tight")
plt.close()

#SHAP bar
plt.figure(figsize=(10, 6))
shap.summary_plot(
    shap_values, X_test[:50],
    feature_names=anfis_features,
    plot_type="bar",
    show=False
)
plt.tight_layout()
plt.savefig(f"{graph_dir}/shap_bar.png", bbox_inches="tight")
plt.close()

#SHAP force
i = 0  
plt.figure(figsize=(14, 3)) 
shap.force_plot(
    explainer.expected_value,
    shap_values[i],
    X_test[i],
    feature_names=anfis_features,
    matplotlib=True
)
plt.title(f"SHAP Force Plot – Test Sample {i}", fontsize=12)
plt.tight_layout()

plt.savefig(
    f"{graph_dir}/force_plot_sample_{i}.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

#Human Intention Modeling
embedding_layer = model.layers[-5].output
embedding_model = Model(
    inputs=model.layers[0].input,
    outputs=embedding_layer
)

embeddings = embedding_model.predict(X_test[:50])
print("Embedding shape:", embeddings.shape)

kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(embeddings)

intention_df = pd.DataFrame({
    "cluster": clusters,
    "ce_score": y_test[:50]
})

cluster_mean = intention_df.groupby("cluster").mean()

cluster_mean.to_csv(
    f"{graph_dir}/intention_cluster_means.csv"
)

print("\nIntention Cluster Means")
print(cluster_mean)
intention_df.groupby("cluster").mean().to_csv(
    f"{graph_dir}/intention_cluster_means.csv"
)

plt.figure(figsize=(8, 5))
cluster_mean["ce_score"].plot(
    kind="bar",
    color=["#4CAF50", "#FF9800", "#03A9F4", "#9C27B0"],
    edgecolor="black"
)
plt.title("CER-45: Intention Clusters vs Cognitive Efficiency", fontsize=14)
plt.xlabel("Intention Cluster", fontsize=12)
plt.ylabel("Mean Cognitive Efficiency", fontsize=12)
plt.xticks(rotation=0)
plt.grid(axis="y", linestyle="--", alpha=0.6)

#Add value labels on bars
for i, v in enumerate(cluster_mean["ce_score"]):
    plt.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=10)
plt.tight_layout()
plt.savefig(f"{graph_dir}/intention_cluster_means.png", dpi=300)
plt.close()