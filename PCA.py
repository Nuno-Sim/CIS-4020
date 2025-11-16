import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ----------------------------
# 1. LOAD DATA
# ----------------------------
df = pd.read_csv("cleaned_pca_ready.csv")  # replace with your filename

# ----------------------------
# 2. DROP NON-PCA COLUMNS
# ----------------------------
df = df.drop(columns=["GameId"])  # ID column

# ----------------------------
# 3. HANDLE CATEGORICAL COLUMNS
# ----------------------------
categorical_cols = ["Formation", "PlayType", "PassType"]
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# ----------------------------
# 4. SCALE DATA
# ----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# ----------------------------
# 5. PCA
# ----------------------------
pca = PCA(n_components=0.95)  # keep 95% variance
X_pca = pca.fit_transform(X_scaled)

# ----------------------------
# 6. PCA RESULTS
# ----------------------------
print("Original shape:", df.shape)
print("PCA shape:", X_pca.shape)
print("Explained variance ratio:", pca.explained_variance_ratio_)

# Optional: create a DataFrame with PCA components
df_pca = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(X_pca.shape[1])])
df_pca.to_csv("pca_result.csv", index=False)
print("PCA result saved to pca_result.csv")


# ----------------------------
# 2. BIPLOT
# ----------------------------
plt.figure(figsize=(10, 7))

# Plot scores
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA Biplot")

# Plot loadings (arrows)
features = df.columns
for i, feature in enumerate(features):
    plt.arrow(0, 0, 
              pca.components_[0, i]*5,  # scale arrows for visibility
              pca.components_[1, i]*5,
              color='r', alpha=0.7)
    plt.text(pca.components_[0, i]*5*1.15, 
             pca.components_[1, i]*5*1.15, 
             feature, color='r', fontsize=8)

plt.grid(True)
plt.show()