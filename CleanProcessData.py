import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# ----------------------------
# 1. LOAD DATA
# ----------------------------
df = pd.read_csv("./data/pbp-2019.csv", low_memory=False)

# Replace empty strings with NaN
df.replace("", np.nan, inplace=True)

# ----------------------------
# 2. REMOVE IRRELEVANT COLUMNS
# ----------------------------
drop_cols = [
    "GameDate",
    "Description",
    "PenaltyType",
    "Challenger",
    ""  # handles empty-column-name in your sample
]

df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

# ----------------------------
# 3. HANDLE MISSING VALUES
# ----------------------------
numeric_cols = df.select_dtypes(include=["number"]).columns
df[numeric_cols] = df[numeric_cols].fillna(0)

categorical_cols = df.select_dtypes(include=["object"]).columns
df[categorical_cols] = df[categorical_cols].fillna("UNKNOWN")

# ----------------------------
# 4. FILTER TO SUCCESSFUL 4TH DOWNS BEFORE DROPPING OUTCOME COLUMNS
# ----------------------------

# Ensure outcome columns exist before filtering
def exists(col): 
    return col in df.columns

df_success = df[
    (df["Down"] == 4) &                         # must be 4th down
    ( (not exists("IsFumble")) | (df.get("IsFumble", 0) == 0) ) &
    ( (not exists("IsInterception")) | (df.get("IsInterception", 0) == 0) ) &
    ( (not exists("IsIncomplete")) | (df.get("IsIncomplete", 0) == 0) ) &
    ( (not exists("IsNoPlay")) | (df.get("IsNoPlay", 0) == 0) )
].copy()

# ----------------------------
# 5. REMOVE NON-PCA COLUMNS (IDs, etc.)
# ----------------------------
id_cols = ["GameId"]
cols_to_drop = [
    "Unnamed: 10",
    "SeriesFirstDown",
    "Unnamed: 12",
    "NextScore",
    "TeamWin",
    "Unnamed: 16",
    "Unnamed: 17"
]

df_success = df_success.drop(columns=[c for c in cols_to_drop if c in df_success.columns], errors="ignore")

cols_to_drop = [
    "OffenseTeam", "DefenseTeam",  # team labels

    # outcome / leak columns (safe to remove now)
    "IsTouchdown", "IsIncomplete", "IsInterception", "IsFumble",
    "IsTwoPointConversion", "IsTwoPointConversionSuccessful",
    "IsPenalty", "IsPenaltyAccepted", "IsNoPlay", "IsChallenge",
    "IsChallengeReversed", "IsMeasurement", "PenaltyTeam", "PenaltyYards",

    # categorical / redundant
    "YardLineDirection", "PassType", "RushDirection",
    "Formation", "PlayType",

    # redundant always-zero
    "IsRush", "IsPass", "IsSack",

    # duplicate yardline transformation
    "YardLineFixed",
]

df_success = df_success.drop(columns=[c for c in cols_to_drop if c in df_success.columns], errors="ignore")

# ----------------------------
# 6. SAVE CLEANED DATA
# ----------------------------
df_success.to_csv("cleaned_pca_ready.csv", index=False)

print("Finished! PCA-ready dataset saved as cleaned_pca_ready.csv")


# ----------------------------
# 7. LOAD CLEANED DATA
# ----------------------------
df = pd.read_csv("cleaned_pca_ready.csv")

# Drop ID columns (not used in PCA)
id_cols = ["GameId"]
df_features = df.drop(columns=[c for c in id_cols if c in df.columns], errors="ignore")

# ----------------------------
# 8. STANDARDIZE FEATURES
# ----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_features)

# ----------------------------
# 9. PERFORM PCA (keep 95% variance)
# ----------------------------
from sklearn.decomposition import PCA

pca = PCA(n_components=0.95)   # keep components that explain 95%
X_pca = pca.fit_transform(X_scaled)

# ----------------------------
# 10. SAVE PCA-REDUCED DATASET
# ----------------------------
df_pca = pd.DataFrame(
    X_pca,
    columns=[f"PC{i+1}" for i in range(X_pca.shape[1])]
)

# Add back ID columns
for col in id_cols:
    if col in df.columns:
        df_pca[col] = df[col].values

df_pca.to_csv("pca_reduced.csv", index=False)

# ----------------------------
# 11. PRINT EXPLAINED VARIANCE
# ----------------------------
print("\nExplained Variance Ratio:")
for i, v in enumerate(pca.explained_variance_ratio_):
    print(f"  PC{i+1}: {v:.4f}")

print(f"\nTotal Variance Explained: {pca.explained_variance_ratio_.sum():.4f}")
print("Saved PCA dataset as pca_reduced.csv")
