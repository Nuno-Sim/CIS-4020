import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import glob
import os

# ----------------------------
# 1. LOAD DATA FOR 2019–2024
# ----------------------------
all_dfs = []

for year in range(2019, 2025):   # 2019, 2020, 2021, 2022, 2023, 2024
    path = f"./data/pbp-{year}.csv"
    if os.path.exists(path):
        df_year = pd.read_csv(path, low_memory=False)
        df_year["SeasonYear"] = year     # add year column
        all_dfs.append(df_year)
    else:
        print(f"Warning: {path} not found, skipping.")

# Combine all years
df = pd.concat(all_dfs, ignore_index=True)

# Replace empty strings with NaN
df.replace("", np.nan, inplace=True)

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
    "YardLineDirection", "RushDirection",
    # duplicate yardline transformation
    "YardLineFixed",
]

df_success = df_success.drop(columns=[c for c in cols_to_drop if c in df_success.columns], errors="ignore")

# ----------------------------
# 6. SAVE CLEANED DATA
# ----------------------------
df_success.to_csv("cleaned_pca_ready.csv", index=False)

print("Finished! PCA-ready dataset saved as cleaned_pca_ready.csv")


