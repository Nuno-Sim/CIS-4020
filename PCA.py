import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def simple_feature_reduction(csv_file_path):
    """
    Simple PCA to identify which features to KEEP and which to REMOVE
    """
    df = pd.read_csv(csv_file_path)
    
    # Select features
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = ['GameId', 'SeasonYear', 'ConversionSuccess', 'FieldGoalSuccess', 'PlaySuccess']
    features = [f for f in numeric_features if f not in exclude]
    
    X = df[features].fillna(df[features].median())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit PCA to understand feature importance
    pca = PCA()
    pca.fit(X_scaled)
    
    # Calculate feature importance
    loadings = pca.components_
    feature_importance = np.sum(np.abs(loadings), axis=0)
    
    # Create feature ranking
    importance_df = pd.DataFrame({
        'feature': features,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    print("🎯 FEATURE REDUCTION RECOMMENDATIONS:")
    print("=" * 50)
    
    # Recommend which features to KEEP
    keep_threshold = importance_df['importance'].quantile(0.5)  # Keep top 50%
    features_to_keep = importance_df[importance_df['importance'] >= keep_threshold]['feature'].tolist()
    
    print(f"\n✅ KEEP these {len(features_to_keep)} features (high importance):")
    for feature in features_to_keep:
        importance = importance_df[importance_df['feature'] == feature]['importance'].values[0]
        print(f"   - {feature} (importance: {importance:.3f})")
    
    # Recommend which features to REMOVE
    features_to_remove = importance_df[importance_df['importance'] < keep_threshold]['feature'].tolist()
    
    print(f"\n❌ REMOVE these {len(features_to_remove)} features (low importance):")
    for feature in features_to_remove:
        importance = importance_df[importance_df['feature'] == feature]['importance'].values[0]
        print(f"   - {feature} (importance: {importance:.3f})")
    
    # Create reduced dataset
    df_reduced = df[['GameId', 'SeasonYear', 'OffenseTeam', 'ConversionSuccess'] + features_to_keep]
    
    print(f"\n📊 Reduced from {len(features)} to {len(features_to_keep)} features")
    print(f"📁 New dataset shape: {df_reduced.shape}")
    
    return df_reduced, importance_df

# Run it

# Run the analysis
if __name__ == "__main__":
    # Replace with your actual file path
    csv_file = "fourth_down_processed_2019_2024.csv"  # or your actual file name
    
    try:
        df_reduced, importance_df = simple_feature_reduction("fourth_down_processed_2019_2024.csv")

        # Save the reduced dataset
        df_reduced.to_csv("fourth_down_reduced_features.csv", index=False)
        print("💾 Saved reduced dataset to: fourth_down_reduced_features.csv")

        
    except FileNotFoundError:
        print(f"❌ File not found: {csv_file}")
        print("Please make sure the CSV file exists in the current directory.")
    except Exception as e:
        print(f"❌ Error during PCA analysis: {e}")
        import traceback
        traceback.print_exc()