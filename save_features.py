import pandas as pd
import joblib

# Load the RDKit feature file (already cleaned and created in Colab)
df = pd.read_csv("models/rdkit_features.csv")

# Drop target columns — keep only features
target_cols = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
X = df.drop(columns=target_cols, errors="ignore")

# Save the feature names used during training
joblib.dump(X.columns.tolist(), "models/rdkit_feature_names.pkl")

print("✅ Saved rdkit_feature_names.pkl with", len(X.columns), "features.")
