from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
from flask import send_file
import io
import csv
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator

app = Flask(__name__)

# Load trained models
rf_tg_model = joblib.load("models/rdkit_feature_extraction_rf_model_tg.pkl")
cb_ffv_model = joblib.load("models/rdkit_feature_extraction_catboost_model_ffv.pkl")
cb_tc_model = joblib.load("models/rdkit_feature_extraction_catboost_model_tc.pkl")
cb_density_model = joblib.load("models/rdkit_feature_extraction_catboost_model_density.pkl")
cb_rg_model = joblib.load("models/rdkit_feature_extraction_catboost_model_rg.pkl")

# Load training feature names
trained_feature_names = joblib.load("models/rdkit_feature_names.pkl")

# RDKit descriptor setup
descriptor_names = [desc[0] for desc in Descriptors._descList]
calc = MolecularDescriptorCalculator(descriptor_names)

# Convert SMILES to RDKit descriptors
def smiles_to_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        desc = calc.CalcDescriptors(mol)
        df = pd.DataFrame([desc], columns=descriptor_names)

        # Clean the features
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(axis=1, how='all', inplace=True)
        df.fillna(df.median(numeric_only=True), inplace=True)

        # Align to model features
        feature_names = rf_tg_model.feature_names_in_
        df = df.reindex(columns=feature_names, fill_value=0)

        return df
    return None



@app.route("/", methods=["GET"])
def home():
    print("🟢 Home route accessed")
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    prediction = {}
    error = None
    smiles = request.form.get("smiles")

    features = smiles_to_features(smiles)
    if features is None:
        error = "Invalid SMILES string. Please try again."
    else:
        try:
            prediction = {
                "Tg": round(rf_tg_model.predict(features)[0], 4),
                "FFV": round(cb_ffv_model.predict(features)[0], 4),
                "Tc": round(cb_tc_model.predict(features)[0], 4),
                "Density": round(cb_density_model.predict(features)[0], 4),
                "Rg": round(cb_rg_model.predict(features)[0], 4),
            }
        except Exception as e:
            error = f"⚠️ Prediction error: {str(e)}"

    return render_template("result.html", prediction=prediction, error=error)
@app.route("/download", methods=["POST"])
def download():
    prediction = request.form.to_dict()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Property", "Predicted Value"])
    for key, val in prediction.items():
        writer.writerow([key, val])
    output.seek(0)
    return send_file(io.BytesIO(output.getvalue().encode()),
                     mimetype="text/csv",
                     as_attachment=True,
                     download_name="prediction_report.csv")

if __name__ == "__main__":
    app.run(debug=True)
