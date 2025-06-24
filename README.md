# polymer-prop-pedictor
A web-based machine learning app to predict important polymer properties from SMILES strings using RDKit descriptors and trained ML models.
# Project Overview

This project predicts 5 key polymer properties based on molecular structure (SMILES format):

- **Tg** (Glass Transition Temperature)
- **FFV** (Fractional Free Volume)
- **Tc** (Critical Temperature)
- **Density**
- **Rg** (Radius of Gyration)
  
# Features

- Input: SMILES of polymer backbone  
- RDKit-based feature extraction  
- Machine learning predictions (Random Forest, CatBoost)  
- Radar chart for visualization  
- Inference box to explain results  
- Downloadable `.csv` report  
- Responsive, visually appealing frontend

# Dataset Source

This project uses data from a Kaggle competition on polymer property prediction:

> 🔗 [Polymer Property Dataset on Kaggle](https://www.kaggle.com/)

⚠️ **Note**: The dataset is not included in this repository due to licensing.  
Please download it manually from Kaggle.

# Directory Structure

polymer_predictor/
├── app.py # Flask backend
├── models/ # Trained ML models (.pkl)
│ ├── rf_model_tg.pkl
│ ├── catboost_model_ffv.pkl
| ├── catboost_model_density.pkl
| ├── catboost_model_tc.pkl
| ├── catboost_model_rg.pkl
│ └── rdkit_feature_names.pkl
├── templates/
│ ├── index.html # Homepage
│ └── result.html # Results page
├── static/
│ └── style.css # Styling
└── README.md # You are here!
