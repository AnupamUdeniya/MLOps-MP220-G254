# MP220 - Rooftop Solar Energy (Guwahati)
**Group ID:** G254  
**Batch:** 78  
**Faculty:** Upasna Joshi  

### Members:
- S24CSEU0641 - ANUPAM UDENIYA
- S24CSEU0440 - Sachit Dadwal
- S24CSEU0392 - SANCHAY JAIN
- S24CSEU0360 - NAMAN SHARMA

## MLflow Experiment Tracking

The repository includes an MLflow experiment database and five logged model artifacts.

### Setup (PowerShell)

1. Create a virtual environment: python -m venv .venv
2. Activate it: .\.venv\Scripts\Activate.ps1
3. Install dependencies: pip install -r requirements-mlflow.txt
4. Repair local artifact paths: python scripts/prepare_mlflow.py
5. Start the dashboard: mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5002 --workers 1
6. Open http://127.0.0.1:5002

### Experiment Details

Experiment: mp220_solar_prediction
Registered model: mp220_solar_model
Version: 1
Alias: champion
V1 validation MAE: approximately 1.47984

The experiment contains five completed runs, including four candidate models and the final V1 registration run. Press Ctrl+C in the server terminal to stop the dashboard.
