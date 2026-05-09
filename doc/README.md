# ML Compass

ML Compass is a beginner-friendly expert system that helps students choose a suitable machine learning problem type, baseline models, metrics, preprocessing steps, and starter code.

## Run

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m streamlit run app.py
```

## Current Scope

- Classification
- Regression
- Clustering
- Time-series forecasting
- Basic NLP

This first version uses a JSON knowledge base and a simple rule engine. It does not train models, inspect uploaded datasets, or use an LLM.
