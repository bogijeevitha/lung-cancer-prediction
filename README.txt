# LungCare AI - Lung Cancer Prediction (Academic Prototype)

## Files
- `app.py` - Gradio website with Home, Patient Input, Prediction Output, Model Information and About tabs.
- `evaluate_models.py` - evaluates Logistic Regression, SVM and Random Forest using stratified 5-fold cross-validation.
- `lung_cancer_examples.csv` - the supplied project dataset.
- `requirements.txt` - Python dependencies.

## Run on Windows
1. Install Python 3.10 or newer.
2. Extract this ZIP file.
3. Open Command Prompt in the extracted project folder.
4. Run:
   ```
   py -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   python app.py
   ```
5. Open the local URL printed in Command Prompt (usually `http://127.0.0.1:7860`).

## Evaluate models
Run:
```
python evaluate_models.py
```

## Important dataset notes
The dataset has 59 records and includes the columns `Name`, `Surname`, `Age`, `Smokes`, `AreaQ`, `Alkhol`, and `Result`.
Only `Age`, `Smokes`, `AreaQ`, and `Alkhol` are used as predictors. Names are excluded.
The meaning of target labels 0 and 1 must be confirmed from the original dataset source before describing them as cancer/no-cancer.

This is a classroom demonstration only. The dataset is very small and does not establish clinical accuracy. The application is not a medical diagnostic device.
