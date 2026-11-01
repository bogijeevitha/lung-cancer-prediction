import os from pathlib import Path
import pandas as pd
import gradio as gr
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "lung_cancer_examples.csv"

FEATURES = ["Age", "Smokes", "AreaQ", "Alkhol"]
TARGET = "Result"

def load_and_train():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}. Keep lung_cancer_examples.csv beside app.py."
        )

    df = pd.read_csv(DATA_PATH)
    required = FEATURES + [TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    # Use only the actual feature columns; do not use names/identifiers.
    data = df[required].copy()
    for col in required:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna()

    if data.empty or data[TARGET].nunique() < 2:
        raise ValueError("The dataset must contain valid features and at least two target classes.")

    X = data[FEATURES]
    y = data[TARGET].astype(int)

    # Cross-validation is used for a more stable classroom estimate.
    min_class_count = int(y.value_counts().min())
    n_splits = min(5, min_class_count)
    if n_splits < 2:
        raise ValueError("Not enough examples in each class for stratified cross-validation.")

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    evaluation_model = RandomForestClassifier(
        n_estimators=200, random_state=42, class_weight="balanced"
    )
    scores = cross_val_score(evaluation_model, X, y, cv=cv, scoring="accuracy")

    # Final demonstration model is fitted on all available rows.
    final_model = RandomForestClassifier(
        n_estimators=200, random_state=42, class_weight="balanced"
    )
    final_model.fit(X, y)

    return final_model, len(data), sorted(y.unique().tolist()), float(scores.mean()), float(scores.std())

model, row_count, class_labels, cv_accuracy, cv_std = load_and_train()

def predict(age, smokes, areaq, alkhol):
    values = [age, smokes, areaq, alkhol]
    names = ["Age", "Smokes", "AreaQ", "Alkhol"]

    if any(v is None for v in values):
        return "Please enter all four values."

    try:
        numeric = [float(v) for v in values]
    except (TypeError, ValueError):
        return "Please enter valid numeric values."

    if not all(np.isfinite(numeric)):
        return "Please enter finite numeric values."

    # Prevent obviously invalid age input; other feature scales must follow the CSV.
    if numeric[0] < 1 or numeric[0] > 120:
        return "Age must be between 1 and 120. For other fields, use the dataset's scale."

    row = pd.DataFrame([dict(zip(FEATURES, numeric))], columns=FEATURES)
    predicted = int(model.predict(row)[0])

    probability_text = ""
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(row)[0]
        probability_text = "\n\nModel class probabilities:\n" + "\n".join(
            f"Class {label}: {prob:.1%}"
            for label, prob in zip(model.classes_, probabilities)
        )

    return (
        f"Predicted dataset class: {predicted}"
        f"{probability_text}\n\n"
        "Important: This is an academic demonstration using a very small dataset. "
        "The meanings of class 0 and class 1 must be confirmed from the dataset source. "
        "This output is NOT a medical diagnosis."
    )

def dataset_summary():
    df = pd.read_csv(DATA_PATH)
    return (
        f"Dataset records used: {row_count}\n"
        f"Input features: {', '.join(FEATURES)}\n"
        f"Target column: {TARGET}\n"
        f"Target classes found: {class_labels}\n"
        f"Stratified cross-validation accuracy: {cv_accuracy:.3f} "
        f"(standard deviation {cv_std:.3f})"
    )

css = """
.gradio-container {max-width: 1050px !important; margin: auto !important;}
#hero {
  padding: 25px; border-radius: 16px;
  background: linear-gradient(120deg, #123b35, #287a68);
  color: white; margin-bottom: 14px;
}
"""

with gr.Blocks(
    title="LungCare AI - Lung Cancer Prediction",
    theme=gr.themes.Soft(),
    css=css
) as demo:
    gr.HTML("""
    <div id="hero">
      <h1>🫁 LungCare AI</h1>
      <h3>Lung Cancer Prediction System</h3>
      <p>Academic machine-learning project | CSI Wesley Institute of Technology & Sciences</p>
    </div>
    """)

    with gr.Tabs():
        with gr.Tab("🏠 Home"):
            gr.Markdown("""
            ## Welcome
            This application demonstrates a machine-learning classifier trained with
            the supplied structured CSV dataset.

            **How it works:** Open Patient Input → enter the four dataset features →
            click Predict → read the predicted dataset class.

            **Project team:** B. Jeevitha · R. Manisha Shree · P. Sai Srujan  
            **Project guide:** Mr. Abdul Subhan
            """)
            gr.Markdown("### Project workflow")
            gr.Markdown("Dataset → Data validation → Feature selection → Model training → Prediction → Result")

        with gr.Tab("📝 Patient Input"):
            gr.Markdown("""
            ## Enter sample feature values
            Enter values using the same scale as the dataset. These fields are model
            features, not a substitute for clinical examination.
            """)
            with gr.Row():
                age = gr.Number(label="Age", value=45, minimum=1, maximum=120, precision=0)
                smokes = gr.Number(label="Smokes (dataset scale)", value=5)
            with gr.Row():
                areaq = gr.Number(label="AreaQ (dataset scale)", value=5)
                alkhol = gr.Number(label="Alkhol (dataset scale)", value=3)
            predict_btn = gr.Button("🔍 Predict", variant="primary")
            result = gr.Textbox(label="Prediction Output", lines=8, interactive=False)
            predict_btn.click(
                fn=predict,
                inputs=[age, smokes, areaq, alkhol],
                outputs=result
            )

        with gr.Tab("📊 Model Information"):
            gr.Markdown("## Dataset and evaluation summary")
            summary = gr.Textbox(value=dataset_summary(), label="Training summary", lines=7, interactive=False)
            gr.Markdown("""
            **Evaluation note:** The reported cross-validation score is an internal estimate
            on a dataset of only 59 records. It is not evidence of real-world diagnostic
            performance. Do not use this application to make healthcare decisions.
            """)

        with gr.Tab("ℹ️ About"):
            gr.Markdown("""
            ## About the Project
            **Title:** Predictive Modeling for Lung Cancer Detection Using Deep Learning

            This working application currently uses a Random Forest classifier on the
            four numerical columns available in the supplied CSV. A CNN is a separate
            proposed extension and is not claimed as the model running in this app.

            **Guide:** Mr. Abdul Subhan  
            **Students:** B. Jeevitha, R. Manisha Shree, P. Sai Srujan  
            **College:** CSI Wesley Institute of Technology & Sciences
            """)

if __name__ == "__main__":
    demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 10000))
)

