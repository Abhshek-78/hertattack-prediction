# Heart Attack Prediction

A simple Streamlit app for heart disease prediction using a pre-trained KNN model.

## Project structure

- `app.py`: Streamlit web application.
- `heart.csv`: original dataset (optional).
- `knn_heart.pkl`: serialized trained model.
- `Sscaler.pkl`: saved scaler for input normalization.
- `columns.pkl`: expected column order for the model.

## Requirements

- Python 3.8+
- streamlit
- pandas
- joblib

## Setup

1. Open terminal in the project folder:
   ```powershell
   cd "hertattack prediction"
   ```
2. (Optional) create and activate virtual environment:
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```
3. Install dependencies:
   ```powershell
   pip install streamlit pandas joblib
   ```

## Run

```powershell
streamlit run app.py
```

Then open the URL shown in terminal (typically `http://localhost:8501`).

## Usage

- Adjust the inputs in the sidebar and fields.
- Click **Predict Heart Disease Risk**.
- See result message: high risk or low risk.

## Notes

- Make sure `knn_heart.pkl`, `Sscaler.pkl`, and `columns.pkl` are present in the same folder as `app.py`.
- If any file is missing, the app will raise a `FileNotFoundError`.
