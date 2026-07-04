# Breast Cancer Classification App

A Streamlit web app for breast cancer tumor classification using a Logistic Regression model.

## Files

- `app.py` - Streamlit dashboard for dataset exploration and prediction.
- `train_model.py` - Script to train the model and save `model.pkl` and `encoder.pkl`.
- `breast-cancer.csv` - Dataset used for training and visualization.
- `style.css` - Custom CSS styling for the Streamlit app.
- `requirements.txt` - Python dependencies for the project.

## Installation

1. Create a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Train the model (if `model.pkl` and `encoder.pkl` are not already present):
   ```bash
   python train_model.py
   ```
2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Notes

- The app predicts whether a tumor is benign or malignant.
- The `train_model.py` script uses scikit-learn to train a Logistic Regression model and saves the artifacts.
