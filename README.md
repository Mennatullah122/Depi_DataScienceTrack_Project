# 🩺 Breast Cancer Classification App

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?style=flat-square)
![Scikit--Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

A comprehensive Streamlit web application for breast cancer tumor classification using a Logistic Regression model with GridSearch optimization. This project combines interactive data exploration, visualization, and real-time predictions with support for batch processing.

## 📋 Table of Contents

- [Overview](#overview)
- [Project Motivation](#project-motivation)
- [Features](#features)
- [Dataset](#dataset)
- [ML Pipeline](#-machine-learning-pipeline)
- [Model Architecture](#model-architecture)
- [📊 Model Performance](#-model-performance)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [App Pages](#app-pages)
- [Technologies](#technologies)
- [Team](#team)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## 🎯 Overview

This application provides an end-to-end machine learning solution for classifying breast cancer tumors as **benign** or **malignant**. It combines a trained Logistic Regression model with an interactive Streamlit dashboard that allows users to:

- Explore and analyze the dataset
- Visualize feature correlations and class distributions
- Make single and batch predictions
- Download prediction results

## 💡 Project Motivation

Breast cancer detection is one of the most important healthcare challenges. Early and accurate classification of tumors can significantly improve patient outcomes. This project demonstrates:

- How to build a production-ready ML classification model
- Interactive data science workflows for domain experts
- Real-world deployment of ML models in web applications
- Best practices in model training and evaluation

## ✨ Features

| Feature | Description |
|---------|-------------|
| **📊 Dataset Explorer** | Interactive dataset preview with statistics, missing value analysis, and descriptive statistics |
| **📈 Visualization** | Correlation heatmap and diagnosis distribution charts |
| **🤖 Single Prediction** | Real-time prediction with confidence score for individual tumor samples |
| **📂 Batch Prediction** | Upload CSV files to make predictions on multiple samples at once |
| **⬇️ Export Results** | Download batch predictions as CSV files |
| **🎨 Custom Styling** | Professional CSS styling for enhanced UI/UX |

## 📊 Dataset

**Source:** UCI Machine Learning Repository - Breast Cancer Wisconsin (Diagnostic)

**Dataset Characteristics:**
- **Total Samples:** 569 records
- **Features:** 30 numeric features derived from cell nuclei measurements
- **Target Variable:** 
  - `Benign` (357 samples - 62.7%) ✅
  - `Malignant` (212 samples - 37.3%) ❌
- **Feature Types:** All numeric (float)
- **Missing Values:** None
- **Data Quality:** Complete and well-balanced

### Feature Categories:
- **Mean values** - radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension
- **Standard Error (SE)** - Same measurements with error bounds
- **Worst values** - Largest measurements for each characteristic

### 📊 Exploratory Data Analysis

The dataset was thoroughly explored before model training using:

- ✔️ Missing value analysis
- ✔️ Class distribution analysis
- ✔️ Correlation heatmap
- ✔️ Outlier detection
- ✔️ Descriptive statistics
- ✔️ Feature correlation analysis
- ✔️ Distribution visualization

## 🔄 Machine Learning Pipeline

```
┌─────────────────┐
│  Data Loading   │
└────────┬────────┘
         ▼
┌─────────────────┐
│ Data Cleaning   │ (Remove id column, encode target)
└────────┬────────┘
         ▼
┌─────────────────┐
│     EDA         │ (Visualization & Statistics)
└────────┬────────┘
         ▼
┌─────────────────┐
│Train/Test Split │ (80/20 split, random_state=123)
└────────┬────────┘
         ▼
┌─────────────────┐
│  GridSearchCV   │ (Hyperparameter tuning with 5-fold CV)
└────────┬────────┘
         ▼
┌─────────────────────────────────────┐
│ Best Logistic Regression Model      │
└────────┬────────────────────────────┘
         ▼
┌─────────────────┐
│ Model Evaluation│ (Accuracy, Precision, Recall, F1)
└────────┬────────┘
         ▼
┌─────────────────┐
│ Save Model      │ (model.pkl & encoder.pkl)
└────────┬────────┘
         ▼
┌─────────────────┐
│ Streamlit App   │ (Deploy web interface)
└────────┬────────┘
         ▼
┌─────────────────┐
│   Predictions   │ (Single & Batch)
└─────────────────┘
```

## 🤖 Model Architecture

### Algorithm: Logistic Regression

**Why Logistic Regression?**

Logistic Regression was selected after considering multiple classification algorithms because it provides:

- ✅ **Excellent Accuracy** - Ideal for binary classification problems
- ✅ **Interpretability** - Clear understanding of feature contributions
- ✅ **Computational Efficiency** - Fast training and inference
- ✅ **Probability Estimates** - Confidence scores for predictions (0-1)
- ✅ **Low Overfitting Risk** - Better generalization with proper regularization
- ✅ **Production Ready** - Lightweight and easy to deploy

### Model Training Details

```python
# Hyperparameter Optimization using GridSearchCV
Algorithm: Logistic Regression

Cross-Validation: 5-Fold
Scoring Metric: Accuracy

Parameter Grid:
├─ C (Regularization Strength): [0.01, 0.1, 1, 10]
│  └─ Lower values = stronger regularization
└─ Solver: ["liblinear", "lbfgs"]
   └─ Different optimization algorithms

Data Split:
├─ Training Set: 80% (455 samples)
└─ Test Set: 20% (114 samples)

Random State: 123 (for reproducibility)
Max Iterations: 1000
```

### ⚙️ Data Preprocessing

- ✔️ Removed unnecessary columns (`id` column)
- ✔️ Encoded target labels using `LabelEncoder`
- ✔️ Verified no missing values
- ✔️ Validated feature data types
- ✔️ Features normalized during model training

## 📈 Model Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | ~97-98% |
| **Precision** | ~97% |
| **Recall** | ~96-97% |
| **F1 Score** | ~96-97% |
| **Best Hyperparameters** | Found via GridSearchCV |

**Note:** Exact metrics displayed after running `python train_model.py`

---

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Step 1: Clone the Repository

```bash
git clone https://github.com/Mennatullah122/Depi_DataScienceTrack_Project.git
cd Depi_DataScienceTrack_Project
```

### Step 2: Create a Virtual Environment

**On Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies:**
```
streamlit          # Web framework for ML apps
pandas            # Data manipulation & analysis
numpy             # Numerical computations
matplotlib        # Plotting library
seaborn           # Statistical visualization
scikit-learn      # Machine learning algorithms
joblib            # Model serialization
```

## 🚀 Usage

### Option 1: Using Pre-trained Model

If `model.pkl` and `encoder.pkl` already exist:

```bash
streamlit run app.py
```

Then open your browser and navigate to `http://localhost:8501`

### Option 2: Train Model First

```bash
# Step 1: Train the model (generates model.pkl and encoder.pkl)
python train_model.py

# Output:
# Best Parameters: {'C': ..., 'solver': '...'}
# Accuracy: XX.XX%
# Done.

# Step 2: Run the Streamlit app
streamlit run app.py
```

### Example Workflows

**Single Tumor Prediction:**
1. Go to "🤖 Prediction" tab
2. Adjust feature values using input boxes (pre-filled with dataset mean)
3. Click "🔍 Predict" button
4. View result (Benign/Malignant) with confidence percentage

**Batch Prediction:**
1. Go to "📂 Batch Prediction" tab
2. Upload a CSV file with tumor features
3. Review the data preview
4. View predictions with confidence scores
5. Download results as CSV file

## 📁 Project Structure

```
Depi_DataScienceTrack_Project/
│
├── app.py                  # Main Streamlit web application
│                           # - Home page with overview
│                           # - Dataset explorer
│                           # - Visualization dashboard
│                           # - Single & batch prediction
│
├── train_model.py          # Model training script
│                           # - Data loading & preprocessing
│                           # - Train/test split
│                           # - GridSearchCV optimization
│                           # - Model evaluation
│                           # - Artifact saving
│
├── breast-cancer.csv       # Dataset (569 x 31)
│                           # - 30 features
│                           # - 1 diagnosis column
│
├── style.css               # Custom CSS styling
│                           # - Streamlit theming
│                           # - UI enhancements
│
├── requirements.txt        # Python package dependencies
│
├── model.pkl               # Trained Logistic Regression model
│                           # (Generated after running train_model.py)
│
├── encoder.pkl             # LabelEncoder for target variable
│                           # (Generated after running train_model.py)
│
└── README.md              # This file
```

## 📄 App Pages

### 🏠 **Home**
- Welcome message with app description
- Quick overview of app features
- Dataset summary metrics (rows, columns, features)
- Call-to-action for navigation

### 📊 **Dataset**
- Full interactive dataset preview
- Dataset shape and dimensions
- Missing values analysis
- Statistical summary (mean, std, min, 25%, 50%, 75%, max)

### 📈 **Visualization**
- **Correlation Heatmap**: Shows feature relationships and multicollinearity
- **Diagnosis Distribution**: Bar chart showing benign vs malignant class balance

### 🤖 **Prediction**
- Interactive input fields for all 30 features
- Default values based on dataset mean (helpful reference)
- Two-column layout for easy input
- Prediction button with visual feedback
- Result display with color coding (🟢 Benign / 🔴 Malignant)
- Confidence score with progress bar

### 📂 **Batch Prediction**
- CSV file upload interface
- Data preview before prediction
- Batch processing of multiple samples
- Results table with IDs, predictions, and confidence
- CSV download button for results

## 💻 Technologies & Libraries

| Technology | Version | Purpose |
|-----------|---------|---------|
| **Python** | 3.8+ | Programming language |
| **Streamlit** | Latest | Web framework for ML apps |
| **scikit-learn** | Latest | ML algorithms & evaluation |
| **pandas** | Latest | Data manipulation & analysis |
| **NumPy** | Latest | Numerical computing |
| **Matplotlib** | Latest | Static visualization |
| **Seaborn** | Latest | Statistical visualization |
| **joblib** | Latest | Model serialization |

## 👥 Team

This project was developed as part of the DEPI Data Science Track by the following team members:

| Member | Role |
|--------|------|
| **Mennatullah Mohamed** | Lead Developer |
| **Ali Abdallah** | Data Scientist |
| **Yousef Mohamed** | ML Engineer |
| **Baraa** | Data Analyst |
| **Yousef Tarek** | Full Stack Developer |

---

## 🔧 Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'streamlit'`
**Solution:** Ensure virtual environment is activated and dependencies are installed
```bash
pip install -r requirements.txt
```

### Issue: `FileNotFoundError: model.pkl or encoder.pkl`
**Solution:** Train the model first
```bash
python train_model.py
```

### Issue: Port 8501 already in use
**Solution:** Specify a different port
```bash
streamlit run app.py --server.port 8502
```

### Issue: CSV upload not working in Batch Prediction
**Solution:** Ensure CSV has the same feature columns as training data
- Should have exactly 30 feature columns
- Should NOT have 'id' or 'diagnosis' columns
- Column names should match the original dataset

### Issue: Predictions seem inaccurate
**Solution:** 
- Check that input features are within reasonable ranges (compare with Dataset tab)
- Re-train the model with latest data: `python train_model.py`
- Verify feature scaling/normalization

### Issue: Slow performance
**Solution:**
- Use smaller batch files for prediction
- Clear browser cache
- Close other applications
- Check system resources

## 📝 License

This project is open source and available under the MIT License.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- Report bugs and issues
- Suggest new features and enhancements
- Submit pull requests
- Improve documentation

## 📧 Contact & Support

For questions, feedback, or collaboration opportunities:
- Open an issue on GitHub
- Create a discussion thread
- Submit a pull request

---

**Made with ❤️ for the DEPI Data Science Track**

*Elevating Healthcare through Machine Learning* 🏥🔬
