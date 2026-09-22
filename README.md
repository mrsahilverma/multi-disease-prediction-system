# Multi-Disease Prediction System

A Flask-based machine learning web application for predicting the risk of four diseases:

- Heart Disease
- Diabetes
- Kidney Disease
- Liver Disease

The project provides separate prediction forms for each disease and uses trained machine learning models to generate predictions through a simple web interface.

> **Medical Disclaimer:** This project is intended for educational and demonstration purposes only. Its predictions are not medical diagnoses and should not be used as a substitute for professional medical advice, examination, or treatment.

## Project Overview

The system combines machine learning models with a Flask web application. Users enter health-related information through a web form, the application processes the input, loads the corresponding trained model, and displays the prediction result.

### Supported Diseases

| Disease | Dataset | Model |
|---|---|---|
| Heart Disease | Heart Disease dataset | `heart_model.pkl` |
| Diabetes | Pima Indians Diabetes Database | `diabetes_model.pkl` |
| Kidney Disease | Chronic Kidney Disease dataset | `kidney_model.pkl` |
| Liver Disease | Indian Liver Patient Dataset (ILPD) | `liver_model.pkl` |

## Project Structure

```text
multi-disease-prediction-system/
│
├── app.py
├── requirements.txt
├── .gitignore
│
├── datasets/
│   ├── diabetes.csv
│   ├── heart.csv
│   ├── kidney_disease.csv
│   └── liver_disease.csv
│
├── models/
│   ├── diabetes_model.pkl
│   ├── heart_model.pkl
│   ├── kidney_model.pkl
│   └── liver_model.pkl
│
├── notebooks/
│   ├── diabetes.ipynb
│   ├── heart.ipynb
│   ├── kidney.ipynb
│   └── liver.ipynb
│
└── templates/
    ├── base.html
    ├── index.html
    ├── diabetes.html
    ├── heart.html
    ├── kidney.html
    ├── liver.html
    └── result.html
```

## Technology Stack

- **Python 3.12**
- **Flask** — web application framework
- **Pandas** — data manipulation
- **NumPy** — numerical computing
- **Scikit-learn 1.5.1** — machine learning
- **XGBoost** — gradient boosting models
- **Seaborn** — data visualization
- **Joblib / Pickle** — model serialization
- **Jupyter Notebook** — model development and analysis
- **HTML/CSS** — web interface

## Machine Learning Workflow

The project follows a typical machine learning workflow:

```text
Dataset
   ↓
Data Loading
   ↓
Data Cleaning / Preprocessing
   ↓
Exploratory Data Analysis
   ↓
Feature Preparation
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Model Serialization
   ↓
Flask Application
   ↓
User Input
   ↓
Prediction
   ↓
Result
```

The individual notebooks contain the disease-specific data analysis and model development work.

## Running the Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/mrsahilverma/multi-disease-prediction-system.git
cd multi-disease-prediction-system
```

### 2. Create and activate a Python environment

Using Conda:

```bash
conda create -n multi_disease python=3.12
conda activate multi_disease
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the Flask application

```bash
python app.py
```

Then open the local address shown by Flask in your browser.

## Dataset Sources

The project uses publicly available datasets associated with Kaggle and the UCI Machine Learning Repository.

### Diabetes

**Pima Indians Diabetes Database**

The dataset contains medical diagnostic measurements used for diabetes prediction and is commonly distributed through Kaggle.

### Heart Disease

**Heart Disease Dataset**

The `heart.csv` file used by this project contains 1,025 records and 14 columns and is a Kaggle-distributed heart disease dataset based on the commonly used UCI heart disease data.

### Kidney Disease

**Chronic Kidney Disease Dataset**

Original source:

UCI Machine Learning Repository — Chronic Kidney Disease Dataset

https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease

### Liver Disease

**Indian Liver Patient Dataset (ILPD)**

Original source:

UCI Machine Learning Repository — ILPD

https://archive.ics.uci.edu/dataset/225/ilpd+indian+liver+patient+dataset

The UCI listing identifies the dataset with a **CC BY 4.0** license. Please retain appropriate attribution when redistributing the dataset.

## Model Outputs

The application displays a prediction result and, where supported by the trained model, a probability/confidence value returned by `predict_proba()`.

These values represent the model's output for the supplied input. They **must not be interpreted as medical certainty or as model accuracy**.

## Notebooks

The `notebooks/` directory contains separate notebooks for:

- Diabetes data analysis and model development
- Heart disease data analysis and model development
- Kidney disease data analysis and model development
- Liver disease data analysis and model development

The notebooks can be opened with Jupyter Notebook or JupyterLab.

## Requirements

The main dependencies are listed in `requirements.txt`.


## Future Improvements

Possible future improvements include:

- Improved model evaluation and comparison
- Hyperparameter tuning
- More robust input validation
- Improved UI/UX
- Model explainability
- Additional disease prediction models
- Deployment to a cloud platform
- Automated testing
- Better handling of missing and invalid input values

## Author

**Sahil Verma**

- GitHub: https://github.com/mrsahilverma
- LinkedIn: https://www.linkedin.com/in/mrsahilverma/

## License

No project license has currently been specified.

If you plan to distribute or reuse this project, review the licenses and attribution requirements of the individual datasets and dependencies before adding a project-level license.
