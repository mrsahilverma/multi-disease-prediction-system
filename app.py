from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load all models
MODEL_PATH = 'models/'

def load_model(model_name):
    """Load a pickle model file"""
    try:
        with open(os.path.join(MODEL_PATH, model_name), 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading {model_name}: {str(e)}")
        return None

# Load all models at startup
heart_model = load_model('heart_model.pkl')
diabetes_model = load_model('diabetes_model.pkl')
kidney_model = load_model('kidney_model.pkl')
liver_model = load_model('liver_model.pkl')

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/heart')
def heart():
    """Heart disease prediction form"""
    return render_template('heart.html')

@app.route('/diabetes')
def diabetes():
    """Diabetes prediction form"""
    return render_template('diabetes.html')

@app.route('/kidney')
def kidney():
    """Kidney disease prediction form"""
    return render_template('kidney.html')

@app.route('/liver')
def liver():
    """Liver disease prediction form"""
    return render_template('liver.html')

@app.route('/predict/heart', methods=['POST'])
def predict_heart():
    """Predict heart disease"""
    try:
        
        age = int(request.form['age'])
        sex = int(request.form['sex'])
        cp = int(request.form['cp'])
        trestbps = int(request.form['trestbps'])
        chol = int(request.form['chol'])
        fbs = int(request.form['fbs'])
        restecg = int(request.form['restecg'])
        thalach = int(request.form['thalach'])
        exang = int(request.form['exang'])
        oldpeak = float(request.form['oldpeak'])
        slope = int(request.form['slope'])
        ca = int(request.form['ca'])
        thal = int(request.form['thal'])
        
        # Create feature array
        features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]])
        
        # Make prediction
        prediction = heart_model.predict(features)[0]
        probability = heart_model.predict_proba(features)[0] if hasattr(heart_model, 'predict_proba') else None
        
        result = {
            'disease': 'Heart Disease',
            'prediction': int(prediction),
            'result': 'Positive' if prediction == 1 else 'Negative',
            'probability': float(probability[1]) * 100 if probability is not None else None
        }
        
        return render_template('result.html', result=result)
    
    except Exception as e:
        return render_template('result.html', error=str(e))

@app.route('/predict/diabetes', methods=['POST'])
def predict_diabetes():
    """Predict diabetes"""
    try:
        # Extract basic form data
        pregnancies = int(request.form['pregnancies'])
        glucose = float(request.form['glucose'])
        blood_pressure = float(request.form['blood_pressure'])
        skin_thickness = float(request.form['skin_thickness'])
        insulin = float(request.form['insulin'])
        bmi = float(request.form['bmi'])
        diabetes_pedigree = float(request.form['diabetes_pedigree'])
        age = int(request.form['age'])
        
        # Calculate derived features based on BMI categories
        newbmi_obesity1 = 1 if 30 <= bmi < 35 else 0
        newbmi_obesity2 = 1 if 35 <= bmi < 40 else 0
        newbmi_obesity3 = 1 if bmi >= 40 else 0
        newbmi_overweight = 1 if 25 <= bmi < 30 else 0
        newbmi_underweight = 1 if bmi < 18.5 else 0
        
        # Calculate insulin score
        newinsulin_normal = 1 if 16 <= insulin <= 166 else 0
        
        # Calculate glucose categories
        newglucose_low = 1 if glucose <= 70 else 0
        newglucose_normal = 1 if 70 < glucose <= 99 else 0
        newglucose_overweight = 1 if 100 <= glucose <= 125 else 0
        newglucose_secret = 1 if glucose > 125 else 0
        
        # Create feature array (19 features total)
        features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, diabetes_pedigree, age,
                            0,  # Outcome placeholder
                            newbmi_obesity1, newbmi_obesity2, newbmi_obesity3,
                            newbmi_overweight, newbmi_underweight,
                            newinsulin_normal, newglucose_low, newglucose_normal,
                            newglucose_overweight, newglucose_secret]])
        
        # Remove outcome column if model expects 18 features
        features = np.delete(features, 8, axis=1)
        
        # Make prediction
        prediction = diabetes_model.predict(features)[0]
        probability = diabetes_model.predict_proba(features)[0] if hasattr(diabetes_model, 'predict_proba') else None
        
        result = {
            'disease': 'Diabetes',
            'prediction': int(prediction),
            'result': 'Positive' if prediction == 1 else 'Negative',
            'probability': float(probability[1]) * 100 if probability is not None else None
        }
        
        return render_template('result.html', result=result)
    
    except Exception as e:
        return render_template('result.html', error=str(e))

@app.route('/predict/kidney', methods=['POST'])
def predict_kidney():
    """Predict kidney disease"""
    try:
        # Extract all 24 features
        age = float(request.form['age'])
        blood_pressure = float(request.form['blood_pressure'])
        specific_gravity = float(request.form['specific_gravity'])
        albumin = float(request.form['albumin'])
        sugar = float(request.form['sugar'])
        red_blood_cells = int(request.form['red_blood_cells'])
        pus_cell = int(request.form['pus_cell'])
        pus_cell_clumps = int(request.form['pus_cell_clumps'])
        bacteria = int(request.form['bacteria'])
        blood_glucose_random = float(request.form['blood_glucose_random'])
        blood_urea = float(request.form['blood_urea'])
        serum_creatinine = float(request.form['serum_creatinine'])
        sodium = float(request.form['sodium'])
        potassium = float(request.form['potassium'])
        haemoglobin = float(request.form['haemoglobin'])
        packed_cell_volume = float(request.form['packed_cell_volume'])
        white_blood_cell_count = float(request.form['white_blood_cell_count'])
        red_blood_cell_count = float(request.form['red_blood_cell_count'])
        hypertension = int(request.form['hypertension'])
        diabetes_mellitus = int(request.form['diabetes_mellitus'])
        coronary_artery_disease = int(request.form['coronary_artery_disease'])
        appetite = int(request.form['appetite'])
        peda_edema = int(request.form['peda_edema'])
        aanemia = int(request.form['aanemia'])
        
        # Create feature array (24 features, excluding class)
        features = np.array([[age, blood_pressure, specific_gravity, albumin, sugar,
                            red_blood_cells, pus_cell, pus_cell_clumps, bacteria,
                            blood_glucose_random, blood_urea, serum_creatinine,
                            sodium, potassium, haemoglobin, packed_cell_volume,
                            white_blood_cell_count, red_blood_cell_count,
                            hypertension, diabetes_mellitus, coronary_artery_disease,
                            appetite, peda_edema, aanemia]])
        
        # Make prediction
        prediction = kidney_model.predict(features)[0]
        probability = kidney_model.predict_proba(features)[0] if hasattr(kidney_model, 'predict_proba') else None
        
        result = {
            'disease': 'Kidney Disease',
            'prediction': int(prediction),
            'result': 'Positive' if prediction == 1 else 'Negative',
            'probability': float(probability[1]) * 100 if probability is not None else None
        }
        
        return render_template('result.html', result=result)
    
    except Exception as e:
        return render_template('result.html', error=str(e))

@app.route('/predict/liver', methods=['POST'])
def predict_liver():
    """Predict liver disease"""
    try:
        # Extract form data
        age = int(request.form['age'])
        gender = int(request.form['gender'])
        total_bilirubin = float(request.form['total_bilirubin'])
        direct_bilirubin = float(request.form['direct_bilirubin'])
        alkaline_phosphotase = int(request.form['alkaline_phosphotase'])
        alamine_aminotransferase = int(request.form['alamine_aminotransferase'])
        aspartate_aminotransferase = int(request.form['aspartate_aminotransferase'])
        total_protiens = float(request.form['total_protiens'])
        albumin = float(request.form['albumin'])
        albumin_globulin_ratio = float(request.form['albumin_globulin_ratio'])
        
        # Create feature array
        features = np.array([[age, gender, total_bilirubin, direct_bilirubin,
                            alkaline_phosphotase, alamine_aminotransferase,
                            aspartate_aminotransferase, total_protiens,
                            albumin, albumin_globulin_ratio]])
        
        # Make prediction
        prediction = liver_model.predict(features)[0]
        probability = liver_model.predict_proba(features)[0] if hasattr(liver_model, 'predict_proba') else None
        
        result = {
            'disease': 'Liver Disease',
            'prediction': int(prediction),
            'result': 'Positive' if prediction == 1 else 'Negative',
            'probability': float(probability[1]) * 100 if probability is not None else None
        }
        
        return render_template('result.html', result=result)
    
    except Exception as e:
        return render_template('result.html', error=str(e))

# Error handlers
@app.errorhandler(404)
def not_found(e):
    return render_template('index.html'), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('result.html', error="Internal server error occurred"), 500

if __name__ == '__main__':
    # Check if models exist
    models = ['heart_model.pkl', 'diabetes_model.pkl', 'kidney_model.pkl', 'liver_model.pkl']
    for model in models:
        if not os.path.exists(os.path.join(MODEL_PATH, model)):
            print(f"Warning: {model} not found in {MODEL_PATH}")
    
    app.run(debug=True, host='0.0.0.0', port=5000)