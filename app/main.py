import json
import os
import warnings
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

warnings.filterwarnings("ignore", category=UserWarning)

app = FastAPI(title="Medical Diagnosis API")

# ------------------ Paths ------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

MODEL_PATH = os.path.join(ROOT_DIR, "model.pkl")
SCALER_PATH = os.path.join(ROOT_DIR, "scaler.pkl")
LABEL_ENCODER_PATH = os.path.join(ROOT_DIR, "label_encoder.pkl")

# ------------------ Load Model ------------------
model = joblib.load(MODEL_PATH)         # MACHINE LEARNING MODEL
scaler = joblib.load(SCALER_PATH)       # FEATURE SCALER
label_enc = joblib.load(LABEL_ENCODER_PATH)  # LABEL ENCODER

# ------------------ Database ------------------
DB_URL = (
    f"mysql+pymysql://{os.environ.get('DB_USERNAME')}:"
    f"{os.environ.get('DB_PASSWORD')}@"
    f"{os.environ.get('DB_HOST')}:"
    f"{os.environ.get('DB_PORT')}/"
    f"{os.environ.get('DB_DATABASE')}"
)

engine = create_engine(DB_URL)

# ------------------ Pydantic Models ------------------
class Symptom(BaseModel):
    name: str
    present: Optional[bool] = True
    severity: Optional[str] = "Unknown"

class PatientRecord(BaseModel):
    patient_id: int
    age: int
    gender: str
    temperature: float
    symptoms: List[Symptom] = []

# ------------------ Prediction Endpoint ------------------
@app.post("/predict")
def predict_records():

    try:
        query = """
            SELECT 
                mr.id,
                TIMESTAMPDIFF(YEAR, p.date_of_birth, CURDATE()) AS Age,
                LOWER(p.gender) AS Gender,
                mr.temperature AS temperature,
                mr.symptoms_data AS symptoms
            FROM medical_records mr
            JOIN patients p ON mr.patient_id = p.patient_id
        """

        df = pd.read_sql(query, engine)
        if df.empty:
            return {"info": "No data available."}

        # Normalize gender
        df['Gender'] = df['Gender'].map({
            'male': 1, 'm': 1,
            'female': 0, 'f': 0
        }).fillna(0)

        # Load disease map
        disease_map = pd.read_sql("SELECT disease_id, disease_name FROM disease", engine)
        disease_name_to_id = dict(
            zip(disease_map['disease_name'].str.lower(), disease_map['disease_id'])
        )

        expected_features = [
            'Age', 'Gender', 'temperature', 'nausea', 'joint_pain', 'abdominal_pain',
            'high_fever', 'chills', 'fatigue', 'runny_nose', 'pain_behind_the_eyes',
            'dizziness','headache','chest_pain','vomiting','cough','shivering','asthma_history',
            'high_cholesterol','diabetes','obesity','hiv_aids','nasal_polyps','asthma',
            'high_blood_pressure','severe_headache','weakness','trouble_seeing','fever','body_aches',
            'sore_throat','sneezing','diarrhea','rapid_breathing','rapid_heart_rate','pain_behind_eyes',
            'swollen_glands','rashes','sinus_headache','facial_pain','shortness_of_breath',
            'reduced_smell_and_taste','skin_irritation','itchiness','throbbing_headache','confusion',
            'back_pain','knee_ache'
        ]

        # Initialize all symptom columns to zero
        for f in expected_features[3:]:
            df[f] = 0

        MIN_SYMPTOMS_REQUIRED = 2
        symptom_counts = []

        for idx, row in df.iterrows():
            try:
                symptoms = json.loads(row['symptoms'])
                count = 0
                for s in symptoms:
                    name = s.get('name', '').lower().replace(' ', '_')
                    if name in expected_features:
                        df.at[idx, name] = 1
                        count += 1
                symptom_counts.append(count)
            except Exception:
                symptom_counts.append(0)

        df['symptom_count'] = symptom_counts

        # Only rows with enough symptoms
        mask = df['symptom_count'] >= MIN_SYMPTOMS_REQUIRED
        if not mask.any():
            return {"info": "Not enough symptoms for prediction."}

        features_df = df.loc[mask, expected_features]
        scaled_features = scaler.transform(features_df)

        predictions = model.predict(scaled_features)
        predicted_diseases = label_enc.inverse_transform(predictions)

        # Save results to DB
        with engine.begin() as conn:
            for db_id, disease_name in zip(df.loc[mask, 'id'], predicted_diseases):
                disease_id = disease_name_to_id.get(disease_name.lower())
                if disease_id:
                    conn.execute(
                        text("UPDATE medical_records SET disease_id=:disease_id, diagnosis=:diagnosis WHERE id=:id"),
                        {"disease_id": int(disease_id), "diagnosis": disease_name, "id": int(db_id)}
                    )

        return {
            "info": "Predictions updated successfully",
            "count": len(predicted_diseases)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
