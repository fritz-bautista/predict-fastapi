import os
import json
import joblib
import warnings
import pandas as pd
from fastapi import FastAPI, HTTPException
from sqlalchemy import create_engine, text
from typing import List, Optional
from pydantic import BaseModel
from dotenv import load_dotenv

# ---------------- Load environment variables ----------------
load_dotenv()

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------- FastAPI instance ----------------
app = FastAPI(title="Medical Diagnosis API")

# ---------------- Database ----------------
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_DATABASE")
DB_USER = os.getenv("DB_USERNAME")
DB_PASS = os.getenv("DB_PASSWORD")

DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DATABASE_URL)

# ---------------- Load ML Models ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

model = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
label_enc = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))

# ---------------- Request Schema ----------------
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

# ---------------- Prediction Endpoint ----------------
@app.post("/predict")
def predict_records():
    try:
        # Load records from DB
        query = """
            SELECT mr.id,
                   TIMESTAMPDIFF(YEAR, p.date_of_birth, CURDATE()) AS Age,
                   LOWER(p.gender) AS Gender,
                   mr.temperature AS temperature,
                   mr.symptoms_data AS symptoms
            FROM medical_records mr
            JOIN patients p ON mr.patient_id = p.patient_id
        """
        user_df = pd.read_sql(query, engine)
        if user_df.empty:
            return {"info": "No data available for prediction."}

        # Map gender
        user_df['Gender'] = user_df['Gender'].map({'male': 1, 'm': 1, 'female': 0, 'f': 0}).fillna(0)

        # Disease mapping
        disease_map = pd.read_sql("SELECT disease_id, disease_name FROM disease", engine)
        disease_name_to_id = dict(zip(disease_map['disease_name'].str.lower(), disease_map['disease_id']))

        # Features
        expected_features = ['Age','Gender','temperature','nausea','joint_pain','abdominal_pain','high_fever','chills','fatigue',
            'runny_nose','pain_behind_the_eyes','dizziness','headache','chest_pain','vomiting','cough','shivering',
            'asthma_history','high_cholesterol','diabetes','obesity','hiv_aids','nasal_polyps','asthma','high_blood_pressure',
            'severe_headache','weakness','trouble_seeing','fever','body_aches','sore_throat','sneezing','diarrhea',
            'rapid_breathing','rapid_heart_rate','pain_behind_eyes','swollen_glands','rashes','sinus_headache','facial_pain',
            'shortness_of_breath','reduced_smell_and_taste','skin_irritation','itchiness','throbbing_headache','confusion',
            'back_pain','knee_ache']

        # Initialize symptom columns
        for feature in expected_features[3:]:
            user_df[feature] = 0

        # Parse symptoms_data
        MIN_SYMPTOMS_REQUIRED = 2
        symptom_counts = []

        for idx, row in user_df.iterrows():
            try:
                symptoms = json.loads(row['symptoms'])
                count = 0
                for symptom in symptoms:
                    name = symptom.get('name','').lower().replace(' ','_')
                    if name in expected_features:
                        user_df.at[idx,name] = 1
                        count += 1
                symptom_counts.append(count)
            except Exception:
                symptom_counts.append(0)
        user_df['symptom_count'] = symptom_counts

        # Predict
        complete_mask = user_df['symptom_count'] >= MIN_SYMPTOMS_REQUIRED
        if not complete_mask.any():
            return {"info": "No rows have enough symptoms for prediction."}

        features_df = user_df.loc[complete_mask, expected_features]
        scaled_features = scaler.transform(features_df)
        predictions = model.predict(scaled_features)
        predicted_diseases = label_enc.inverse_transform(predictions)

        # Update DB
        with engine.begin() as conn:
            for db_id, disease_name in zip(user_df.loc[complete_mask, 'id'], predicted_diseases):
                disease_id = disease_name_to_id.get(disease_name.lower())
                if disease_id:
                    conn.execute(
                        text("UPDATE medical_records SET disease_id = :disease_id, diagnosis = :diagnosis WHERE id = :id"),
                        {"disease_id": int(disease_id), "diagnosis": disease_name, "id": int(db_id)}
                    )

        return {"info": "Predictions updated successfully", "predicted_count": len(predicted_diseases)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
