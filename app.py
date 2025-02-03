from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
import firebase_admin
from firebase_admin import credentials, firestore
from flask_cors import CORS
import google.generativeai as genai
import time
import datetime
import os
import json
import requests

# URLs de los archivos en Google Drive
PIPELINE_URL = "https://drive.google.com/uc?id=1rSxM2HEQiaDhij6h_iN2hSO3p35L8CGD"
MODEL_URL = "https://drive.google.com/uc?id=17yvOgBwqAWPXPAjtAhmwuuRl4WsirGse"

# Rutas donde se guardarán los archivos descargados en Render
PIPELINE_PATH = "/tmp/pipePreprocesadores.pickle"
MODEL_PATH = "/tmp/modeloRF.pickle"

def download_file(url, destination):
    response = requests.get(url)
    if response.status_code == 200:
        with open(destination, "wb") as file:
            file.write(response.content)
        print(f"✅ Archivo descargado correctamente: {destination}")
    else:
        print(f"❌ Error al descargar {destination}, código: {response.status_code}")

# Descargar los archivos al inicio
print("🔹 Descargando modelos...")
download_file(PIPELINE_URL, PIPELINE_PATH)
download_file(MODEL_URL, MODEL_PATH)

# Inicializar la aplicación Flask
app = Flask(__name__)
CORS(app)

# Configurar Firebase desde variable de entorno
firebase_credentials_json = os.environ.get("FIREBASE_CREDENTIALS")
if firebase_credentials_json:
    firebase_credentials = json.loads(firebase_credentials_json)
    cred = credentials.Certificate(firebase_credentials)
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("✅ Firebase inicializado correctamente.")
else:
    print("❌ No se encontró la variable de entorno FIREBASE_CREDENTIALS")

# Configurar la API de Google Gemini
api_key = os.environ.get("GOOGLE_GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
    print("✅ API de Google Gemini configurada correctamente.")
else:
    print("❌ No se encontró la variable de entorno GOOGLE_GEMINI_API_KEY")

# Cargar el modelo y el pipeline desde los archivos descargados
with open(MODEL_PATH, "rb") as model_file:
    model = pickle.load(model_file)
with open(PIPELINE_PATH, "rb") as pipeline_file:
    pipeline = pickle.load(pipeline_file)

print("✅ Modelo y pipeline cargados correctamente.")

@app.route('/')
def home():
    return "API para la predicción con modelo Random Forest funcionando correctamente."

# Función para limpiar el texto generado
def clean_explanation(text):
    text = text.replace("*", "").replace("`", "").replace("\n", " ")
    replacements = {
        "ProductRelated": "Número de páginas de producto visitadas",
        "ProductRelated_Duration": "Tiempo en páginas de producto (segundos)",
        "Administrative": "Interacción con secciones administrativas",
        "Informational": "Interacción con secciones informativas",
        "BounceRates": "Tasa de rebote",
        "ExitRates": "Tasa de salida",
        "PageValues": "Valor de las páginas visitadas",
        "SpecialDay": "Día especial (promociones o eventos)"
    }
    for key, value in replacements.items():
        text = text.replace(key, value)
    return text

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No se enviaron datos."}), 400

        input_data = pd.DataFrame([data])
        transformed_data = pipeline.transform(input_data)

        prediction = model.predict(transformed_data)
        prediction_proba = model.predict_proba(transformed_data)
        timestamp = datetime.datetime.utcnow().isoformat()

        prediction_result = {
            "input_data": data,
            "prediction": int(prediction[0]),
            "probability": prediction_proba[0].tolist(),
            "timestamp": timestamp
        }

        explanation_prompt = f"""
        Con base en los siguientes datos clave:
        - **Valor de las páginas visitadas**: {data.get("PageValues", "N/A")}
        - **Tasa de salida**: {data.get("ExitRates", "N/A")}
        - **Tiempo en páginas de producto**: {data.get("ProductRelated_Duration", "N/A")} horas
        - **Tipo de visitante**: {data.get("VisitorType", "N/A")}
        El modelo predijo que el usuario {'REALIZARÁ' if prediction_result["prediction"] == 1 else 'NO REALIZARÁ'} una compra.
        Explica esto de manera sencilla para el dueño del sitio de ventas.
        """

        try:
            model_gemini = genai.GenerativeModel("gemini-1.5-flash")
            response = model_gemini.generate_content(explanation_prompt)
            prediction_result["explanation"] = clean_explanation(response.text) if response.text else "Explicación no disponible."
        except Exception as ge:
            prediction_result["explanation"] = f"Explicación no disponible: {str(ge)}"

        doc_ref = db.collection("predictions").add(prediction_result)
        prediction_id = doc_ref[1].id

        return jsonify({
            "prediction": prediction_result["prediction"],
            "probability": prediction_result["probability"],
            "explanation": prediction_result["explanation"],
            "timestamp": timestamp,
            "predictionId": prediction_id
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/history', methods=['GET'])
def get_history():
    try:
        predictions_ref = db.collection("predictions")
        docs = predictions_ref.stream()
        history = [{"id": doc.id, **doc.to_dict()} for doc in docs]
        return jsonify({"history": history}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
