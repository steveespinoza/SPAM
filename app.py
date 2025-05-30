from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib

# Cargar modelo y vectorizador
model = joblib.load("model/logistic_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

# Crear app
app = FastAPI()

# Configurar CORS (esto es clave)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ⚠️ O especifica ["http://127.0.0.1:5500"]
    allow_credentials=True,
    allow_methods=["*"],  # Incluye OPTIONS, GET, POST, etc.
    allow_headers=["*"]
)

# Esquema de entrada
class Message(BaseModel):
    text: str

@app.post("/predict")
def predict_spam(message: Message):
    vect_text = vectorizer.transform([message.text])
    prediction = model.predict(vect_text)
    return {"prediction": int(prediction[0])}
