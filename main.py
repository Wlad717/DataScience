from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

app = FastAPI()

# Загружаем модель и препроцессор
try:
    model = joblib.load("models/best_model.pkl")
    preprocessor = joblib.load("models/preprocessor.pkl")
except Exception as e:
    raise RuntimeError(f"Ошибка загрузки модели или препроцессора: {e}")

class InputData(BaseModel):
    City: str
    CO: float
    SO2: float
    O3: float
    PM10: float
    Month: int
    Day: int
    Hour: int
    Day_of_Year: int

@app.post("/predict")
def predict(data: InputData):
    try:
        # Преобразуем входные данные в DataFrame
        input_df = pd.DataFrame([data.dict()])
        
        # Применяем препроцессинг
        processed_data = preprocessor.transform(input_df)  # Теперь preprocessor обучен
        
        # Получаем предсказание
        prediction = model.predict(processed_data)
        return {"predicted_AQI": float(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))