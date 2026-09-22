from fastapi import FastAPI, UploadFile, File, HTTPException
import cv2
import numpy as np
from ultralytics import YOLO
import shutil
import os

# Instancia principal que busca Uvicorn
app = FastAPI(title="PorciTech AI Backend", version="1.0")

# Cargar el modelo de IA base (yolov8n-seg.pt)
print("Cargando modelo de IA...")
model = YOLO('best.pt')

UPLOAD_DIR = "uploads_temp"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Factor de conversión temporal (luego se usará XGBoost y ArUco)
FACTOR_PIXELES_A_KILOS = 0.0005 

@app.post("/api/v1/predecir-peso")
async def predecir_peso_imagen(file: UploadFile = File(...)):
    try:
        # 1. Guardar temporalmente la imagen recibida
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # 2. Leer la imagen con OpenCV
        frame = cv2.imread(file_path)
        if frame is None:
            raise HTTPException(status_code=400, detail="No se pudo leer la imagen enviada.")

        # 3. Ejecutar inferencia con YOLO (segmentación)
        resultados = model(frame, verbose=False)
        
        peso_estimado = 0.0
        area_detectada = 0.0
        encontrado = False

        for resultado in resultados:
            if resultado.masks is not None:
                for mask in resultado.masks.xy:
                    contorno = np.array(mask, dtype=np.int32)
                    area_pixeles = cv2.contourArea(contorno)
                    
                    # Filtro de área (puedes ajustarlo si el objeto detectado es muy pequeño)
                    if area_pixeles > 500:
                        area_detectada = float(area_pixeles)
                        peso_estimado = float(area_pixeles * FACTOR_PIXELES_A_KILOS)
                        encontrado = True
                        break # Tomamos la primera detección válida

        # 4. Limpiar el archivo temporal del servidor
        if os.path.exists(file_path):
            os.remove(file_path)

        if not encontrado:
            return {
                "status": "warning", 
                "mensaje": "Se procesó la imagen, pero no se detectaron siluetas válidas.",
                "peso_estimado_kg": 0.0,
                "area_pixeles": 0.0
            }

        # 5. Retornar el resultado real hacia tu frontend
        return {
            "status": "success",
            "peso_estimado_kg": round(peso_estimado, 2),
            "area_pixeles": area_detectada
        }

    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))