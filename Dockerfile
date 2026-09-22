# Usar una imagen ligera de Python
FROM python:3.11-slim

# Instalar dependencias del sistema requeridas por OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Establecer el directorio de trabajo en el contenedor
WORKDIR /app

# Copiar el archivo de dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar todo el código fuente y el modelo entrenado (best.pt)
COPY . .

# Exponer el puerto estándar de FastAPI
EXPOSE 8000

# Comando para arrancar el servidor en la nube
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]